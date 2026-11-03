package org.koitharu.kotatsu.reader.translate

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Typeface
import android.net.Uri
import android.util.Base64
import android.text.Layout
import android.text.StaticLayout
import android.text.TextPaint
import androidx.core.net.toFile
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.nl.languageid.LanguageIdentification
import com.google.mlkit.nl.languageid.LanguageIdentificationOptions
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import org.koitharu.kotatsu.core.util.ext.isZipUri
import org.koitharu.kotatsu.parsers.util.runCatchingCancellable
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.Locale
import java.util.zip.ZipFile
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

data class TranslateConfig(
	val source: String, // "auto" or language code
	val target: String,
	val engine: String, // "mlkit" or "gemini"
	val geminiKey: String,
	val geminiModel: String,
) {
	val useGemini: Boolean get() = engine == "gemini" && geminiKey.isNotBlank()
	val cacheKey: String get() = "v3_${source}_${target}_${if (useGemini) "g" else "m"}"
}

/**
 * On-device manga page translator.
 *  1. OCR (ML Kit) finds text lines
 *  2. lines are merged into speech bubbles, so a bubble is translated as one sentence
 *  3. translation: ML Kit (offline) or Gemini (best quality, needs API key)
 *  4. original text is erased with the real bubble colour and the translation is typeset
 *     to fill the bubble (font size is fitted automatically)
 * Results are cached, every page is processed once.
 */
@Singleton
class MangaTranslator @Inject constructor(
	@ApplicationContext private val context: Context,
) {

	private val lock = Mutex() // one page at a time: OCR and big bitmaps are memory hungry
	private val cacheDir = File(context.cacheDir, "manga_translate").apply { mkdirs() }

	private val recognizers = HashMap<String, TextRecognizer>()
	private val translators = HashMap<String, Translator>()
	private val langId by lazy {
		LanguageIdentification.getClient(
			LanguageIdentificationOptions.Builder().setConfidenceThreshold(0.75f).build(),
		)
	}

	/** @return uri of the translated image, or [uri] itself if there was nothing to translate */
	suspend fun translate(uri: Uri, cfg: TranslateConfig): Uri {
		if (cfg.source == cfg.target) return uri
		val name = "${sha1(uri.toString())}_${cfg.cacheKey}"
		val out = File(cacheDir, "$name.jpg")
		val none = File(cacheDir, "$name.none")
		if (none.exists()) return uri
		if (out.exists() && out.length() > 0) return Uri.fromFile(out)
		return lock.withLock {
			if (out.exists() && out.length() > 0) return@withLock Uri.fromFile(out)
			val bitmap = withContext(Dispatchers.IO) { decode(uri) } ?: return@withLock uri
			try {
				val bubbles = group(recognize(bitmap, cfg.source))
				if (bubbles.isEmpty()) {
					none.createNewFile() // no text on this page, do not scan it again
					return@withLock uri
				}
				val outcome = translateBubbles(bubbles, cfg, bitmap)
				if (outcome.items.isEmpty()) return@withLock uri
				withContext(Dispatchers.Default) { render(bitmap, outcome.items) }
				// if something failed to translate the result is NOT cached, so the next open retries
				val dst = if (outcome.complete) out else File(cacheDir, "${name}_partial.jpg")
				withContext(Dispatchers.IO) {
					val tmp = File(cacheDir, name + ".tmp")
					tmp.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 92, it) }
					dst.delete()
					tmp.renameTo(dst)
					trimCache()
				}
				Uri.fromFile(dst)
			} finally {
				bitmap.recycle()
			}
		}
	}

	// ---------------------------------------------------------------- OCR

	private class TextLine(val rect: Rect, val text: String, val block: Int)

	private fun recognizer(key: String): TextRecognizer = recognizers.getOrPut(key) {
		TextRecognition.getClient(
			when (key) {
				"ja" -> JapaneseTextRecognizerOptions.Builder().build()
				"zh" -> ChineseTextRecognizerOptions.Builder().build()
				"ko" -> KoreanTextRecognizerOptions.Builder().build()
				"hi" -> DevanagariTextRecognizerOptions.Builder().build()
				else -> TextRecognizerOptions.DEFAULT_OPTIONS
			},
		)
	}

	private suspend fun recognize(bitmap: Bitmap, source: String): List<TextLine> {
		if (source != "auto") {
			return recognizeWith(if (source == "en") "latin" else source, bitmap)
		}
		val latin = recognizeWith("latin", bitmap)
		if (latin.size >= 3) return latin
		var best = latin
		var bestLen = latin.sumOf { it.text.length } * 1.5 // Latin wins unless another script reads clearly more
		for (key in listOf("ja", "ko", "zh", "hi")) {
			val r = recognizeWith(key, bitmap)
			val len = r.sumOf { it.text.length }
			if (len > bestLen) {
				best = r
				bestLen = len.toDouble()
			}
		}
		return best
	}

	/** Very tall pages (webtoons) are scanned in overlapping tiles. */
	private suspend fun recognizeWith(key: String, bitmap: Bitmap): List<TextLine> {
		val rec = recognizer(key)
		val tileH = 2400
		val overlap = 240
		val w = bitmap.width
		val h = bitmap.height
		val result = ArrayList<TextLine>()
		// small pages: upscale for OCR, ML Kit reads small text much better this way
		val scale = if (w < 1000) 1.5f else 1f
		var blockId = 0
		var y = 0
		while (y < h) {
			val end = min(h, y + tileH)
			val whole = y == 0 && end == h
			val crop = if (whole) bitmap else Bitmap.createBitmap(bitmap, 0, y, w, end - y)
			val tile = if (scale > 1f) {
				Bitmap.createScaledBitmap(crop, (crop.width * scale).toInt(), (crop.height * scale).toInt(), true)
			} else {
				crop
			}
			try {
				val text = rec.process(InputImage.fromBitmap(tile, 0)).await()
				val ownTop = if (y == 0) 0 else y + overlap / 2
				val ownBottom = if (end == h) h else end - overlap / 2
				for (block in text.textBlocks) {
					blockId++
					for (line in block.lines) {
						val box = line.boundingBox ?: continue
						val s = line.text.trim()
						if (s.isEmpty()) continue
						val rect = Rect(
							(box.left / scale).toInt(),
							(box.top / scale).toInt() + y,
							(box.right / scale).toInt(),
							(box.bottom / scale).toInt() + y,
						)
						if (rect.centerY() in ownTop until ownBottom) result.add(TextLine(rect, s, blockId))
					}
				}
			} finally {
				if (tile !== crop) tile.recycle()
				if (crop !== bitmap) crop.recycle()
			}
			if (end == h) break
			y = end - overlap
		}
		return result
	}

	// ---------------------------------------------------------------- bubbles

	private class Bubble(val lines: List<TextLine>) {
		val rect = Rect(lines[0].rect).also { r -> lines.forEach { r.union(it.rect) } }
		val thickness = lines.map { min(it.rect.width(), it.rect.height()) }.average().toFloat()
		val vertical = lines.count { it.rect.height() > it.rect.width() * 1.5f } * 2 > lines.size

		fun text(): String {
			val ordered = if (vertical) {
				lines.sortedByDescending { it.rect.centerX() }
			} else {
				lines.sortedBy { it.rect.centerY() }
			}
			val cjk = ordered.any { l -> l.text.any { scriptOf(it) == "ja" || scriptOf(it) == "zh" } }
			return ordered.joinToString(if (cjk) "" else " ") { it.text }
				.replace(Regex("\\s+"), " ").trim()
		}
	}

	private fun group(lines: List<TextLine>): List<Bubble> {
		val n = lines.size
		val parent = IntArray(n) { it }
		fun find(i: Int): Int {
			var x = i
			while (parent[x] != x) {
				parent[x] = parent[parent[x]]
				x = parent[x]
			}
			return x
		}
		for (i in 0 until n) {
			for (j in i + 1 until n) {
				if (lines[i].block == lines[j].block || shouldMerge(lines[i].rect, lines[j].rect)) {
					parent[find(i)] = find(j)
				}
			}
		}
		return lines.indices.groupBy { find(it) }.values.map { ids -> Bubble(ids.map { lines[it] }) }
	}

	private fun shouldMerge(a: Rect, b: Rect): Boolean {
		val va = a.height() > a.width() * 1.5f
		val vb = b.height() > b.width() * 1.5f
		if (va != vb) return false
		val ta = min(a.width(), a.height()).toFloat()
		val tb = min(b.width(), b.height()).toFloat()
		val tmax = max(ta, tb)
		if (tmax > min(ta, tb) * 2.0f) return false
		val dx = max(0, max(a.left, b.left) - min(a.right, b.right))
		val dy = max(0, max(a.top, b.top) - min(a.bottom, b.bottom))
		return dx <= tmax * 1.1f && dy <= tmax * 1.1f
	}

	// ---------------------------------------------------------------- translation

	private class Translated(val bubble: Bubble, val text: String)

	private class Outcome(val items: List<Translated>, val complete: Boolean)

	private suspend fun translateBubbles(bubbles: List<Bubble>, cfg: TranslateConfig, page: Bitmap): Outcome {
		val items = bubbles.map { it to it.text() }.filter { (_, t) -> t.count { c -> c.isLetter() } >= 2 }
		if (items.isEmpty()) return Outcome(emptyList(), true)
		val pageLatin = pageLatinLanguage(items.map { it.second }, cfg)
		val sources = items.map { (_, t) -> detectSource(t, cfg, pageLatin) }
		// skip text that is already in the target language
		val keep = items.indices.filter { sources[it] != cfg.target }
		if (keep.isEmpty()) return Outcome(emptyList(), true)

		if (cfg.useGemini) {
			val g = runCatchingCancellable { geminiTranslate(keep.map { items[it].second }, cfg, page) }.getOrNull()
			if (g != null) {
				val out = ArrayList<Translated>()
				var complete = true
				keep.forEachIndexed { k, idx ->
					val t = g[k]
					when {
						t == null -> complete = false // model skipped it by mistake
						t.isBlank() -> Unit // sound effect / watermark: intentionally left as is
						else -> out.add(Translated(items[idx].first, t.trim()))
					}
				}
				return Outcome(out, complete)
			}
		}
		var complete = true
		val out = keep.mapNotNull { i ->
			val t = runCatchingCancellable { mlkitTranslate(items[i].second, sources[i], cfg.target) }
				.getOrNull()?.trim()
			if (t.isNullOrEmpty()) {
				complete = false
				null
			} else {
				Translated(items[i].first, t)
			}
		}
		return Outcome(out, complete)
	}

	/**
	 * Language of the Latin-script text on the whole page. Guessing it per bubble is unreliable
	 * (short ALL CAPS lines get random languages, and then ML Kit leaves them untranslated).
	 */
	private suspend fun pageLatinLanguage(texts: List<String>, cfg: TranslateConfig): String {
		if (cfg.source != "auto") return cfg.source
		val latin = texts.filter { t -> t.any { scriptOf(it) == "latin" } }.joinToString(". ").lowercase()
		if (latin.length < 40) return "en"
		val id = runCatchingCancellable { langId.identifyLanguage(latin).await() }.getOrNull()
		return if (id != null && id in TranslateLanguage.getAllLanguages()) id else "en"
	}

	private fun detectSource(text: String, cfg: TranslateConfig, pageLatin: String): String {
		if (cfg.source != "auto") return cfg.source
		val counts = HashMap<String, Int>()
		for (c in text) {
			val s = scriptOf(c)
			if (s.isNotEmpty()) counts[s] = (counts[s] ?: 0) + 1
		}
		val top = counts.entries.maxByOrNull { it.value }?.key ?: return "en"
		if (counts["ja"] != null) return "ja" // kana present: Japanese even if kanji dominate
		return if (top != "latin") top else pageLatin
	}

	private val readyPairs = HashSet<String>()

	private suspend fun mlkitTranslate(text: String, from: String, to: String): String {
		val pair = "$from>$to"
		val translator = translators.getOrPut(pair) {
			Translation.getClient(
				TranslatorOptions.Builder().setSourceLanguage(from).setTargetLanguage(to).build(),
			)
		}
		if (pair !in readyPairs) {
			translator.downloadModelIfNeeded(DownloadConditions.Builder().build()).await() // first use only
			readyPairs.add(pair)
		}
		val input = normalizeForMt(text)
		var last: Exception? = null
		for (attempt in 0 until 3) {
			try {
				val result = translator.translate(input).await().trim()
				if (result.isNotEmpty()) return result
			} catch (e: CancellationException) {
				throw e
			} catch (e: Exception) {
				last = e
			}
			delay(400)
		}
		throw last ?: IllegalStateException("ML Kit returned empty text")
	}

	/**
	 * ML Kit treats ALL CAPS words as names / acronyms and leaves them untranslated or breaks them.
	 * Manga is mostly ALL CAPS, so convert it to normal sentence case before translating.
	 */
	private fun normalizeForMt(text: String): String {
		var t = text.replace('|', 'I')
		val letters = t.filter { it.isLetter() }
		val allCaps = letters.length >= 3 && letters.all { it.isUpperCase() } && letters.any { scriptOf(it) == "latin" }
		if (!allCaps) return t
		val sb = StringBuilder()
		var cap = true
		for (c in t.lowercase()) {
			if (cap && c.isLetter()) {
				sb.append(c.uppercaseChar())
				cap = false
			} else {
				sb.append(c)
			}
			if (c == '.' || c == '!' || c == '?') cap = true
		}
		t = sb.toString().replace(Regex("\\bi\\b"), "I")
		return t
	}

	private suspend fun geminiTranslate(texts: List<String>, cfg: TranslateConfig, page: Bitmap): Map<Int, String> {
		val image = withContext(Dispatchers.Default) { pageJpegBase64(page) }
		var last: Exception? = null
		for (attempt in 0 until 2) {
			try {
				return requestGemini(texts, cfg, image)
			} catch (e: CancellationException) {
				throw e
			} catch (e: Exception) {
				last = e
				if (attempt == 0) delay(1500)
			}
		}
		throw last ?: IllegalStateException("Gemini failed")
	}

	/** The page is sent as an image too, so the model sees who speaks and can fix OCR mistakes. */
	private fun pageJpegBase64(page: Bitmap): String? {
		if (page.height > page.width * 3) return null // webtoon strip: text only
		val scale = min(1f, 1600f / max(page.width, page.height))
		val img = if (scale < 1f) {
			Bitmap.createScaledBitmap(page, (page.width * scale).toInt(), (page.height * scale).toInt(), true)
		} else {
			page
		}
		val bytes = ByteArrayOutputStream().also { img.compress(Bitmap.CompressFormat.JPEG, 80, it) }.toByteArray()
		if (img !== page) img.recycle()
		return Base64.encodeToString(bytes, Base64.NO_WRAP)
	}

	private suspend fun requestGemini(texts: List<String>, cfg: TranslateConfig, imageB64: String?): Map<Int, String> =
		withContext(Dispatchers.IO) {
			val target = Locale(cfg.target).getDisplayLanguage(Locale.ENGLISH)
			val src = if (cfg.source == "auto") "its original language" else Locale(cfg.source).getDisplayLanguage(Locale.ENGLISH)
			val numbered = JSONArray()
			texts.forEachIndexed { i, t -> numbered.put(JSONObject().put("id", i).put("text", t)) }
			val prompt = "You are a professional manga / webtoon scanlation translator and typesetter.\n" +
				"${if (imageB64 != null) "The attached image is the page. " else ""}" +
				"Below is a JSON array of text snippets found on it by OCR. Each snippet is typeset on its own place on the page. " +
				"OCR may contain mistakes, wrong letters or split words: fix them using the image and the context.\n" +
				"Translate every snippet from $src into $target.\n" +
				"Rules:\n" +
				"- Natural spoken language that matches the speaker's tone, as a human translator would write it.\n" +
				"- Short enough to fit in a speech bubble. No explanations, no notes.\n" +
				"- Keep character names consistent and transliterate them naturally.\n" +
				"- Snippets that are only sound effects, watermarks, site names, logos, credits, page numbers or random noise: return an empty string.\n" +
				"Return ONLY a JSON array of objects {\"id\": <same id>, \"text\": <translation>} for ALL ${texts.size} snippets.\n\n" +
				numbered.toString()
			val parts = JSONArray()
			if (imageB64 != null) {
				parts.put(JSONObject().put("inline_data", JSONObject().put("mime_type", "image/jpeg").put("data", imageB64)))
			}
			parts.put(JSONObject().put("text", prompt))
			val body = JSONObject()
				.put("contents", JSONArray().put(JSONObject().put("parts", parts)))
				.put("generationConfig", JSONObject().put("responseMimeType", "application/json").put("temperature", 0.3))
			val conn = URL("https://generativelanguage.googleapis.com/v1beta/models/${cfg.geminiModel}:generateContent")
				.openConnection() as HttpURLConnection
			try {
				conn.requestMethod = "POST"
				conn.connectTimeout = 15_000
				conn.readTimeout = 90_000
				conn.doOutput = true
				conn.setRequestProperty("Content-Type", "application/json")
				conn.setRequestProperty("x-goog-api-key", cfg.geminiKey)
				conn.outputStream.use { it.write(body.toString().toByteArray()) }
				check(conn.responseCode in 200..299) { "Gemini HTTP ${conn.responseCode}" }
				val resp = conn.inputStream.bufferedReader().use { it.readText() }
				val answer = JSONObject(resp).getJSONArray("candidates").getJSONObject(0)
					.getJSONObject("content").getJSONArray("parts").getJSONObject(0).getString("text")
				val root = JSONTokener(answer).nextValue()
				val arr = when (root) {
					is JSONArray -> root
					is JSONObject -> root.optJSONArray("translations") ?: root.optJSONArray("items")
						?: error("Unexpected Gemini answer")
					else -> error("Unexpected Gemini answer")
				}
				val map = HashMap<Int, String>()
				for (i in 0 until arr.length()) {
					val o = arr.get(i)
					if (o is JSONObject) {
						map[o.getInt("id")] = o.optString("text", "")
					} else if (arr.length() == texts.size) {
						map[i] = arr.getString(i)
					}
				}
				check(map.isNotEmpty()) { "Gemini returned nothing" }
				map
			} finally {
				conn.disconnect()
			}
		}

	// ---------------------------------------------------------------- rendering

	private class Ring(val color: Int, val uniform: Boolean) {
		val luma: Float get() = 0.299f * Color.red(color) + 0.587f * Color.green(color) + 0.114f * Color.blue(color)
	}

	private fun render(bitmap: Bitmap, items: List<Translated>) {
		val canvas = Canvas(bitmap)

		// pass 1: erase the original text pixels and rebuild the background underneath (inpainting),
		// so no rectangles stay visible, neither on white bubbles nor on artwork
		val rings = items.map { ringInfo(bitmap, it.bubble.rect, max(4, (it.bubble.thickness * 0.35f).toInt())) }
		items.forEachIndexed { i, item -> eraseText(bitmap, item.bubble, rings[i]) }

		// pass 2: typeset the translation so it fills the bubble
		val typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
		val minSize = max(9f, bitmap.width * 0.011f)
		items.forEachIndexed { i, item ->
			val b = item.bubble
			val ring = rings[i]
			val area: Rect
			if (ring.uniform) {
				val grow = (max(b.rect.width(), b.rect.height()) * 0.6f).toInt() + 8
				area = growWhileUniform(bitmap, b.rect, ring.color, grow)
			} else {
				val px = (b.thickness * 0.5f).toInt()
				val py = (b.thickness * 0.2f).toInt()
				area = Rect(b.rect).apply { inset(-px, -py) }
			}
			if (area.width() < area.height() * 0.6f) { // narrow (vertical) text: use a wider box
				val w = (area.height() * 0.6f).toInt()
				val cx = area.centerX()
				area.set(cx - w / 2, area.top, cx + w / 2, area.bottom)
			}
			area.intersect(0, 0, bitmap.width, bitmap.height)
			area.inset((area.width() * 0.06f).toInt(), (area.height() * 0.06f).toInt())
			if (area.width() < 8 || area.height() < 8) return@forEachIndexed

			val dark = ring.luma < 128f
			val paint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
				this.typeface = typeface
				color = if (dark) Color.WHITE else Color.BLACK
			}
			var lo = minSize
			var hi = min(80f, max(minSize, item.bubble.thickness * 1.25f))
			repeat(10) {
				val mid = (lo + hi) / 2f
				paint.textSize = mid
				if (layoutOf(item.text, paint, area.width()).height <= area.height()) lo = mid else hi = mid
			}
			paint.textSize = lo
			val layout = layoutOf(item.text, paint, area.width())

			canvas.save()
			canvas.translate(area.left.toFloat(), area.exactCenterY() - layout.height / 2f)
			if (!ring.uniform) { // artwork behind the text: add an outline so it stays readable
				val outline = TextPaint(paint).apply {
					style = Paint.Style.STROKE
					strokeWidth = lo * 0.14f
					strokeJoin = Paint.Join.ROUND
					color = if (dark) Color.BLACK else Color.WHITE
				}
				layoutOf(item.text, outline, area.width()).draw(canvas)
			}
			layout.draw(canvas)
			canvas.restore()
		}
	}

	/** Removes only the glyph pixels (plus a small halo) and fills them from the surrounding background. */
	private fun eraseText(bmp: Bitmap, b: Bubble, ring: Ring) {
		val pad = max(3, (b.thickness * 0.22f).toInt())
		val roi = Rect(b.rect).apply { inset(-pad, -pad) }
		if (!roi.intersect(0, 0, bmp.width, bmp.height)) return
		val w = roi.width()
		val h = roi.height()
		if (w < 2 || h < 2) return
		val px = IntArray(w * h)
		bmp.getPixels(px, 0, w, roi.left, roi.top, w, h)
		val mask = BooleanArray(w * h)
		for (line in b.lines) {
			val lr = Rect(line.rect).apply { inset(-pad, -pad) }
			if (!lr.intersect(roi)) continue
			val bg = if (ring.uniform) ring.color else ringInfo(bmp, line.rect, pad + 2).color
			for (y in lr.top until lr.bottom) {
				val row = (y - roi.top) * w - roi.left
				for (x in lr.left until lr.right) {
					if (dist(px[row + x], bg) > 100) mask[row + x] = true
				}
			}
		}
		dilate(mask, w, h, max(1, (b.thickness * 0.12f).toInt()))
		inpaint(px, mask, w, h, ring.color)
		bmp.setPixels(px, 0, w, roi.left, roi.top, w, h)
	}

	private fun dilate(mask: BooleanArray, w: Int, h: Int, r: Int) {
		val tmp = BooleanArray(mask.size)
		for (y in 0 until h) {
			for (x in 0 until w) {
				if (!mask[y * w + x]) continue
				for (dx in -r..r) {
					val nx = x + dx
					if (nx in 0 until w) tmp[y * w + nx] = true
				}
			}
		}
		val out = BooleanArray(mask.size)
		for (y in 0 until h) {
			for (x in 0 until w) {
				if (!tmp[y * w + x]) continue
				for (dy in -r..r) {
					val ny = y + dy
					if (ny in 0 until h) out[ny * w + x] = true
				}
			}
		}
		out.copyInto(mask)
	}

	/** Fills masked pixels: average of horizontal and vertical interpolation, then a few smoothing passes. */
	private fun inpaint(px: IntArray, mask: BooleanArray, w: Int, h: Int, fallback: Int) {
		val hv = IntArray(px.size)
		val hOk = BooleanArray(px.size)
		val vv = IntArray(px.size)
		val vOk = BooleanArray(px.size)
		for (y in 0 until h) interpolateLine(px, mask, y * w, 1, w, hv, hOk)
		for (x in 0 until w) interpolateLine(px, mask, x, w, h, vv, vOk)
		for (i in px.indices) {
			if (!mask[i]) continue
			px[i] = when {
				hOk[i] && vOk[i] -> mix(hv[i], vv[i], 0.5f)
				hOk[i] -> hv[i]
				vOk[i] -> vv[i]
				else -> fallback
			}
		}
		repeat(6) {
			for (y in 0 until h) {
				for (x in 0 until w) {
					val i = y * w + x
					if (!mask[i]) continue
					var r = 0
					var g = 0
					var b = 0
					var n = 0
					if (x > 0) { val c = px[i - 1]; r += Color.red(c); g += Color.green(c); b += Color.blue(c); n++ }
					if (x < w - 1) { val c = px[i + 1]; r += Color.red(c); g += Color.green(c); b += Color.blue(c); n++ }
					if (y > 0) { val c = px[i - w]; r += Color.red(c); g += Color.green(c); b += Color.blue(c); n++ }
					if (y < h - 1) { val c = px[i + w]; r += Color.red(c); g += Color.green(c); b += Color.blue(c); n++ }
					if (n > 0) px[i] = Color.rgb(r / n, g / n, b / n)
				}
			}
		}
	}

	private fun interpolateLine(
		px: IntArray, mask: BooleanArray, start: Int, stride: Int, len: Int, out: IntArray, ok: BooleanArray,
	) {
		var i = 0
		while (i < len) {
			if (!mask[start + i * stride]) {
				i++
				continue
			}
			var j = i
			while (j < len && mask[start + j * stride]) j++
			val li = i - 1
			val ri = j
			for (k in i until j) {
				val idx = start + k * stride
				when {
					li >= 0 && ri < len -> {
						out[idx] = mix(px[start + li * stride], px[start + ri * stride], (k - li).toFloat() / (ri - li))
						ok[idx] = true
					}
					li >= 0 -> { out[idx] = px[start + li * stride]; ok[idx] = true }
					ri < len -> { out[idx] = px[start + ri * stride]; ok[idx] = true }
				}
			}
			i = j
		}
	}

	private fun mix(a: Int, b: Int, t: Float): Int = Color.rgb(
		(Color.red(a) + (Color.red(b) - Color.red(a)) * t).toInt(),
		(Color.green(a) + (Color.green(b) - Color.green(a)) * t).toInt(),
		(Color.blue(a) + (Color.blue(b) - Color.blue(a)) * t).toInt(),
	)

	private fun layoutOf(text: String, paint: TextPaint, width: Int): StaticLayout =
		StaticLayout.Builder.obtain(text, 0, text.length, paint, width)
			.setAlignment(Layout.Alignment.ALIGN_CENTER)
			.setLineSpacing(0f, 0.95f)
			.setBreakStrategy(Layout.BREAK_STRATEGY_BALANCED)
			.setHyphenationFrequency(Layout.HYPHENATION_FREQUENCY_NONE)
			.build()

	/** Average colour of a thin ring around [r] and whether that ring is a flat colour (e.g. white bubble). */
	private fun ringInfo(bmp: Bitmap, r: Rect, margin: Int): Ring {
		val l = (r.left - margin).coerceIn(0, bmp.width - 1)
		val t = (r.top - margin).coerceIn(0, bmp.height - 1)
		val rt = (r.right + margin).coerceIn(0, bmp.width - 1)
		val b = (r.bottom + margin).coerceIn(0, bmp.height - 1)
		val samples = ArrayList<Int>()
		val stepX = max(1, (rt - l) / 60)
		val stepY = max(1, (b - t) / 30)
		var x = l
		while (x <= rt) {
			samples.add(bmp.getPixel(x, t))
			samples.add(bmp.getPixel(x, b))
			x += stepX
		}
		var y = t
		while (y <= b) {
			samples.add(bmp.getPixel(l, y))
			samples.add(bmp.getPixel(rt, y))
			y += stepY
		}
		if (samples.isEmpty()) return Ring(Color.WHITE, false)
		var mean = meanOf(samples)
		val inliers = samples.filter { dist(it, mean) <= 90 }
		if (inliers.isNotEmpty()) mean = meanOf(inliers)
		val flat = samples.count { dist(it, mean) <= 55 }
		return Ring(mean, flat >= samples.size * 0.8f)
	}

	private fun meanOf(list: List<Int>): Int {
		var r = 0L
		var g = 0L
		var b = 0L
		for (p in list) {
			r += Color.red(p)
			g += Color.green(p)
			b += Color.blue(p)
		}
		val n = list.size
		return Color.rgb((r / n).toInt(), (g / n).toInt(), (b / n).toInt())
	}

	private fun dist(a: Int, b: Int) =
		abs(Color.red(a) - Color.red(b)) + abs(Color.green(a) - Color.green(b)) + abs(Color.blue(a) - Color.blue(b))

	/** Grows [r] outwards while the new edge strip is still the bubble colour, i.e. finds the bubble interior. */
	private fun growWhileUniform(bmp: Bitmap, r: Rect, bg: Int, maxGrow: Int): Rect {
		val out = Rect(r)
		val step = 3
		var gl = 0
		var gr = 0
		var gt = 0
		var gb = 0
		var l = true
		var rt = true
		var t = true
		var b = true
		while (l || rt || t || b) {
			if (l) {
				val x = out.left - step
				if (x < 0 || gl + step > maxGrow || !isFlat(bmp, x, out.top, x + 1, out.bottom, bg)) l = false
				else { out.left = x; gl += step }
			}
			if (rt) {
				val x = out.right + step
				if (x >= bmp.width || gr + step > maxGrow || !isFlat(bmp, x, out.top, x + 1, out.bottom, bg)) rt = false
				else { out.right = x; gr += step }
			}
			if (t) {
				val y = out.top - step
				if (y < 0 || gt + step > maxGrow || !isFlat(bmp, out.left, y, out.right, y + 1, bg)) t = false
				else { out.top = y; gt += step }
			}
			if (b) {
				val y = out.bottom + step
				if (y >= bmp.height || gb + step > maxGrow || !isFlat(bmp, out.left, y, out.right, y + 1, bg)) b = false
				else { out.bottom = y; gb += step }
			}
		}
		return out
	}

	private fun isFlat(bmp: Bitmap, x0: Int, y0: Int, x1: Int, y1: Int, bg: Int): Boolean {
		val left = x0.coerceIn(0, bmp.width - 1)
		val top = y0.coerceIn(0, bmp.height - 1)
		val w = (x1.coerceIn(left + 1, bmp.width) - left)
		val h = (y1.coerceIn(top + 1, bmp.height) - top)
		val px = IntArray(w * h)
		bmp.getPixels(px, 0, w, left, top, w, h)
		return px.count { dist(it, bg) <= 60 } >= px.size * 0.94f
	}

	// ---------------------------------------------------------------- helpers

	private fun decode(uri: Uri): Bitmap? {
		val options = BitmapFactory.Options().apply { inMutable = true }
		return openStream(uri).use { BitmapFactory.decodeStream(it, null, options) }
	}

	private fun openStream(uri: Uri): InputStream = if (uri.isZipUri()) {
		val zip = ZipFile(uri.schemeSpecificPart)
		val entry = zip.getEntry(uri.fragment)
		object : InputStream() {
			private val s = zip.getInputStream(entry)
			override fun read() = s.read()
			override fun read(b: ByteArray, off: Int, len: Int) = s.read(b, off, len)
			override fun close() {
				s.close()
				zip.close()
			}
		}
	} else {
		uri.toFile().inputStream()
	}

	/** keep translated cache under ~300 MB */
	private fun trimCache() {
		val files = cacheDir.listFiles()?.sortedBy { it.lastModified() } ?: return
		var total = files.sumOf { it.length() }
		for (f in files) {
			if (total <= 300L * 1024 * 1024) break
			total -= f.length()
			f.delete()
		}
	}

	private fun sha1(s: String) = MessageDigest.getInstance("SHA-1")
		.digest(s.toByteArray()).joinToString("") { "%02x".format(it) }

	private companion object {
		/** "ja", "zh", "ko", "hi", "ar", "latin" or "" for digits/symbols */
		fun scriptOf(c: Char): String {
			if (!c.isLetter()) return ""
			return when (Character.UnicodeBlock.of(c)) {
				Character.UnicodeBlock.HIRAGANA,
				Character.UnicodeBlock.KATAKANA,
				Character.UnicodeBlock.KATAKANA_PHONETIC_EXTENSIONS -> "ja"

				Character.UnicodeBlock.HANGUL_SYLLABLES,
				Character.UnicodeBlock.HANGUL_JAMO,
				Character.UnicodeBlock.HANGUL_COMPATIBILITY_JAMO -> "ko"

				Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS,
				Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS_EXTENSION_A,
				Character.UnicodeBlock.CJK_COMPATIBILITY_IDEOGRAPHS -> "zh"

				Character.UnicodeBlock.DEVANAGARI -> "hi"
				Character.UnicodeBlock.ARABIC -> "ar"
				else -> "latin"
			}
		}
	}
}
