package org.koitharu.kotatsu.reader.translate

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.net.Uri
import android.text.Layout
import android.text.StaticLayout
import android.text.TextPaint
import androidx.core.net.toFile
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import org.koitharu.kotatsu.core.util.ext.isZipUri
import java.io.File
import java.io.InputStream
import java.security.MessageDigest
import java.util.zip.ZipFile
import javax.inject.Inject
import javax.inject.Singleton

/**
 * On-device manga page translator:
 *  1. OCR (ML Kit) finds text blocks (speech bubbles)
 *  2. ML Kit Translate translates each block (models are downloaded once, then it works offline)
 *  3. translated text is drawn over the original block
 *
 * Result is cached as a file so every page is processed only once.
 */
@Singleton
class MangaTranslator @Inject constructor(
	@ApplicationContext private val context: Context,
) {

	private val lock = Mutex() // one page at a time: OCR + big bitmaps are memory hungry
	private val cacheDir = File(context.cacheDir, "manga_translate").apply { mkdirs() }

	private val recognizers = HashMap<String, TextRecognizer>()
	private val translators = HashMap<String, Translator>()

	/**
	 * @return uri of the translated image, or [uri] itself if there was nothing to translate
	 */
	suspend fun translate(uri: Uri, sourceLang: String, targetLang: String): Uri {
		if (sourceLang == targetLang) return uri
		val out = File(cacheDir, "${sha1(uri.toString())}_${sourceLang}_$targetLang.png")
		if (out.exists() && out.length() > 0) return Uri.fromFile(out)
		return lock.withLock {
			if (out.exists() && out.length() > 0) return@withLock Uri.fromFile(out)
			val bitmap = withContext(Dispatchers.IO) { decode(uri) } ?: return@withLock uri
			try {
				val blocks = recognize(bitmap, sourceLang)
				if (blocks.isEmpty()) return@withLock uri
				val translated = translateBlocks(blocks, sourceLang, targetLang)
				if (translated.isEmpty()) return@withLock uri
				withContext(Dispatchers.Default) { render(bitmap, translated) }
				withContext(Dispatchers.IO) {
					val tmp = File(cacheDir, out.name + ".tmp")
					tmp.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
					tmp.renameTo(out)
					trimCache()
				}
				Uri.fromFile(out)
			} finally {
				bitmap.recycle()
			}
		}
	}

	private data class Block(val rect: Rect, val text: String)

	private suspend fun recognize(bitmap: Bitmap, lang: String): List<Block> {
		val recognizer = recognizers.getOrPut(lang) {
			TextRecognition.getClient(
				when (lang) {
					"ja" -> JapaneseTextRecognizerOptions.Builder().build()
					"zh" -> ChineseTextRecognizerOptions.Builder().build()
					"ko" -> KoreanTextRecognizerOptions.Builder().build()
					else -> TextRecognizerOptions.DEFAULT_OPTIONS
				},
			)
		}
		val result = recognizer.process(InputImage.fromBitmap(bitmap, 0)).await()
		val joiner = if (lang == "ja" || lang == "zh") "" else " "
		return result.textBlocks.mapNotNull { block ->
			val rect = block.boundingBox ?: return@mapNotNull null
			val text = block.lines.joinToString(joiner) { it.text.trim() }.trim()
			// skip noise: single symbols, page numbers, etc.
			if (text.length < 2 || text.none { it.isLetter() }) null else Block(rect, text)
		}
	}

	private suspend fun translateBlocks(blocks: List<Block>, from: String, to: String): List<Block> {
		val translator = translators.getOrPut("$from>$to") {
			Translation.getClient(
				TranslatorOptions.Builder().setSourceLanguage(from).setTargetLanguage(to).build(),
			)
		}
		// downloads the language models on first use (needs internet once)
		translator.downloadModelIfNeeded(DownloadConditions.Builder().build()).await()
		return blocks.mapNotNull { block ->
			val t = translator.translate(block.text).await().trim()
			if (t.isEmpty()) null else Block(block.rect, t)
		}
	}

	private fun render(bitmap: Bitmap, blocks: List<Block>) {
		val canvas = Canvas(bitmap)
		val bg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE }
		val pad = (bitmap.width * 0.006f).coerceAtLeast(3f)
		val textPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.BLACK }
		for (block in blocks) {
			val box = RectF(block.rect).apply { inset(-pad, -pad) }
			canvas.drawRoundRect(box, pad * 2, pad * 2, bg)
			val layout = fitText(block.text, textPaint, box.width() - pad, box.height() - pad)
			canvas.save()
			canvas.translate(box.left + pad / 2, box.top + (box.height() - layout.height) / 2f)
			layout.draw(canvas)
			canvas.restore()
		}
	}

	/** Largest font size that fits into the box (down to a readable minimum). */
	private fun fitText(text: String, paint: TextPaint, w: Float, h: Float): StaticLayout {
		val width = w.toInt().coerceAtLeast(8)
		var size = (h / 2f).coerceIn(14f, 48f)
		val min = 10f
		var layout: StaticLayout
		do {
			paint.textSize = size
			layout = StaticLayout.Builder.obtain(text, 0, text.length, paint, width)
				.setAlignment(Layout.Alignment.ALIGN_CENTER)
				.build()
			size -= 1f
		} while (layout.height > h && size >= min)
		return layout
	}

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
}
