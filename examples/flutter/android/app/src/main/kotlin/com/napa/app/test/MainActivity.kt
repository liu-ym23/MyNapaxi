package com.napa.app.test

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

/// Demo activity.
///
/// Model sideload channel — `downloadLocalLlmModel` streams a
/// GGUF from `url` into `<filesDir>/local-llm/<filename>` on a worker thread
/// (progress reported through `onProgress` as 0..100). The host serves the
/// multi-hundred-MB weights and maps the port with `adb reverse`, which
/// sidesteps the FUSE layer entirely (files `adb push`ed to shared storage are
/// owned by `shell`/the media uid and invisible to this app process).
///
/// The transfer resumes across retries via HTTP Range: the `adb reverse` USB
/// tunnel stalls intermittently, so each attempt continues from however many
/// bytes are already in `<filename>.part` until the file is complete.
///
/// Intended flow (host side):
///   python3 -m http.server 8888 --directory /tmp/qwen25-model
///   adb reverse tcp:8888 tcp:8888
///   invokeMethod('downloadLocalLlmModel',
///                {'url': 'http://127.0.0.1:8888/model.gguf',
///                 'filename': 'qwen2.5-0_5b-instruct-q4_k_m.gguf'})
class MainActivity : FlutterActivity() {
    private var toolingChannel: MethodChannel? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        toolingChannel = MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            "com.napa.app.test/local_llm_tooling",
        ).also { channel ->
            channel.setMethodCallHandler { call, result ->
                when (call.method) {
                    "downloadLocalLlmModel" -> {
                        val url = call.argument<String>("url")
                        val filename = call.argument<String>("filename")
                        if (url.isNullOrBlank() || filename.isNullOrBlank()) {
                            result.error("bad_args", "url and filename are required", null)
                            return@setMethodCallHandler
                        }
                        downloadModel(channel, url, filename, result)
                    }
                    else -> result.notImplemented()
                }
            }
        }
    }

    private fun downloadModel(
        channel: MethodChannel,
        url: String,
        filename: String,
        result: MethodChannel.Result,
    ) {
        val destDir = File(filesDir, "local-llm")
        val dest = File(destDir, filename)
        val handler = android.os.Handler(android.os.Looper.getMainLooper())
        thread(name = "local-llm-download") {
            try {
                destDir.mkdirs()
                // Already-complete model: nothing to do.
                if (dest.isFile && dest.length() > 0) {
                    handler.post { result.success(dest.length().toDouble()) }
                    return@thread
                }
                // Delegate to the system curl: HttpURLConnection stalls
                // irrecoverably on the adb-reverse USB tunnel, while the
                // device's own curl (with --retry + -C - resume) handles the
                // stalls. The curl process inherits this app's uid, so the
                // file it writes is app-owned — no FUSE visibility problem.
                val total = fetchTotalLength(url)
                val maxAttempts = 20
                var attempt = 0
                var exit: Int = -1
                var lastPct = -1
                val reportProgress = {
                    val pct = if (total > 0) (dest.length() * 100 / total).toInt() else 0
                    if (pct != lastPct) {
                        lastPct = pct
                        handler.post { channel.invokeMethod("onProgress", pct) }
                    }
                }
                reportProgress()
                while (attempt < maxAttempts) {
                    attempt++
                    val process = ProcessBuilder(
                        "curl", "-s", "--fail",
                        "--connect-timeout", "10",
                        "--speed-time", "30", "--speed-limit", "1024",
                        "--retry", "5", "--retry-all-errors",
                        "-C", "-",
                        "-o", dest.absolutePath,
                        url,
                    ).redirectErrorStream(true).start()
                    // Drain stdout on a reader thread and poll the output file
                    // so the UI sees progress while curl is still running.
                    val outputReader = StringBuilder()
                    val drainThread = Thread {
                        process.inputStream.bufferedReader().forEachLine { line ->
                            synchronized(outputReader) { outputReader.appendLine(line) }
                        }
                    }
                    drainThread.isDaemon = true
                    drainThread.start()
                    while (process.isAlive) {
                        reportProgress()
                        Thread.sleep(500)
                    }
                    exit = process.waitFor()
                    reportProgress()
                    if (exit == 0 && dest.isFile && dest.length() == total) break
                    val output = synchronized(outputReader) { outputReader.toString() }
                    if (output.isNotBlank()) {
                        android.util.Log.d("local-llm-dl", "attempt $attempt: $output")
                    }
                    Thread.sleep(500)
                }
                if (exit != 0 || !dest.isFile || dest.length() != total) {
                    throw IllegalStateException(
                        "curl failed after $maxAttempts attempts (exit=$exit size=${dest.length()}/$total)",
                    )
                }
                handler.post { result.success(dest.length().toDouble()) }
            } catch (e: Exception) {
                handler.post { result.error("download_failed", e.message, null) }
            }
        }
    }

    private fun fetchTotalLength(url: String): Long {
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.connectTimeout = 10_000
        conn.readTimeout = 30_000
        try {
            return conn.contentLengthLong
        } finally {
            conn.disconnect()
        }
    }

    override fun cleanUpFlutterEngine(flutterEngine: FlutterEngine) {
        toolingChannel?.setMethodCallHandler(null)
        toolingChannel = null
        super.cleanUpFlutterEngine(flutterEngine)
    }
}
