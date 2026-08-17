package com.napa.app.test

import android.content.Intent
import android.system.Os
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

/// Demo activity. When launched with a `benchmark_b64` string extra the raw
/// extra is surfaced through the "startup extras" method channel so the Dart
/// benchmark runner can run a headless case without UI interaction. Benchmark
/// launches also enable the Rust-side LLM request trace (see
/// crates/core/src/tools/loop/llm_trace.rs) by exporting the env var before
/// the Flutter engine starts.
class MainActivity : FlutterActivity() {
    private var startupChannel: MethodChannel? = null

    override fun onCreate(savedInstanceState: android.os.Bundle?) {
        val isBenchmarkLaunch =
            intent?.getStringExtra("benchmark_b64") != null
        if (isBenchmarkLaunch) {
            runCatching { Os.setenv("NAPAXI_LLM_TRACE", "1", true) }
        }
        super.onCreate(savedInstanceState)
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        startupChannel = MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            "com.napa.app.test/startup",
        ).also { channel ->
            channel.setMethodCallHandler { call, result ->
                if (call.method == "getLaunchExtras") {
                    result.success(intent?.getStringExtra("benchmark_b64"))
                } else {
                    result.notImplemented()
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
    }

    override fun cleanUpFlutterEngine(flutterEngine: FlutterEngine) {
        startupChannel?.setMethodCallHandler(null)
        startupChannel = null
        super.cleanUpFlutterEngine(flutterEngine)
    }
}
