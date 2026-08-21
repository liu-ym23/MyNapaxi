/// Headless benchmark runner for the Napaxi demo app.
///
/// Launched by the host harness with
/// `am start --es benchmark_b64 <base64(json)>`. Instead of the chat UI the
/// app configures the model from the supplied payload, sends one benchmark
/// case to a fresh session, records the six harness metrics from the chat
/// event stream, and writes a JSON result file that the host pulls before the
/// app is uninstalled for the next case.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter/material.dart';
import 'package:napaxi_flutter/napaxi_flutter.dart' as sdk;
import 'package:path_provider/path_provider.dart';

/// Pending benchmark payload when the app was launched in benchmark UI mode
/// (see main()): the chat UI renders normally while a controller drives model
/// configuration, the warm-up turn and the measured prompt, and collects the
/// same six metrics from the session event stream.
Map<String, dynamic>? pendingBenchmarkPayload;

/// Reads the `benchmark_b64` string extra from the launching intent.
Future<Map<String, dynamic>?> readBenchmarkPayload() async {
  const channel = MethodChannel('com.napa.app.test/startup');
  try {
    final raw = await channel.invokeMethod<String>('getLaunchExtras');
    if (raw == null || raw.trim().isEmpty) return null;
    final decoded = utf8.decode(base64Decode(raw.trim()));
    return jsonDecode(decoded) as Map<String, dynamic>;
  } on PlatformException {
    return null;
  } catch (_) {
    return null;
  }
}

class BenchmarkCase {
  BenchmarkCase.fromMap(Map<String, dynamic> map)
    : id = map['id'] as String? ?? 'case',
      prompt = map['prompt'] as String? ?? '',
      setupPrompt = map['setup_prompt'] as String?,
      timeoutSeconds = (map['timeout_seconds'] as num?)?.toInt() ?? 300,
      earlySuccessTool = map['early_success_tool'] as String?;

  final String id;
  final String prompt;

  /// Optional warm-up message sent (and fully awaited) before the measured
  /// prompt, in the same fresh session — used to stage workspace fixtures
  /// such as a file the measured turn must read. Setup tool calls are NOT
  /// counted toward the metrics.
  final String? setupPrompt;

  final int timeoutSeconds;

  /// Tool whose invocation cuts the turn short (process exits) as soon as its
  /// ToolCallEvent arrives, without waiting for the tool result. Used for
  /// tools that block forever in a headless environment (take_photo,
  /// ask_human). This is execution control only — the case is scored
  /// afterwards by the host-side LLM judge from the recorded trajectory.
  final String? earlySuccessTool;
}

class BenchmarkConfig {
  BenchmarkConfig.fromMap(Map<String, dynamic> map)
    : runId = map['run_id'] as String? ?? 'run',
      suite = map['suite'] as String? ?? 'default',
      baseUrl = map['base_url'] as String? ?? '',
      apiKey = map['api_key'] as String? ?? '',
      model = map['model'] as String? ?? '',
      provider = map['provider'] as String? ?? 'openai_compatible',
      maxToolIterations = (map['max_tool_iterations'] as num?)?.toInt() ?? 0,
      responseLanguage = map['response_language'] as String? ?? 'zh',
      caseSpec = BenchmarkCase.fromMap(
        Map<String, dynamic>.from(map['case'] as Map? ?? const {}),
      );

  final String runId;
  final String suite;
  final String baseUrl;
  final String apiKey;
  final String model;
  final String provider;
  final int maxToolIterations;
  final String responseLanguage;
  final BenchmarkCase caseSpec;
}

class BenchmarkResult {
  BenchmarkResult(this.config, this.caseSpec);

  final BenchmarkConfig config;
  final BenchmarkCase caseSpec;

  // The harness metrics. Completion is scored off-device by the host-side
  // LLM judge from the recorded trajectory (see benchmark/judge.py); the app
  // only captures timing, tokens, tool-call counts and the full trace.
  int totalDurationMs = 0;
  int? ttftMs;
  int? totalTokens;
  int? promptTokens;
  int? outputTokens;
  int toolCallCount = 0;
  int toolCallSuccessCount = 0;
  int toolCallErrorCount = 0;

  // Supporting detail.
  String finalResponse = '';
  String runStatus = '';
  String error = '';
  final List<Map<String, dynamic>> toolCalls = [];
  final Map<String, dynamic> contextStatusRaw = {};

  /// Raw request-level trace entries collected from the Rust dump, one per
  /// LLM call: {ts, model, system_prompt, messages, tools}. Post-processed
  /// by [_buildTrace] into the deduplicated `trace` section of the result.
  final List<Map<String, dynamic>> llmTrace = [];

  /// Number of leading conversation messages produced by the warm-up turn
  /// ("你好" + reply); they stay visible in `conversation` (the model sees
  /// them as context) but are flagged `warmup: true` and excluded from
  /// offset-based analysis.
  int warmupMessageCount = 0;

  /// Wall-clock seconds for the whole per-case cycle (reset + run + pull),
  /// stamped by the host harness after pulling the result.
  int wallSeconds = 0;

  /// Set when the case's early-success tool fired and the result file was
  /// flushed from inside the event handler (see [_checkEarlySuccess]).
  bool earlySuccessTriggered = false;

  Map<String, dynamic> toMap() => {
    'schema': 'napaxi-benchmark-result/3',
    'run_id': config.runId,
    'suite': config.suite,
    'wall_seconds': wallSeconds,
    'case': {
      'id': caseSpec.id,
      'prompt': caseSpec.prompt,
      if (caseSpec.setupPrompt != null) 'setup_prompt': caseSpec.setupPrompt,
      if (caseSpec.earlySuccessTool != null)
        'early_success_tool': caseSpec.earlySuccessTool,
    },
    'model': {
      'provider': config.provider,
      'base_url': config.baseUrl,
      'model': config.model,
    },
    'metrics': {
      'duration_ms': totalDurationMs,
      'ttft_ms': ttftMs,
      'tokens': {
        'prompt': promptTokens,
        'output': outputTokens,
        'total': totalTokens,
      },
      'tool_calls': {
        'count': toolCallCount,
        'success': toolCallSuccessCount,
        'error': toolCallErrorCount,
      },
    },
    'outcome': {
      'status': runStatus,
      'error': error,
      'response': finalResponse,
    },
    'context_status': contextStatusRaw,
    'trace': _buildTrace(),
  };

  /// Full-turn trace for conversation replay:
  ///
  /// - `system_prompt`: the effective system prompt (stored once);
  /// - `tools`: the full tool descriptors visible to the model (once);
  /// - `conversation`: the linear message list of the whole turn. Each entry
  ///   is `{offset_ms, role, content}` where role is `user`, `assistant` or
  ///   `tool`, carrying the message's own timestamp — derived by diffing the
  ///   consecutive per-call message snapshots so nothing is duplicated;
  /// - `llm_calls`: one entry per LLM request `{offset_ms, visible_tools?}`
  ///   (tool names only recorded when they deviate from `tools`, e.g. the
  ///   skill-protocol gate);
  /// - `tool_calls`: executed calls with results and timing, linked to the
  ///   assistant/tool conversation entries by `call_id`.
  Map<String, dynamic> _buildTrace() {
    if (llmTrace.isEmpty) return const {};
    final baseToolNames = _toolNamesOf(llmTrace.first);
    final firstTs = DateTime.tryParse(llmTrace.first['ts'] as String? ?? '');

    final llmCalls = <Map<String, dynamic>>[];
    final conversation = <Map<String, dynamic>>[];
    var consumed = 0;
    for (final entry in llmTrace) {
      final entryTs = DateTime.tryParse(entry['ts'] as String? ?? '');
      final offsetMs = firstTs != null && entryTs != null
          ? entryTs.difference(firstTs).inMilliseconds
          : null;
      final names = _toolNamesOf(entry);
      final call = <String, dynamic>{
        if (offsetMs != null) 'offset_ms': offsetMs,
        if (names.isNotEmpty && !_sameNames(names, baseToolNames))
          'visible_tools': names,
      };
      llmCalls.add(call);

      // Each LLM call's messages are a superset of the previous call's; only
      // the newly appended messages are new conversation events.
      final messages = (entry['messages'] as List?) ?? const [];
      for (var i = consumed; i < messages.length; i++) {
        final message = messages[i];
        if (message is! Map) continue;
        final isWarmup = conversation.length < warmupMessageCount;
        conversation.add({
          if (isWarmup)
            'warmup': true
          else if (offsetMs != null)
            'offset_ms': offsetMs,
          'role': message['role'] ?? '',
          'content': message['content'] ?? _toolCallsPayloadOf(message),
          if (message['reasoning_content'] != null &&
              (message['reasoning_content'] as String).isNotEmpty)
            'reasoning': message['reasoning_content'],
          if (message['tool_calls'] != null)
            'tool_calls': message['tool_calls'],
          if (message['tool_call_id'] != null)
            'tool_call_id': message['tool_call_id'],
        });
      }
      consumed = messages.length;
    }

    // The dumps capture the request side only, so the final assistant reply
    // — the response of the last LLM call — never appears in any snapshot.
    // Append it from the event-stream-collected response so the conversation
    // is complete, stamped at the turn end.
    if (finalResponse.trim().isNotEmpty) {
      conversation.add({
        'offset_ms': totalDurationMs,
        'role': 'assistant',
        'content': finalResponse,
      });
    }

    final tools = <Map<String, dynamic>>[];
    for (final tool in (llmTrace.first['tools'] as List?) ?? const []) {
      if (tool is Map) {
        tools.add(Map<String, dynamic>.from(tool));
      }
    }
    return {
      'system_prompt': llmTrace.first['system_prompt'] ?? '',
      'tools': tools,
      'conversation': conversation,
      'llm_calls': llmCalls,
      'tool_calls': toolCalls,
    };
  }

  /// Fallback content for tool-role messages that carry their payload in a
  /// non-standard field.
  static String _toolCallsPayloadOf(Map message) {
    for (final key in const ['content', 'tool_result', 'output']) {
      final value = message[key];
      if (value is String && value.isNotEmpty) return value;
    }
    return '';
  }

  static List<String> _toolNamesOf(Map<String, dynamic> entry) {
    return ((entry['tools'] as List?) ?? const [])
        .whereType<Map>()
        .map((tool) => tool['name'] as String? ?? '')
        .where((name) => name.isNotEmpty)
        .toList(growable: false);
  }

  static bool _sameNames(List<String> a, List<String> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

/// Writes [payload] to `<externalFilesDir>/benchmark/<file>` and appends a
/// line to `<file>.done` so the host can poll for completion.
///
/// Uses path_provider (the app's own external files dir) rather than the
/// literal /storage path, which newer Android scoped-storage rules reject.
Future<String> writeResultFile(String file, String payload) async {
  final base = await getExternalStorageDirectory();
  if (base == null) {
    throw StateError('external files directory unavailable');
  }
  final dir = Directory('${base.path}/benchmark');
  if (!await dir.exists()) {
    await dir.create(recursive: true);
  }
  final file_ = File('${dir.path}/$file');
  await file_.writeAsString(payload, flush: true);
  await File('${dir.path}/$file.done').writeAsString('done\n', flush: true);
  return file_.path;
}

/// Executes one headless benchmark case against a freshly created engine.
Future<BenchmarkResult> runBenchmarkCase(BenchmarkConfig config) async {
  final result = BenchmarkResult(config, config.caseSpec);
  final stopwatch = Stopwatch();

  sdk.NapaxiEngine? engine;
  try {
    final llmConfig = sdk.LlmConfig(
      provider: config.provider,
      apiKey: config.apiKey,
      baseUrl: config.baseUrl.isEmpty ? null : config.baseUrl,
      model: config.model,
      responseLanguage: config.responseLanguage,
      maxToolIterations: config.maxToolIterations,
      // Match the demo app posture: sandbox workspace is the blast radius,
      // no approval bridge is wired in headless benchmark mode.
      shellSecurity: const sdk.ShellSecurityConfig(
        approvalMode: sdk.ShellApprovalMode.trustedAllow,
      ),
    );
    engine = await sdk.NapaxiEngine.create(config: llmConfig);
    final session = await engine.createSession(
      channelType: 'benchmark',
      accountId: 'benchmark',
      threadId: 'bench-${config.runId}-${DateTime.now().millisecondsSinceEpoch}',
    );

    // Warm-up: send "你好" after configuring the model and await its full
    // reply before the measured prompt. This initializes the engine/sandbox
    // and the model connection so the measured turn does not pay one-off
    // setup costs. Warm-up LLM requests are excluded from the trace; its
    // user/assistant messages stay in the conversation (the model sees them
    // as prior context) but are flagged `warmup: true`.
    debugPrint('[benchmark] warm-up turn for ${config.caseSpec.id}');
    final warmupOutcome = await _runTurn(
      engine: engine,
      session: session,
      prompt: '你好',
      timeoutSeconds: config.caseSpec.timeoutSeconds,
      onEvent: (_) {},
    );
    debugPrint('[benchmark] warm-up done: ${warmupOutcome ?? 'done'}');

    final setupPrompt = config.caseSpec.setupPrompt;
    if (setupPrompt != null && setupPrompt.trim().isNotEmpty) {
      debugPrint('[benchmark] setup turn for ${config.caseSpec.id}');
      await _runTurn(
        engine: engine,
        session: session,
        prompt: setupPrompt,
        timeoutSeconds: config.caseSpec.timeoutSeconds,
        onEvent: (_) {},
      );
    }

    // The warm-up/setup turns share the thread's LLM trace file; drop their
    // entries so only the measured turn's requests are reported.
    await _truncateLlmTraceBefore(
      engine.filesDir,
      session.threadId,
      DateTime.now(),
    );

    _activeFilesDir = engine.filesDir;
    _activeThreadId = session.threadId;
    stopwatch.start();
    debugPrint('[benchmark] measured turn starting');
    await _runTurn(
      engine: engine,
      session: session,
      prompt: config.caseSpec.prompt,
      timeoutSeconds: config.caseSpec.timeoutSeconds,
      onEvent: (event) {
        _recordEvent(result, event, stopwatch);
        _checkEarlySuccess(result, event);
      },
    );
    stopwatch.stop();
    debugPrint('[benchmark] measured turn done');
    if (result.earlySuccessTriggered) {
      // The process exits inside _checkEarlySuccess; reaching here means the
      // early-exit raced with normal completion (event delivered, exit
      // scheduled). Treat the case as complete either way.
    }
    result.totalDurationMs = stopwatch.elapsedMilliseconds;

    // Read token usage recorded by the Rust context engine for this thread.
    try {
      final status = await engine.contextStatus(session.threadId);
      result.promptTokens = status.lastPromptTokens;
      result.outputTokens = status.lastOutputTokens;
      result.totalTokens =
          status.lastTotalTokens ??
          (status.lastPromptTokens == null && status.lastOutputTokens == null
              ? null
              : (status.lastPromptTokens ?? 0) + (status.lastOutputTokens ?? 0));
      result.contextStatusRaw.addAll({
        'last_prompt_tokens': status.lastPromptTokens,
        'last_output_tokens': status.lastOutputTokens,
        'last_total_tokens': status.lastTotalTokens,
        'cache_read_tokens': status.cacheReadTokens,
        'cache_write_tokens': status.cacheWriteTokens,
        'estimated_tokens': status.estimatedTokens,
        'display_source': status.displaySource,
      });
    } catch (error) {
      result.contextStatusRaw['error'] = error.toString();
    }

    debugPrint('[benchmark] collecting trace');
    // Collect the request-level LLM trace dumped by the Rust tool loop for
    // the measured turn (system prompt + messages + visible tools per call).
    await _collectLlmTrace(result, engine.filesDir, session.threadId);
    result.warmupMessageCount = _countWarmupMessages(
      result.llmTrace,
      config.caseSpec.prompt,
    );
  } catch (error) {
    result.error = '$error';
  } finally {
    engine?.dispose();
  }
  return result;
}

/// Counts how many leading messages of the first trace snapshot precede the
/// measured prompt — i.e. the warm-up turn's user/assistant exchange.
int _countWarmupMessages(
  List<Map<String, dynamic>> trace,
  String measuredPrompt,
) {
  if (trace.isEmpty) return 0;
  final messages = (trace.first['messages'] as List?) ?? const [];
  var measuredIndex = -1;
  for (var i = messages.length - 1; i >= 0; i--) {
    final message = messages[i];
    if (message is Map &&
        message['role'] == 'user' &&
        (message['content'] ?? '') == measuredPrompt) {
      measuredIndex = i;
      break;
    }
  }
  return measuredIndex < 0 ? 0 : measuredIndex;
}

/// Rewrites the thread's LLM trace file keeping only entries whose `ts` is at
/// or after [cutoff], dropping warm-up/setup-turn requests. Both sides are
/// compared in UTC: the dumped `ts` is an RFC3339 UTC instant while callers
/// hold a local `DateTime.now()`.
/// Public variant for the UI-mode hooks (drops warm-up trace entries).
Future<void> truncateLlmTraceBefore(
  String filesDir,
  String threadId,
  DateTime cutoff,
) => _truncateLlmTraceBefore(filesDir, threadId, cutoff);

Future<void> _truncateLlmTraceBefore(
  String filesDir,
  String threadId,
  DateTime cutoff,
) async {
  final cutoffUtc = cutoff.toUtc();
  try {
    final file = File('$filesDir/llm-trace/llm-trace-$threadId.jsonl');
    if (!await file.exists()) return;
    final lines = await file.readAsLines();
    final kept = lines.where((line) {
      if (line.trim().isEmpty) return false;
      try {
        final ts = (jsonDecode(line) as Map<String, dynamic>)['ts'] as String?;
        return ts == null || DateTime.tryParse(ts)?.isBefore(cutoffUtc) == false;
      } catch (_) {
        return true;
      }
    }).toList();
    await file.writeAsString(kept.isEmpty ? '' : '${kept.join('\n')}\n');
  } catch (_) {
    // Trace truncation is best-effort.
  }
}

/// Reads the per-thread LLM request trace file written by the Rust tool loop
/// (see crates/core/src/tools/loop/llm_trace.rs; requires the
/// NAPAXI_LLM_TRACE=1 platform environment) into [result.llmTrace]. Missing
/// or unreadable traces are silently skipped so a misconfigured environment
/// never fails a benchmark case.
/// Public variant used by the UI-mode hooks: appends the thread's Rust LLM
/// trace dump into [result.llmTrace].
Future<void> collectLlmTraceInto(
  BenchmarkResult result,
  String filesDir,
  String threadId,
) async {
  await _collectLlmTrace(result, filesDir, threadId);
}

Future<void> _collectLlmTrace(
  BenchmarkResult result,
  String filesDir,
  String threadId,
) async {
  try {
    final file = File('$filesDir/llm-trace/llm-trace-$threadId.jsonl');
    final lines = await file.readAsLines();
    result.llmTrace.addAll(
      lines
          .where((line) => line.trim().isNotEmpty)
          .map((line) => jsonDecode(line) as Map<String, dynamic>),
    );
  } catch (_) {
    // Trace collection is best-effort.
  }
}

/// Sends one turn to [session] and awaits completion, dispatching every chat
/// event to [onEvent]. Returns null/'done' on success or an error message.
Future<String?> _runTurn({
  required sdk.NapaxiEngine engine,
  required sdk.SessionKey session,
  required String prompt,
  required int timeoutSeconds,
  required void Function(sdk.ChatEvent) onEvent,
}) async {
  final completer = Completer<String?>();
  String? streamError;

  final subscription = engine
      .sendToSession(session, prompt, maxIterations: 0)
          .listen(
            (event) {
              onEvent(event);
              if (event is sdk.ErrorEvent) {
                streamError = event.message;
              } else if (event is sdk.RunCompletedEvent && !completer.isCompleted) {
                completer.complete('done');
              }
            },
            onError: (Object error) {
              streamError = error.toString();
              if (!completer.isCompleted) completer.complete('stream error');
            },
            onDone: () {
              if (!completer.isCompleted) completer.complete('done');
            },
            cancelOnError: false,
          );

  final outcome = await completer.future.timeout(
    Duration(seconds: timeoutSeconds),
    onTimeout: () {
      subscription.cancel();
      return 'timeout after ${timeoutSeconds}s';
    },
  );
  await subscription.cancel();
  return streamError ?? outcome;
}

/// Files dir / thread id of the in-flight case, set by [runBenchmarkCase]
/// so the early-success path can collect the trace before exiting.
String _activeFilesDir = '';
String _activeThreadId = '';

/// Early-success cut-off for tools that block forever headless (take_photo,
/// ask_human): the moment their ToolCallEvent arrives the case is scored as
/// successful (grade `early_success`, score 1.0), the result file written and
/// the process terminated — the tool's own completion is never awaited.
void _checkEarlySuccess(BenchmarkResult result, sdk.ChatEvent event) {
  final target = result.caseSpec.earlySuccessTool;
  if (target == null || target.isEmpty || result.earlySuccessTriggered) return;
  if (event is! sdk.ToolCallEvent || event.name != target) return;

  result.earlySuccessTriggered = true;
  _collectLlmTrace(result, _activeFilesDir, _activeThreadId).then((_) {
    result.totalDurationMs = result.toolCalls.last['offset_ms'] as int;
    return writeResultFile(
      'result-${result.config.runId}.json',
      const JsonEncoder.withIndent('  ').convert(result.toMap()),
    );
  }).then((_) {
    debugPrint('[benchmark] early success on $target, exiting');
    exit(0);
  });
}

/// Records one chat event into the benchmark result's metrics and tool-call
/// detail. `stopwatch` must be running; offsets are relative to the measured
/// turn's send time.
void _recordEvent(BenchmarkResult result, sdk.ChatEvent event, Stopwatch stopwatch) {
  final offset = stopwatch.elapsedMilliseconds;
  switch (event) {
    case sdk.RunStartedEvent():
      break;
    case sdk.ResponseDeltaEvent(:final content):
      result.ttftMs ??= offset;
      result.finalResponse += content;
    case sdk.ResponseEvent(:final content):
      result.ttftMs ??= offset;
      result.finalResponse = content;
    case sdk.ReasoningDeltaEvent():
      result.ttftMs ??= offset;
    case sdk.ToolCallEvent(:final callId, :final name, :final arguments):
      result.toolCallCount += 1;
      result.toolCalls.add({
        'call_id': callId,
        'seq': result.toolCallCount,
        'name': name,
        'arguments': arguments.length > 400 ? arguments.substring(0, 400) : arguments,
        'offset_ms': offset,
      });
    case sdk.ToolResultEvent(
        :final callId,
        :final output,
        :final isError,
      ):
      if (isError) {
        result.toolCallErrorCount += 1;
      } else {
        result.toolCallSuccessCount += 1;
      }
      for (final call in result.toolCalls) {
        if (call['call_id'] == callId) {
          call['is_error'] = isError;
          call['output'] = output.length > 400 ? output.substring(0, 400) : output;
        }
      }
    case sdk.RunCompletedEvent(:final status, :final toolCallCount):
      result.runStatus = status;
      if (toolCallCount > 0) result.toolCallCount = toolCallCount;
    case sdk.ErrorEvent(:final message):
      result.error = message;
    case sdk.InterruptedEvent():
    default:
      break;
  }
}

/// Entry point used by main() when a benchmark payload is present.
Future<void> runHeadlessBenchmark(Map<String, dynamic> payload) async {
  final config = BenchmarkConfig.fromMap(payload);
  debugPrint('[benchmark] case=${config.caseSpec.id} model=${config.model}');
  final result = await runBenchmarkCase(config);
  final json = const JsonEncoder.withIndent('  ').convert(result.toMap());
  debugPrint('[benchmark] done ttft=${result.ttftMs}ms');
  await writeResultFile('result-${config.runId}.json', json);
}


/// Drives a benchmark case through the real chat UI (plan B visualization):
/// the caller (chat_screen) hands over the send/observe hooks; this
/// controller stages the model profile, sends the warm-up turn, awaits its
/// completion, sends the measured prompt and records the same metrics the
/// headless runner collects — writing the identical result file.
class BenchmarkUiController {
  BenchmarkUiController(this.payload, this.hooks);

  final Map<String, dynamic> payload;
  final BenchmarkUiHooks hooks;

  late final BenchmarkConfig config = BenchmarkConfig.fromMap(payload);
  late final BenchmarkResult result = BenchmarkResult(config, config.caseSpec);

  bool measuredTurnStarted = false;
  bool warmupDone = false;
  DateTime? measuredSendTime;
  Stopwatch stopwatch = Stopwatch();
  String? filesDir;
  String? threadId;

  Future<void> run() async {
    debugPrint('[benchmark-ui] case=${config.caseSpec.id}');
    debugPrint('[benchmark-ui] configuring model ${config.model}...');
    await hooks.stageModelConfig(config);
    debugPrint('[benchmark-ui] model configured, UI ready');
    // Wait for the UI to settle, then start the warm-up turn.
    await Future<void>.delayed(const Duration(milliseconds: 800));
    hooks.send('你好');
    debugPrint('[benchmark-ui] warm-up turn sent ("你好")');
  }

  /// The UI chat pipeline closes the event stream when a turn ends (there
  /// is no RunCompleted event on this path); this is the authoritative
  /// turn-done signal.
  void onStreamDone() {
    debugPrint('[benchmark-ui] stream done (measured=$measuredTurnStarted)');
    if (!measuredTurnStarted) {
      if (warmupDone) return;
      warmupDone = true;
      Future<void>.microtask(() async {
        await hooks.discardPreliminaryTrace();
        await Future<void>.delayed(const Duration(milliseconds: 600));
        measuredTurnStarted = true;
        measuredSendTime = DateTime.now();
        stopwatch.start();
        hooks.send(config.caseSpec.prompt);
        debugPrint('[benchmark-ui] measured prompt sent: '
            '"${config.caseSpec.prompt.length > 40 ? config.caseSpec.prompt.substring(0, 40) : config.caseSpec.prompt}"');
      });
      return;
    }
    if (!finished) {
      finished = true;
      _finish();
    }
  }

  bool finished = false;

  /// Every chat event of both turns flows through here.
  void onEvent(sdk.ChatEvent event) {
    if (!measuredTurnStarted) return;
    _recordEvent(event);
    _checkEarlySuccessUi(event);
    if (event is sdk.ErrorEvent && !finished) {
      finished = true;
      _finish();
    }
  }

  void _recordEvent(sdk.ChatEvent event) {
    final offset = stopwatch.elapsedMilliseconds;
    switch (event) {
      case sdk.ResponseDeltaEvent(:final content):
        result.ttftMs ??= offset;
        result.finalResponse += content;
      case sdk.ResponseEvent(:final content):
        result.ttftMs ??= offset;
        result.finalResponse = content;
      case sdk.ReasoningDeltaEvent():
        result.ttftMs ??= offset;
      case sdk.ToolCallEvent(:final callId, :final name, :final arguments):
        result.toolCallCount += 1;
        result.toolCalls.add({
          'call_id': callId,
          'seq': result.toolCallCount,
          'name': name,
          'arguments': arguments.length > 400
              ? arguments.substring(0, 400)
              : arguments,
          'offset_ms': offset,
        });
      case sdk.ToolResultEvent(
          :final callId,
          :final output,
          :final isError,
        ):
        if (isError) {
          result.toolCallErrorCount += 1;
        } else {
          result.toolCallSuccessCount += 1;
        }
        for (final call in result.toolCalls) {
          if (call['call_id'] == callId) {
            call['is_error'] = isError;
            call['output'] =
                output.length > 400 ? output.substring(0, 400) : output;
          }
        }
      case sdk.RunCompletedEvent(:final status, :final toolCallCount):
        result.runStatus = status;
        if (toolCallCount > 0) result.toolCallCount = toolCallCount;
      case sdk.ErrorEvent(:final message):
        result.error = message;
      default:
        break;
    }
  }

  void _checkEarlySuccessUi(sdk.ChatEvent event) {
    final target = config.caseSpec.earlySuccessTool;
    if (target == null || target.isEmpty || result.earlySuccessTriggered) {
      return;
    }
    if (event is! sdk.ToolCallEvent || event.name != target) return;
    result.earlySuccessTriggered = true;
    debugPrint('[benchmark-ui] early success on $target');
    // Blocking tools never let the stream close; give the UI a few seconds
    // to show the invoked tool, then finish on our own.
    Future<void>.delayed(const Duration(seconds: 4), () {
      if (!finished) {
        finished = true;
        _finish();
      }
    });
  }

  Future<void> _finish() async {
    stopwatch.stop();
    result.totalDurationMs = stopwatch.elapsedMilliseconds;
    try {
      await hooks.collectTelemetry(result);
    } catch (error) {
      debugPrint('[benchmark-ui] telemetry error: $error');
    }
    debugPrint('[benchmark-ui] done ttft=${result.ttftMs}ms');
    await writeResultFile(
      'result-${config.runId}.json',
      const JsonEncoder.withIndent('  ').convert(result.toMap()),
    );
    hooks.benchmarkFinished();
  }
}

/// Hooks the chat UI provides to the benchmark controller.
abstract class BenchmarkUiHooks {
  /// Persist the benchmark model profile so the UI chat path uses it.
  Future<void> stageModelConfig(BenchmarkConfig config);

  /// Send a user message through the normal UI send pipeline.
  void send(String message);

  /// Fill token usage / trace into [result] after the measured turn.
  Future<void> collectTelemetry(BenchmarkResult result);

  /// Drop the warm-up turn's LLM trace entries before the measured turn
  /// (same thread, same dump file).
  Future<void> discardPreliminaryTrace();

  /// Called after the result file was written.
  void benchmarkFinished();
}
