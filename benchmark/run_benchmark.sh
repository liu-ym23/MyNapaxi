#!/usr/bin/env bash
# Napaxi end-to-end benchmark harness.
#
# Suites live in benchmark/suites/<suite>.jsonl — one JSON case per line.
# For every selected case this script:
#   1. uninstalls the app (wipes workspace/memory/config state),
#   2. reinstalls the benchmark APK,
#   3. launches it headless with the case payload via `am start --es`
#      (an optional case-level setup_prompt runs first inside the app to
#      stage fixtures such as a workspace file),
#   4. polls for the on-device result file and pulls it,
#   5. moves on to the next case,
# then aggregates per-case JSON results into report.md + report.json grouped
# by suite.
#
# Usage:
#   ./benchmark/run_benchmark.sh [--skip-build] [--reset-mode clear|reinstall]
#                                [--suites basic_tool_call] [--cases id1,id2]
#                                [--out DIR] [--record]
#
# --record captures a screen video of every case (via the companion
#   bench-recorder.apk, installed once per run). The system media-projection
#   consent dialog is auto-accepted; the mp4 lands next to each result JSON.
#   Recordings keep running across the case's app reinstall because the
#   recorder is a separate package.
#
# Requires: benchmark/config.local.json with the model API credentials.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$ROOT_DIR/benchmark"
APK="$ROOT_DIR/examples/flutter/build/app/outputs/flutter-apk/app-release.apk"
ADB="${ADB:-/home/liu_ym23/.local/share/android/sdk/platform-tools/adb}"
PACKAGE="com.napa.app.test"
ACTIVITY="$PACKAGE/.MainActivity"
DEVICE_DIR="/storage/emulated/0/Android/data/$PACKAGE/files/benchmark"

SKIP_BUILD=0
RESET_MODE=reinstall
SUITES=""
CASES_FILTER=""
OUT_DIR=""
RECORD=0
RECORDER_PACKAGE="com.napaxi.bench.recorder"
RECORDER_APK="$BENCH_DIR/bench-recorder.apk"

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build) SKIP_BUILD=1 ;;
        --reset-mode) RESET_MODE="$2"; shift ;;
        --suites) SUITES="$2"; shift ;;
        --cases) CASES_FILTER="$2"; shift ;;
        --out) OUT_DIR="$2"; shift ;;
        --record) RECORD=1 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# --- Logging -----------------------------------------------------------------
# Console output is colorized and indented by phase; everything (including the
# per-case logcat capture) is teed into $OUT_DIR/run.log for post-hoc review.
# Colors are disabled automatically when not a tty (e.g. piped to a file).

if [ -t 1 ]; then
    C_RESET="$(printf '\033[0m')"; C_DIM="$(printf '\033[2m')"; C_BOLD="$(printf '\033[1m')"
    C_BLUE="$(printf '\033[34m')"; C_GREEN="$(printf '\033[32m')"; C_YELLOW="$(printf '\033[33m')"
    C_RED="$(printf '\033[31m')"; C_CYAN="$(printf '\033[36m')"; C_MAGENTA="$(printf '\033[35m')"
else
    C_RESET=''; C_DIM=''; C_BOLD=''
    C_BLUE=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_CYAN=''; C_MAGENTA=''
fi

RUN_LOG=""
# Fixed-column log format so every row aligns in the terminal:
#   HH:MM:SS | LABEL   | message body
# The label column is padded to 8 chars; colors wrap label and body.
_log() {  # _log <label-color> <label> <body-color> <body>
    local label_color="$1" label="$2" body_color="$3" body="$4"
    local padded plain
    padded="$(printf '%-8s' "$label")"
    line="$(date +%H:%M:%S) ${label_color}|${padded}|${C_RESET} ${body_color}${body}${C_RESET}"
    printf '%s\n' "$line"
    if [ -n "$RUN_LOG" ]; then
        plain="$(printf '%s' "$line" | sed $'s/\x1b\[[0-9;]*m//g')"
        printf '%s\n' "$plain" >> "$RUN_LOG"
    fi
}
log_phase() { _log "$C_BOLD$C_CYAN"    "PHASE" "$C_BOLD$C_CYAN"    "$*"; }
log_step()  { _log "$C_CYAN"           "STEP"  "$C_BLUE"           "$*"; }
log_case()  { _log "$C_MAGENTA"        "CASE"  "$C_BOLD$C_MAGENTA" "$*"; }
log_sub()   { _log ""                  ""      "$C_DIM"            "  $*"; }
log_ok()    { _log "$C_GREEN"          "OK"    "$C_BOLD$C_GREEN"  "$*"; }
log_warn()  { _log "$C_YELLOW"         "WARN"  "$C_BOLD$C_YELLOW" "$*"; }
log_fail()  { _log "$C_RED"            "FAIL"  "$C_BOLD$C_RED"    "$*"; }
log_heart() { _log "$C_DIM"            "APP"   "$C_DIM"            "$*"; }

info() { log_step "$*"; }   # back-compat alias

err() {
    _log "$C_BOLD$C_RED" "FATAL" "$C_BOLD$C_RED" "$*" >&2
    exit 1
}

# json_get <file> <python-expr over d> — jq-free JSON field access via python3.
# The expression may call _preview(s) to get a single-line, 50-char excerpt.
json_get() {
    python3 -c "
import json, sys
def _preview(s):
    s = ' '.join(str(s).split())
    return (s[:50] + '...') if len(s) > 50 else s
d = json.load(open(sys.argv[1]))
print(eval(sys.argv[2]))" "$1" "$2"
}

[ -f "$BENCH_DIR/config.local.json" ] || err "missing $BENCH_DIR/config.local.json"
command -v python3 >/dev/null || err "python3 is required"
"$ADB" get-state >/dev/null 2>&1 || err "no adb device online"

SERIAL="$("$ADB" devices | awk '$2 == "device" { print $1; exit }')"
[ -n "$SERIAL" ] || err "no adb device in 'device' state"
info "device: $SERIAL"

CONFIG="$BENCH_DIR/config.local.json"
BASE_URL="$(json_get "$CONFIG" "d.get('base_url','')")"
API_KEY="$(json_get "$CONFIG" "d.get('api_key','')")"
MODEL="$(json_get "$CONFIG" "d.get('model','')")"
PROVIDER="$(json_get "$CONFIG" "d.get('provider','openai_compatible')")"
MAX_ITER="$(json_get "$CONFIG" "d.get('max_tool_iterations',30)")"
RESP_LANG="$(json_get "$CONFIG" "d.get('response_language','zh')")"
if [ "$PROVIDER" = "local" ]; then
    # On-device provider: no cloud credentials. The GGUF is sideloaded from
    # this host over adb reverse by the app on first launch; keep the tunnel
    # mapped so a reset app can pull the weights again.
    info "local-LLM mode: cloud credentials not required"
    "$ADB" -s "$SERIAL" reverse tcp:8888 tcp:8888 >/dev/null 2>&1 </dev/null || true
else
    [ -n "$BASE_URL" ] || err "base_url missing in config.local.json"
    [ -n "$API_KEY" ] || err "api_key missing in config.local.json"
    [ -n "$MODEL" ] || err "model missing in config.local.json"
fi

if [ "$SKIP_BUILD" != "1" ]; then
    info "building Android SDK (Rust)"
    (cd "$ROOT_DIR" && ./tools/scripts/build.sh fast android >/dev/null)
    info "building APK"
    (cd "$ROOT_DIR/examples/flutter" && flutter build apk --release \
        --dart-define=NAPA_UMENG_ENABLED=false 2>&1 | tail -1)
fi
[ -f "$APK" ] || err "APK not found: $APK (run without --skip-build)"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-$BENCH_DIR/results/$RUN_ID}"
mkdir -p "$OUT_DIR"
RUN_LOG="$OUT_DIR/run.log"
: > "$RUN_LOG"

# Default: every suite shipped in benchmark/suites/.
if [ -z "$SUITES" ]; then
    SUITES="$(cd "$BENCH_DIR/suites" && ls *.jsonl 2>/dev/null | sed 's/\.jsonl$//' | tr '\n' ' ' | sed 's/ *$//')"
else
    SUITES="$(printf '%s' "$SUITES" | tr ',' ' ')"
fi
[ -n "$SUITES" ] || err "no suites found in $BENCH_DIR/suites/"

log_phase "Napaxi Benchmark Run"
log_sub  "run id   : $RUN_ID"
log_sub  "results  : $OUT_DIR"
log_sub  "device   : $SERIAL"
log_sub  "model    : $MODEL @ $BASE_URL"
log_sub  "suites   : $(echo $SUITES)"
log_sub  "record   : $([ "$RECORD" = "1" ] && echo "on (mp4 per case)" || echo "off")"

poll_done() {
    # adb shell "[ -f x ]" produces no stdout and the exit code does not
    # propagate; use an if/then so the marker check prints an explicit flag.
    "$ADB" -s "$SERIAL" shell \
        "if [ -f $DEVICE_DIR/result-$1.json.done ]; then echo true; else echo false; fi" \
        </dev/null 2>/dev/null | tr -d '\r' | grep -q '^true$'
}

# Runtime permissions the app declares that adb can grant. Granted after
# every (re)install so headless platform tools never block on a permission
# dialog. Unknown/unsupported permissions (device API level dependent) are
# skipped silently.
grant_permissions() {
    local perm
    for perm in \
        POST_NOTIFICATIONS READ_CONTACTS READ_CALENDAR WRITE_CALENDAR \
        CAMERA RECORD_AUDIO ACCESS_FINE_LOCATION ACCESS_COARSE_LOCATION \
        READ_EXTERNAL_STORAGE READ_MEDIA_IMAGES READ_MEDIA_VIDEO \
        BLUETOOTH_CONNECT; do
        "$ADB" -s "$SERIAL" shell pm grant "$PACKAGE" "android.permission.$perm" \
            >/dev/null 2>&1 </dev/null || true
    done
    # AppOps that gate some platform capabilities outside runtime perms.
    "$ADB" -s "$SERIAL" shell appops set "$PACKAGE" POST_NOTIFICATION allow \
        >/dev/null 2>&1 </dev/null || true
    "$ADB" -s "$SERIAL" shell appops set "$PACKAGE" SYSTEM_ALERT_WINDOW allow \
        >/dev/null 2>&1 </dev/null || true
}

reset_app() {
    if [ "$RESET_MODE" = "clear" ]; then
        "$ADB" -s "$SERIAL" shell pm clear "$PACKAGE" >/dev/null 2>&1 </dev/null || true
        grant_permissions
    else
        "$ADB" -s "$SERIAL" uninstall "$PACKAGE" >/dev/null 2>&1 </dev/null || true
        "$ADB" -s "$SERIAL" install -r "$APK" >/dev/null </dev/null
        grant_permissions
    fi
}

# Clear background processes so each case starts from a quiet device and no
# leftover app (camera, dialer, benchmark app itself) keeps resources or
# focus. Called before and after every case.
clear_device_state() {
    # Kill everything running in the background except the shell/l launcher.
    # The bench recorder is spared when recording is on so a case's video
    # spans the whole cycle.
    local pkg
    for pkg in $("$ADB" -s "$SERIAL" shell "dumpsys activity processes 2>/dev/null" \
        </dev/null | grep -oE 'processName=[a-z0-9._]+' | cut -d= -f2 \
        | grep -vE '^(com.android.systemui|com.android.phone|android|com.android.shell|system|com.huawei.android.launcher|com.android.inputmethod)' \
        | sort -u); do
        case "$pkg" in
            com.android.*|com.huawei.*|android|com.google.*) ;; # system vendors
            "$RECORDER_PACKAGE") [ "$RECORD" = "1" ] && continue ;;&
            *) "$ADB" -s "$SERIAL" shell am force-stop "$pkg" >/dev/null 2>&1 </dev/null || true ;;
        esac
    done
    "$ADB" -s "$SERIAL" shell input keyevent KEYCODE_HOME >/dev/null 2>&1 </dev/null || true
}

# --- Optional screen recording (--record) -------------------------------
# The companion recorder writes into its own external files dir (scoped
# storage); the file name carries the case id. Consent dialog is accepted
# via uiautomator coordinate tap.

recorder_files_dir="/storage/emulated/0/Android/data/$RECORDER_PACKAGE/files"

start_recording() {
    local fname="$1"
    "$ADB" -s "$SERIAL" shell am start -n "$RECORDER_PACKAGE/.RecordActivity" \
        --es out_path "$fname" >/dev/null 2>&1 </dev/null
    # The system media-projection consent dialog appears; tap 允许/继续.
    local attempt coords
    for attempt in 1 2 3 4 5 6; do
        sleep 1
        "$ADB" -s "$SERIAL" shell timeout 5 uiautomator dump /sdcard/bench_ui.xml >/dev/null 2>&1 </dev/null || true
        coords="$("$ADB" -s "$SERIAL" shell cat /sdcard/bench_ui.xml 2>/dev/null | python3 -c '
import sys, re
for m in re.finditer(r"text=\"(允许|继续)\"[^>]*bounds=\"\[(\d+),(\d+)\]\[(\d+),(\d+)\]\"", sys.stdin.read()):
    l, t, r, b = map(int, m.groups()[1:])
    print(f"{(l + r) // 2} {(t + b) // 2}")
    break
' 2>/dev/null)"
        if [ -n "$coords" ]; then
            # shellcheck disable=SC2086
            "$ADB" -s "$SERIAL" shell input tap $coords >/dev/null 2>&1 </dev/null
            break
        fi
    done
}

stop_and_pull_recording() {
    local fname="$1" dest="$2"
    # Graceful stop: force-stop kills the process without running
    # onDestroy, so MediaRecorder.stop() never fires and the mp4 is written
    # without its moov index (unplayable). The STOP action makes the service
    # stopSelf() — onDestroy then finalizes the container. force-stop follows
    # as a belt-and-braces cleanup. Every step is best-effort: a recording
    # glitch must never abort the run's measurements (set -e would otherwise
    # kill the whole harness on one failed adb call).
    "$ADB" -s "$SERIAL" shell am startservice -n "$RECORDER_PACKAGE/.RecordService" \
        -a com.napaxi.bench.recorder.STOP >/dev/null 2>&1 </dev/null || true
    sleep 4
    "$ADB" -s "$SERIAL" shell am force-stop "$RECORDER_PACKAGE" >/dev/null 2>&1 </dev/null
    if ! "$ADB" -s "$SERIAL" pull "$recorder_files_dir/$fname" "$dest" >/dev/null 2>&1 </dev/null; then
        log_warn "recording pull failed — case metrics still collected: $fname"
    fi
    "$ADB" -s "$SERIAL" shell rm -f "$recorder_files_dir/$fname" >/dev/null 2>&1 </dev/null || true
}

ensure_recorder_installed() {
    if ! "$ADB" -s "$SERIAL" shell pm path "$RECORDER_PACKAGE" >/dev/null 2>&1 </dev/null; then
        [ -f "$RECORDER_APK" ] || err "recorder APK missing: $RECORDER_APK"
        "$ADB" -s "$SERIAL" install -r "$RECORDER_APK" >/dev/null
    fi
}

# Tail the device logcat for this case's app process and echo new benchmark /
# chat-trace lines (indented, dim) to the console + run.log. Statefully keeps
# track of what was already printed via the printed-lines counter in $1.
stream_app_heartbeat() {
    local -n seen_ref="$1"
    local pid lines new_count line
    # `|| true`: pidof exits 1 once the app process is gone (e.g. killed by
    # the lowmemorykiller mid-case) and adb propagates that; under
    # pipefail + set -e an unguarded pipeline here would abort the whole
    # run instead of letting the case record its timeout.
    pid="$("$ADB" -s "$SERIAL" shell pidof "$PACKAGE" 2>/dev/null </dev/null | tr -d '\r' || true)"
    [ -z "$pid" ] && return 0
    lines="$("$ADB" -s "$SERIAL" shell logcat -d --pid="$pid" -t 400 2>/dev/null </dev/null \
        | grep -E '\[benchmark|napaxiChatTrace' | grep -vE 'stream-done|response-delta-first' || true)"
    new_count=0
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        new_count=$((new_count + 1))
        if [ "$new_count" -gt "${seen_ref:-0}" ]; then
            local body
            body="$(printf '%s' "$line" | sed -e 's/^[0-9-]* [0-9:.]* *[0-9]* *[0-9]* [IWE] flutter *//' -e 's/^: *//')"
            log_heart "$body"
        fi
    done <<< "$lines"
    seen_ref="$new_count"
}

run_case() {
    local suite="$1" case_line="$2" idx="$3" runid="$4"
    local case_id timeout payload b64 t_start t_end deadline result_file
    case_id="$(printf '%s' "$case_line" | json_get /dev/stdin "d['id']")"
    timeout="$(printf '%s' "$case_line" | json_get /dev/stdin "d.get('timeout_seconds',300)")"

    local prompt_preview
    prompt_preview="$(printf '%s' "$case_line" | json_get /dev/stdin "_preview(d['prompt'])" )" || prompt_preview=""
    log_case "($idx) $suite/$case_id"
    log_sub  "prompt   : $prompt_preview"
    log_sub  "timeout  : ${timeout}s   reset: $RESET_MODE"
    t_start=$(date +%s)
    log_step "clearing device state"
    clear_device_state
    log_step "resetting app ($RESET_MODE)"
    reset_app
    log_step "granting permissions"
    # reset_app already granted; this log line documents the phase.
    true
    if [ "$RECORD" = "1" ]; then
        log_step "starting screen recording"
        start_recording "$suite-$case_id.mp4"
    fi
    # Clear the logcat buffer so the per-case capture below holds only this
    # case's Flutter/Rust output.
    "$ADB" -s "$SERIAL" logcat -c >/dev/null 2>&1 </dev/null || true

    payload="$(python3 -c "
import json, sys
case = json.loads(sys.argv[1])
cfg = json.load(open(sys.argv[2]))
print(json.dumps({
    'run_id': sys.argv[3],
    'suite': sys.argv[4],
    'base_url': cfg.get('base_url', ''),
    'api_key': cfg.get('api_key', ''),
    'model': cfg.get('model', ''),
    'provider': cfg.get('provider', 'openai_compatible'),
    'max_tool_iterations': cfg.get('max_tool_iterations', 30),
    'response_language': cfg.get('response_language', 'zh'),
    'case': case,
}, ensure_ascii=False))" "$case_line" "$CONFIG" "$runid" "$suite")"
    b64="$(printf '%s' "$payload" | base64 -w0)"

    "$ADB" -s "$SERIAL" shell am start -W -n "$ACTIVITY" \
        --es benchmark_b64 "$b64" >/dev/null </dev/null
    # Every case runs a warm-up turn ("你好") plus an optional setup turn
    # before the measured prompt; allow their budgets on top.
    local setup_budget
    setup_budget="$(printf '%s' "$case_line" | json_get /dev/stdin "int(bool(d.get('setup_prompt'))) * $timeout")"
    local warmup_budget=$(( timeout ))
    log_step "launched; polling for result (budget ${timeout}s + warmup + setup ${setup_budget}s + 120s)"

    deadline=$(( $(date +%s) + timeout + warmup_budget + setup_budget + 120 ))
    local app_pid=""
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if poll_done "$runid"; then
            break
        fi
        # Stream the app's benchmark/chat heartbeat from logcat while waiting,
        # so the case's live progress (warm-up, prompt, tool calls) shows here.
        stream_app_heartbeat app_pid
        sleep 2
    done

    result_file="$OUT_DIR/result-$suite-$case_id.json"
    if [ "$RECORD" = "1" ]; then
        log_step "stopping screen recording"
        stop_and_pull_recording "$suite-$case_id.mp4" "$OUT_DIR/$suite-$case_id.mp4"
    fi
    # Capture this case's logcat for the run log dir: keep the Flutter app
    # output (benchmark/chat trace) and AndroidRuntime crashes, drop system noise.
    "$ADB" -s "$SERIAL" logcat -d 2>/dev/null </dev/null         | grep -E "flutter|AndroidRuntime|FATAL|napaxi"         > "$OUT_DIR/$suite-$case_id.logcat" || true
    clear_device_state
    if poll_done "$runid"; then
        "$ADB" -s "$SERIAL" pull "$DEVICE_DIR/result-$runid.json" "$result_file" >/dev/null 2>&1 </dev/null
        t_end=$(date +%s)
        python3 -c "
import json, sys
path, wall = sys.argv[1], int(sys.argv[2])
d = json.load(open(path)); d['wall_seconds'] = wall
json.dump(d, open(path, 'w'), ensure_ascii=False, indent=2)" "$result_file" $((t_end - t_start))
        summarize_case_result "$result_file" $((t_end - t_start))
    else
        t_end=$(date +%s)
        printf '{"schema":"napaxi-benchmark-result/2","run_id":"%s","suite":"%s","wall_seconds":%d,"case":{"id":"%s","prompt":%s},"metrics":{},"outcome":{"status":"timeout","error":"harness timeout","response":""}}\n' \
            "$runid" "$suite" $((t_end - t_start)) "$case_id" \
            "$(printf '%s' "$case_line" | json_get /dev/stdin "json.dumps(d['prompt'], ensure_ascii=False)")" \
            > "$result_file"
        log_fail "TIMEOUT after $((t_end - t_start))s — result marked failed"
    fi
    echo
}

# Pretty-print a finished case: score banner + per-tool-call lines pulled
# from the result JSON's trace, each colored by outcome.
summarize_case_result() {
    local result_file="$1" wall="$2"
    # Show on console (colors when tty) and append a plain copy to run.log.
    BENCH_RUN_LOG="$RUN_LOG" python3 - "$result_file" "$wall" <<'PY'
import json, os, re, sys

RESET, DIM, BOLD = "", "", ""
GREEN, YELLOW, RED, CYAN, MAGENTA = "", "", "", "", ""
if sys.stdout.isatty():
    RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"
    GREEN, YELLOW, RED, CYAN, MAGENTA = "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[35m"

_run_log = os.environ.get("BENCH_RUN_LOG", "")
_orig_print = print
def print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    if _run_log:
        plain = re.sub(r"\x1b\[[0-9;]*m", "", " ".join(str(a) for a in args))
        with open(_run_log, "a") as fh:
            _orig_print(plain, file=fh)

path, wall = sys.argv[1], int(sys.argv[2])
d = json.load(open(path))
m = d.get("metrics") or {}
score = m.get("completion_score")
grade = m.get("completion_grade") or ""
outcome = d.get("outcome") or {}
err = outcome.get("error") or ""

if score is None:
    verdict, color = "FAILED", RED
    score_s = "-"
else:
    score_s = f"{score:g}"
    if score >= 1: verdict, color = "PASS", GREEN
    elif score > 0: verdict, color = "PARTIAL", YELLOW
    else: verdict, color = "MISS", RED

dur = m.get("duration_ms")
ttft = m.get("ttft_ms")
tokens = (m.get("tokens") or {}).get("total")
calls = (m.get("tool_calls") or {})

import datetime
def stamp():
    return datetime.datetime.now().strftime("%H:%M:%S")

label = f"{verdict:<8}"
print(f"{stamp()} {color}|{label}|{RESET} {color}{BOLD}score={score_s}{RESET}" + (f"  grade={grade}" if grade else "") +
      (f"  dur={dur/1000:.1f}s" if dur else ""))
if dur is not None or tokens or calls.get("count"):
    extras = []
    if ttft: extras.append(f"ttft={ttft/1000:.1f}s")
    if tokens: extras.append(f"tokens={tokens}")
    if calls.get("count"): extras.append(f"tools={calls['count']}({calls.get('success',0)}ok)")
    extras.append(f"wall={wall}s")
    print(f"{stamp()}         {DIM}{'  '.join(extras)}{RESET}")
if err:
    print(f"{stamp()}         {RED}error: {err[:120]}{RESET}")

# Tool-call timeline from the trace.
trace = d.get("trace") or {}
for c in trace.get("tool_calls", []):
    mark = f"{DIM}?{RESET}" if c.get("is_error") is None else (
        f"{GREEN}✓{RESET}" if not c.get("is_error") else f"{RED}✗{RESET}")
    args = (c.get("arguments") or "").replace("\n", " ")[:60]
    print(f"{stamp()}         {DIM}#{c.get('seq', '?'):>2} @{c.get('offset_ms', 0):>6}ms{RESET} "
          f"{mark} {CYAN}{c.get('name', '?')}{RESET} {DIM}{args}{RESET}")
resp = (outcome.get("response") or "").replace("\n", " ")
if resp:
    print(f"{stamp()}         {DIM}reply: {resp[:100]}{RESET}")
PY
}

if [ "$RECORD" = "1" ]; then
    ensure_recorder_installed
fi

idx=0
for suite in $SUITES; do
    suite_file="$BENCH_DIR/suites/$suite.jsonl"
    [ -f "$suite_file" ] || err "suite not found: $suite_file"
    suite_count="$(grep -c . "$suite_file" 2>/dev/null || true)"
    log_phase "Suite: $suite (${suite_count:-?} cases)"
    while IFS= read -r case_line <&3; do
        [ -z "$case_line" ] && continue
        case_id="$(printf '%s' "$case_line" | json_get /dev/stdin "d['id']")"
        if [ -n "$CASES_FILTER" ] && ! [[ ",$CASES_FILTER," == *",$case_id,"* ]]; then
            continue
        fi
        idx=$((idx + 1))
        run_case "$suite" "$case_line" "$idx" "$RUN_ID-$idx-$case_id"
    done 3< "$suite_file"
done

[ "$idx" -gt 0 ] || err "no cases selected"

log_phase "Aggregating $idx results"
python3 "$BENCH_DIR/aggregate.py" "$OUT_DIR" >/dev/null

log_ok  "report  : $OUT_DIR/report.md"
log_ok  "run log : $OUT_DIR/run.log"
log_sub "logcat  : $OUT_DIR/<suite>-<case>.logcat (one per case)"
