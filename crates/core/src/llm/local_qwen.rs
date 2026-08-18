//! On-device local LLM backend (CPU-only Qwen2 GGUF via vendored `candle`).
//!
//! This is a *first-class provider*: the dispatch layer routes
//! `provider == "local"` (capability `napaxi.llm.local`) here, and the turn
//! then flows through the normal `run_tool_loop` — so local turns get the same
//! skill-gating, compaction, streaming, diagnostics, and context-overflow
//! recovery as cloud turns. There is no parallel "fast path".
//!
//! The adapter is text-protocol: it renders the system prompt, message history,
//! and tool catalogue into a Qwen2.5 ChatML prompt, asks the model to emit
//! `<tool_call>{...}</tool_call>` blocks, and parses those back into
//! [`LlmToolCall`]s — letting the local model participate in tool loops even
//! though candle has no native function-calling API.
//!
//! `Qwen::generate_raw` is synchronous and CPU-bound, so generation runs on a
//! blocking task via [`tokio::task::spawn_blocking`]. The loaded model is held
//! in a process-global singleton loaded lazily on first use.

use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use candle_qwen::{GenerationConfig, Qwen};
use serde_json::Value;

use super::{LlmStreamEvent, LlmToolCall, LlmTurn};
use crate::tool_registry::ToolDescriptor;
use crate::types::PlatformLlmConfig;

static LOCAL_QWEN: OnceLock<Mutex<Option<Qwen>>> = OnceLock::new();
static FILES_DIR: OnceLock<String> = OnceLock::new();

const DEFAULT_MODEL_FILENAME: &str = "qwen2.5-0_5b-instruct-q4_k_m.gguf";
const DEFAULT_TOKENIZER_FILENAME: &str = "qwen2.5-tokenizer.json";

pub(crate) fn set_files_dir(files_dir: &str) {
    let _ = FILES_DIR.set(files_dir.to_string());
}

/// Extract the bundled tokenizer asset to `<files_dir>/local-llm/` on first use.
/// The tokenizer is committed as a small Android asset; the multi-hundred-MB
/// GGUF model is sideloaded to the same dir by the developer (too large for the
/// APK / git). The files are written by the app, so they're app-owned and
/// readable on the device — unlike files pushed via `adb` (owned by `shell`),
/// which the FUSE layer hides from the app process.
#[cfg(target_os = "android")]
fn ensure_local_llm_files() {
    let Some(dir) = FILES_DIR.get() else {
        return;
    };
    let local_dir = std::path::Path::new(dir).join("local-llm");
    let _ = crate::android_assets::extract_asset_to_file(
        DEFAULT_TOKENIZER_FILENAME,
        &local_dir.join(DEFAULT_TOKENIZER_FILENAME),
    );
    // The multi-hundred-MB GGUF is sideloaded straight into `<files_dir>/local-llm/`
    // by the host-side download channel (see the demo's MainActivity): every
    // `adb push` staging location is owned by `shell`/the media uid and hidden
    // from this app process by the FUSE layer, so copying from shared storage
    // does not work — the file has to be written by the app itself.
}

/// Resolve model + tokenizer paths. Prefers the explicitly-configured path, but
/// only if it actually exists on disk; otherwise falls back to
/// `<files_dir>/local-llm/<default>`. The fallback lets us sideload the model to
/// the app's internal dir without changing the config.
fn resolve_paths(config: &PlatformLlmConfig) -> Result<(String, String)> {
    let model_path = resolve_one_path(
        config.local_llm.model_path.as_deref(),
        DEFAULT_MODEL_FILENAME,
    );
    let tokenizer_path = resolve_one_path(
        config.local_llm.tokenizer_path.as_deref(),
        DEFAULT_TOKENIZER_FILENAME,
    );
    match (model_path, tokenizer_path) {
        (Some(m), Some(t)) => Ok((m, t)),
        _ => anyhow::bail!(
            "local LLM files not found (configured model_path={:?}, files_dir={:?})",
            config.local_llm.model_path,
            FILES_DIR.get(),
        ),
    }
}

/// Pick a path: the configured one if it exists, else the files_dir fallback.
fn resolve_one_path(configured: Option<&str>, fallback_name: &str) -> Option<String> {
    if let Some(p) = configured
        .filter(|s| !s.trim().is_empty())
        .filter(|s| std::path::Path::new(s).exists())
    {
        return Some(p.to_string());
    }
    let dir = FILES_DIR.get()?;
    let p = std::path::Path::new(dir).join("local-llm").join(fallback_name);
    p.exists().then(|| p.to_string_lossy().into_owned())
}

/// Run the loaded model against a pre-built ChatML `prompt`. Generation is
/// CPU-bound, so it happens on a blocking thread; the singleton model is loaded
/// lazily on the first call.
async fn generate(
    config: &PlatformLlmConfig,
    prefix: String,
    suffix: String,
    max_new_tokens: usize,
) -> Result<String> {
    let mut gen_cfg = generation_config_from(config);
    gen_cfg.max_new_tokens = max_new_tokens;
    // Extract the bundled tokenizer asset to files_dir on first use (one-time).
    #[cfg(target_os = "android")]
    {
        let _ = tokio::task::spawn_blocking(ensure_local_llm_files).await;
    }
    let (model_path, tokenizer_path) = resolve_paths(config)?;
    tokio::task::spawn_blocking(move || -> Result<String> {
        let mut guard = LOCAL_QWEN
            .get_or_init(|| Mutex::new(None))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(Qwen::from_files(&model_path, &tokenizer_path)?);
        }
        let qwen = guard.as_mut().expect("local model just loaded");
        // Cache the constant prefix's KV across turns; only the suffix is
        // re-prefilled. `generate_raw` would re-prefill the whole prompt each
        // call (and `generate` would mis-wrap it in a single user turn).
        qwen.generate_prompt_cached(&prefix, &suffix, &gen_cfg)
    })
    .await?
}

/// Load the model (if not yet loaded) and prefill the constant system+tools
/// prefix so its KV cache is warm before the first real turn. Returns the
/// cached prefix token length. Same blocking-thread pattern as [`generate`];
/// callers on Android should ensure the tokenizer asset is staged first.
pub(crate) async fn warmup_prefix(
    config: &PlatformLlmConfig,
    prefix: String,
) -> Result<usize> {
    #[cfg(target_os = "android")]
    {
        let _ = tokio::task::spawn_blocking(ensure_local_llm_files).await;
    }
    let (model_path, tokenizer_path) = resolve_paths(config)?;
    tokio::task::spawn_blocking(move || -> Result<usize> {
        let mut guard = LOCAL_QWEN
            .get_or_init(|| Mutex::new(None))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(Qwen::from_files(&model_path, &tokenizer_path)?);
        }
        let qwen = guard.as_mut().expect("local model just loaded");
        qwen.prefill_prefix(&prefix)
    })
    .await?
}

/// Non-streaming local completion.
pub(super) async fn complete_raw(
    config: &PlatformLlmConfig,
    messages: &[Value],
    tools: &[ToolDescriptor],
) -> Result<LlmTurn> {
    // Split the prompt into a constant prefix (system + tools) and a per-turn
    // suffix (history + user + assistant primer). The candle backend keeps the
    // prefix's KV cache warm across turns, so only the suffix is re-prefilled.
    let (prefix, suffix) = build_prompt_parts(config, messages, tools);
    dump_prompt_debug(&format!("{prefix}{suffix}"), tools.len());
    let text = generate(config, prefix, suffix, config.local_llm.max_new_tokens).await?;
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    Ok(parse_local_turn_with_names(&text, &tool_names))
}


/// DEBUG: persist the exact prompt the local model receives, so we can measure
/// prompt size (prefill cost) on-device. Written before generation, so it's
/// available even while a slow turn is still running. Remove after tuning.
fn dump_prompt_debug(prompt: &str, tool_count: usize) {
    let Some(dir) = FILES_DIR.get() else {
        return;
    };
    let path = std::path::Path::new(dir)
        .join("local-llm")
        .join("_last_prompt.txt");
    let header = format!(
        "tool_count={}\nchar_count={}\nbyte_count={}\napprox_tokens={}\n---\n",
        tool_count,
        prompt.chars().count(),
        prompt.len(),
        prompt.chars().count() / 2,
    );
    let _ = std::fs::write(&path, format!("{header}{prompt}"));
}

/// Streaming local completion. candle generation is not incremental, so this is
/// pseudo-streaming: generate the full turn, then emit its content as a single
/// delta. Tool-call deltas are not streamed (the caller learns of tool calls
/// only when the turn completes), matching how cloud providers surface a
/// non-streaming function call.
pub(super) async fn stream_raw_cancelable<F, C>(
    config: &PlatformLlmConfig,
    messages: &[Value],
    tools: &[ToolDescriptor],
    mut on_event: F,
    mut should_cancel: C,
) -> Result<LlmTurn>
where
    F: FnMut(LlmStreamEvent),
    C: FnMut() -> bool,
{
    if should_cancel() {
        anyhow::bail!("Chat cancelled");
    }
    let turn = complete_raw(config, messages, tools).await?;
    if !turn.content.is_empty() {
        on_event(LlmStreamEvent::ResponseDelta(turn.content.clone()));
    }
    Ok(turn)
}

// ── Prompt assembly ──────────────────────────────────────────────────────

/// Assemble a Qwen2.5 ChatML prompt split into a constant prefix (the system
/// and tool catalogue, identical across turns) and a per-turn suffix (recent
/// history plus the assistant primer). The split is at the `<|im_end|>\n` then
/// `<|im_start|>` boundary (a clean special-token boundary), so the candle
/// backend can cache the prefix's KV across turns.
fn build_prompt_parts(
    config: &PlatformLlmConfig,
    messages: &[Value],
    tools: &[ToolDescriptor],
) -> (String, String) {
    let prefix = build_prefix(config, tools);
    let suffix = format!("{}<|im_start|>assistant\n", render_history(messages));
    (prefix, suffix)
}

/// Assemble the full ChatML prompt (prefix + suffix). Used by tests.
#[cfg(test)]
fn build_local_prompt(
    config: &PlatformLlmConfig,
    messages: &[Value],
    tools: &[ToolDescriptor],
) -> String {
    let (prefix, suffix) = build_prompt_parts(config, messages, tools);
    format!("{prefix}{suffix}")
}

/// The system-prefix turn. Local 0.5B models get a deliberately *short*, fixed
/// system prompt plus a *condensed* tool list — see the notes inline. This is
/// intentionally NOT the SDK's prepared `system_prompt`: that is a cloud-oriented
/// onboarding/behavior script (1500+ chars of scenario + skills) which a 0.5B
/// model cannot follow and which dominates prefill (every prompt token is a
/// forward pass on the CPU). `config` is accepted for signature symmetry only.
pub(crate) fn build_prefix(_config: &PlatformLlmConfig, tools: &[ToolDescriptor]) -> String {
    let mut s = String::new();
    s.push_str("<|im_start|>system\n");
    s.push_str(
        "You are a helpful assistant running on-device. 回答简洁，使用中文。\
         需要使用工具时，严格按下面的格式调用。",
    );
    if !tools.is_empty() {
        // Minimal catalogue: "name(params)" per line, ALL tools included. Tool
        // names are self-explanatory for the demo's set (verified per-tool), so
        // descriptions are omitted entirely — prefill on a phone CPU is the
        // dominant per-turn cost (~17 tok/s), and the ~2000-char description
        // block was half the prefix. The lenient output parser recovers
        // arguments from what the model emits.
        s.push_str("\n\n# Tools\n\n可用工具（需要时调用，每次只调一个）：\n");
        for t in tools {
            let required: Vec<&str> = t
                .parameters
                .get("required")
                .and_then(|r| r.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();
            let params = if required.is_empty() {
                String::new()
            } else {
                format!("({})", required.join(", "))
            };
            s.push_str(&format!("- {}{}\n", t.name, params));
        }
        // 0.5B models reliably *recognize* a tool but then describe it in prose
        // ("I'll use set_alarm…") instead of emitting the <tool_call> block, and
        // sometimes mistake a parameter name for the tool name. A hard format
        // nudge + a worked example naming a real tool from the list above (with
        // its real parameter) is the highest-leverage fix short of constrained
        // decoding (which candle lacks).
        let first_tool = tools.first();
        let example = match first_tool {
            Some(t) => {
                let first_param = t
                    .parameters
                    .get("required")
                    .and_then(|r| r.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|v| v.as_str())
                    .unwrap_or("key");
                format!(
                    "{{\"name\": \"{}\", \"arguments\": {{\"{}\": \"value\"}}}}",
                    t.name, first_param
                )
            }
            None => "{\"name\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}".to_string(),
        };
        s.push_str(&format!(
            "\n调用工具时，整条回复只放一个 <tool_call> 块，不要任何解释。\
             name 必须从上面列表里选，arguments 里填参数。\n\
             没有合适工具时直接简短回答用户。\n\
             格式示例：\n\
             <tool_call>\n{example}\n</tool_call>\n",
        ));
    }
    s.push_str("<|im_end|>\n");
    s
}

/// Render the last few messages as ChatML turns. History is windowed to the
/// most recent `HISTORY_WINDOW` turns to fit the small model's context.
fn render_history(messages: &[Value]) -> String {
    const HISTORY_WINDOW: usize = 6;
    let mut s = String::new();
    let start = messages.len().saturating_sub(HISTORY_WINDOW);
    for m in &messages[start..] {
        match m.get("role").and_then(Value::as_str) {
            Some("user") => {
                s.push_str("<|im_start|>user\n");
                if let Some(c) = m.get("content").and_then(Value::as_str) {
                    s.push_str(c);
                }
                s.push_str("<|im_end|>\n");
            }
            Some("assistant") => {
                s.push_str("<|im_start|>assistant\n");
                if let Some(c) = m.get("content").and_then(Value::as_str)
                    && !c.is_empty()
                {
                    s.push_str(c);
                    s.push('\n');
                }
                if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
                    for c in calls {
                        let name = c
                            .pointer("/function/name")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        let args = c
                            .pointer("/function/arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("{}");
                        s.push_str(&format!(
                            "<tool_call>{{\"name\":\"{name}\",\"arguments\":{args}}}</tool_call>\n"
                        ));
                    }
                }
                s.push_str("<|im_end|>\n");
            }
            Some("tool") => {
                // Qwen2.5 renders tool results as a plain user turn (no
                // <tool_result> wrapper) — the call→result link is positional.
                let content = m.get("content").and_then(Value::as_str).unwrap_or("");
                s.push_str(&format!("<|im_start|>user\n\n{content}<|im_end|>\n"));
            }
            _ => {}
        }
    }
    s
}

// ── Output parsing ───────────────────────────────────────────────────────

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// Parse the model's text output into an [`LlmTurn`]. Scans for
/// `<tool_call>…</tool_call>` spans; text outside spans becomes `content`.
/// Malformed spans are skipped (graceful degradation → the tool loop exits
/// cleanly when the model just answers normally).
///
/// Small models often deviate from the `<tool_call>` protocol, so two lenient
/// fallbacks run when no tagged call is found: a function-call notation
/// `tool_name({...})` (matched against `tool_names`), then a bare JSON object /
/// array carrying a tool call.
fn parse_local_turn_with_names(raw: &str, tool_names: &[&str]) -> LlmTurn {
    let mut tool_calls = Vec::new();
    let mut content = String::new();
    let mut rest = raw;
    while let Some(open_rel) = rest.find(TOOL_CALL_OPEN) {
        content.push_str(&rest[..open_rel]);
        let after_open = &rest[open_rel + TOOL_CALL_OPEN.len()..];
        let (body, after) = match after_open.find(TOOL_CALL_CLOSE) {
            Some(c) => (&after_open[..c], &after_open[c + TOOL_CALL_CLOSE.len()..]),
            None => (after_open, ""),
        };
        if let Some(call) = parse_tool_call_json(body.trim(), tool_calls.len()) {
            tool_calls.push(call);
        }
        rest = after;
    }
    content.push_str(rest);
    // Fallback 1: function-call notation `set_alarm({...})` — matched against
    // known tool names so ordinary prose isn't mistaken for a call.
    if tool_calls.is_empty()
        && !tool_names.is_empty()
        && let Some((named, stripped)) = extract_named_tool_calls(&content, tool_names)
    {
        tool_calls = named;
        content = stripped;
    }
    // Fallback 2: a bare JSON object/array carrying a tool call (some models
    // emit `{"name":"x","parameters":{...}}` with no wrapper).
    if tool_calls.is_empty()
        && let Some((bare, stripped)) = extract_bare_tool_calls(&content)
    {
        tool_calls = bare;
        content = stripped;
    }
    LlmTurn {
        content: content.trim().to_string(),
        reasoning_content: None,
        tool_calls,
        usage: None,
    }
}

fn parse_tool_call_json(body: &str, idx: usize) -> Option<LlmToolCall> {
    let v: Value = serde_json::from_str(body).ok()?;
    parse_tool_call_value(&v, idx)
}

/// Build a tool call from a parsed JSON object. Accepts `arguments`,
/// `parameters`, or `params` as the argument key (small models are
/// inconsistent). Returns `None` unless the object has a non-empty `name`.
fn parse_tool_call_value(v: &Value, idx: usize) -> Option<LlmToolCall> {
    let name = v.get("name")?.as_str()?.trim();
    if name.is_empty() {
        return None;
    }
    let arguments = v
        .get("arguments")
        .or_else(|| v.get("parameters"))
        .or_else(|| v.get("params"));
    let arguments = match arguments {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => "{}".to_string(),
    };
    Some(LlmToolCall {
        id: format!("call_local_{idx}"),
        name: name.to_string(),
        arguments,
    })
}

/// Locate the first balanced `{ ... }` / `[ ... ]` substring in `text` and, if
/// it parses to a tool-call object (or an array of them), return the calls plus
/// the text with that JSON removed. Used when a model emits a function call
/// without the `<tool_call>` wrapper.
fn extract_bare_tool_calls(text: &str) -> Option<(Vec<LlmToolCall>, String)> {
    let bytes = text.as_bytes();
    let open = bytes.iter().position(|&b| b == b'[' || b == b'{')?;
    let open_ch = bytes[open];
    let close_ch = if open_ch == b'[' { b']' } else { b'}' };

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    let mut close = None;
    for (i, &b) in bytes[open..].iter().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if in_string {
            if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            c if c == open_ch => depth += 1,
            c if c == close_ch => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let value: Value = serde_json::from_str(&text[open..=close]).ok()?;
    let objects: Vec<&Value> = match &value {
        Value::Array(items) => items.iter().collect(),
        Value::Object(_) => vec![&value],
        _ => return None,
    };
    let mut calls = Vec::new();
    for (i, obj) in objects.iter().enumerate() {
        if let Some(call) = parse_tool_call_value(obj, i) {
            calls.push(call);
        }
    }
    if calls.is_empty() {
        return None;
    }
    let mut stripped = String::with_capacity(text.len());
    stripped.push_str(&text[..open]);
    stripped.push_str(&text[close + 1..]);
    Some((calls, stripped))
}

/// Detect a function-call notation like `set_alarm({"time": "08:00"})` — a
/// known tool name immediately followed by `(JSON)` or `{JSON}`. Only names in
/// `tool_names` match, so ordinary prose isn't misread. Returns the call plus
/// the text with that span removed.
fn extract_named_tool_calls(text: &str, tool_names: &[&str]) -> Option<(Vec<LlmToolCall>, String)> {
    // Earliest word-bounded "name(" / "name{" for any known tool.
    let mut best: Option<(usize, &str)> = None;
    for name in tool_names {
        let Some(name_bytes) = name.as_bytes().first() else {
            continue;
        };
        if !name_bytes.is_ascii_alphanumeric() && *name_bytes != b'_' {
            continue;
        }
        let mut search = 0;
        while search < text.len() {
            let Some(rel) = text[search..].find(name) else { break };
            let abs = search + rel;
            let word_before = text[..abs].ends_with(|c: char| c.is_alphanumeric() || c == '_');
            let after = abs + name.len();
            let opener = text[after..].chars().next();
            if !word_before && matches!(opener, Some('(' | '{')) {
                if best.is_none_or(|(b, _)| abs < b) {
                    best = Some((abs, name));
                }
                break;
            }
            search = abs + name.len().max(1);
        }
    }
    let (name_start, name) = best?;
    let after_name = name_start + name.len();
    let open_ch = text[after_name..].chars().next()?;
    let close_ch = if open_ch == '(' { ')' } else { '}' };

    // Balanced scan for the matching close bracket.
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    let mut close = None;
    for (i, &b) in bytes[after_name..].iter().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if in_string {
            if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            _ if b == open_ch as u8 => depth += 1,
            _ if b == close_ch as u8 => {
                depth -= 1;
                if depth == 0 {
                    close = Some(after_name + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let inner = text[after_name + 1..close].trim();
    // The inner content should be the arguments JSON object. Accept it only if
    // it parses (or is empty).
    let arguments = if inner.is_empty() {
        "{}".to_string()
    } else {
        serde_json::from_str::<Value>(inner).ok()?;
        inner.to_string()
    };
    let call = LlmToolCall {
        id: "call_local_0".to_string(),
        name: name.to_string(),
        arguments,
    };
    let mut stripped = String::with_capacity(text.len());
    stripped.push_str(&text[..name_start]);
    stripped.push_str(&text[close + 1..]);
    Some((vec![call], stripped))
}

fn generation_config_from(config: &PlatformLlmConfig) -> GenerationConfig {
    GenerationConfig {
        temperature: config.local_llm.temperature,
        top_p: None,
        top_k: None,
        max_new_tokens: config.local_llm.max_new_tokens,
        seed: config.local_llm.seed,
        repeat_penalty: config.local_llm.repeat_penalty,
        repeat_last_n: config.local_llm.repeat_last_n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_registry::ToolEffect;
    use serde_json::json;

    // ── parsing ──────────────────────────────────────────────────────────

    #[test]
    fn parses_a_well_formed_tool_call() {
        let turn = parse_local_turn_with_names(
            "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"/x\"}}</tool_call>",
            &[],
        );
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "read_file");
        assert_eq!(turn.tool_calls[0].arguments, r#"{"path":"/x"}"#);
        assert!(turn.content.is_empty());
    }

    #[test]
    fn parses_multiple_calls_and_surrounding_text() {
        let turn = parse_local_turn_with_names(
            "Sure.\n<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\n\
             <tool_call>{\"name\":\"b\",\"arguments\":{\"q\":1}}</tool_call>",
            &[],
        );
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.content, "Sure.");
    }

    #[test]
    fn arguments_as_string_is_accepted() {
        let turn = parse_local_turn_with_names(
            "<tool_call>{\"name\":\"x\",\"arguments\":\"{\\\"k\\\":1}\"}</tool_call>",
            &[],
        );
        assert_eq!(turn.tool_calls[0].arguments, r#"{"k":1}"#);
    }

    #[test]
    fn missing_close_tag_takes_to_end() {
        let turn = parse_local_turn_with_names(
            "<tool_call>{\"name\":\"x\",\"arguments\":{}}",
            &[],
        );
        assert_eq!(turn.tool_calls.len(), 1);
    }

    #[test]
    fn malformed_json_span_is_skipped() {
        let turn = parse_local_turn_with_names("<tool_call>not json</tool_call>", &[]);
        assert!(turn.tool_calls.is_empty());
    }

    #[test]
    fn plain_text_with_no_tool_call_degrades_gracefully() {
        let turn = parse_local_turn_with_names("The answer is 42.", &[]);
        assert!(turn.tool_calls.is_empty());
        assert_eq!(turn.content, "The answer is 42.");
    }

    #[test]
    fn empty_output_yields_empty_turn() {
        let turn = parse_local_turn_with_names("", &[]);
        assert!(turn.tool_calls.is_empty());
        assert!(turn.content.is_empty());
    }

    #[test]
    fn parameters_key_is_accepted_as_arguments() {
        // Small models often emit `parameters` instead of `arguments`.
        let turn = parse_local_turn_with_names(
            "<tool_call>{\"name\":\"set_alarm\",\"parameters\":{\"time\":\"08:00\"}}</tool_call>",
            &[],
        );
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "set_alarm");
        assert_eq!(turn.tool_calls[0].arguments, r#"{"time":"08:00"}"#);
    }

    #[test]
    fn bare_json_object_is_lifted_as_tool_call() {
        // No `<tool_call>` wrapper — the model just emitted the JSON object.
        let turn = parse_local_turn_with_names(
            "{\"name\":\"set_alarm\",\"parameters\":{\"time\":\"08:00\"}}好的，已设定。",
            &[],
        );
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "set_alarm");
        assert_eq!(turn.content, "好的，已设定。");
    }

    #[test]
    fn bare_json_array_is_lifted_as_tool_calls() {
        let turn = parse_local_turn_with_names(
            "[{\"name\":\"a\",\"arguments\":{}},{\"name\":\"b\",\"params\":{\"q\":1}}]done",
            &[],
        );
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.tool_calls[0].name, "a");
        assert_eq!(turn.tool_calls[1].name, "b");
        assert_eq!(turn.content, "done");
    }

    #[test]
    fn prose_json_without_name_is_not_mistaken_for_a_call() {
        let turn = parse_local_turn_with_names(
            "The config is {\"a\":1,\"b\":2}, no name field.",
            &[],
        );
        assert!(turn.tool_calls.is_empty());
        assert!(turn.content.contains("no name field"));
    }

    #[test]
    fn named_function_call_notation_is_lifted() {
        // The 0.5B model often emits `set_alarm({...})` instead of the
        // `<tool_call>` wrapper. Match it against known tool names.
        let turn = parse_local_turn_with_names(
            "set_alarm({\"time\": \"08:00\"})",
            &["set_alarm", "create_calendar_event"],
        );
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "set_alarm");
        assert_eq!(turn.tool_calls[0].arguments, "{\"time\": \"08:00\"}");
        assert!(turn.content.is_empty());
    }

    #[test]
    fn named_function_call_with_surrounding_text() {
        let turn = parse_local_turn_with_names(
            "好的，正在设置。set_alarm({\"time\":\"08:00\"})完成。",
            &["set_alarm"],
        );
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "set_alarm");
        assert!(turn.content.contains("好的"));
        assert!(turn.content.contains("完成"));
    }

    #[test]
    fn unknown_name_in_function_notation_is_not_a_call() {
        let turn =
            parse_local_turn_with_names("foobar({\"a\":1})", &["set_alarm"]);
        assert!(turn.tool_calls.is_empty());
    }

    // ── prompt assembly (against the REAL builder) ───────────────────────

    fn tool(name: &str, params: Value) -> ToolDescriptor {
        ToolDescriptor {
            name: name.to_string(),
            description: format!("{name} tool"),
            parameters: params,
            effect: ToolEffect::Read,
        }
    }

    #[test]
    fn prompt_injects_condensed_tool_catalogue() {
        let config = PlatformLlmConfig::default();
        let tools = vec![tool(
            "read_file",
            json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
        )];
        let prompt = build_local_prompt(&config, &[], &tools);
        // System turn + minimal tool list (name + params, no description) + primer.
        assert!(prompt.contains("<|im_start|>system\n"));
        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("- read_file(path)"));
        // No full JSON schema and no description text.
        assert!(!prompt.contains("\"function\""));
        assert!(!prompt.contains("read_file tool"));
        assert!(prompt.contains("<tool_call>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn prompt_includes_all_tools_not_capped() {
        let config = PlatformLlmConfig::default();
        let tools: Vec<ToolDescriptor> = (0..30)
            .map(|i| tool(&format!("tool_{i}"), json!({})))
            .collect();
        let prompt = build_local_prompt(&config, &[], &tools);
        assert!(prompt.contains("- tool_0"));
        // Previously a `take(24)` cap dropped tools beyond index 23.
        assert!(prompt.contains("- tool_29"));
    }

    #[test]
    fn prompt_ignores_cloud_prepared_system_prompt() {
        // The local builder must NOT inject the SDK's prepared (cloud-oriented)
        // system_prompt — it's a multi-thousand-char onboarding script that
        // bloats prefill and that a 0.5B model can't follow.
        let config = PlatformLlmConfig {
            system_prompt: "X".repeat(3000),
            ..PlatformLlmConfig::default()
        };
        let prompt = build_local_prompt(&config, &[], &[]);
        assert!(!prompt.contains('X'));
        assert!(prompt.contains("helpful assistant"));
    }

    #[test]
    fn prompt_renders_history_and_tool_results() {
        let messages = vec![
            json!({"role":"user","content":"list"}),
            json!({"role":"assistant","tool_calls":[
                {"id":"c1","function":{"name":"ls","arguments":"{\"dir\":\"/\"}"}}
            ]}),
            json!({"role":"tool","tool_call_id":"c1","content":"a.txt"}),
        ];
        let prompt = build_local_prompt(&PlatformLlmConfig::default(), &messages, &[]);
        // Assistant tool call is rendered back as a <tool_call> block.
        assert!(prompt
            .contains("<tool_call>{\"name\":\"ls\",\"arguments\":{\"dir\":\"/\"}}</tool_call>"));
        // Tool result is rendered as a plain user turn carrying the content.
        assert!(prompt.contains("a.txt"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}

#[cfg(test)]
mod host_model_tests {
    use super::*;

    #[test]
    fn local_config_json_parses_and_routes() {
        // Exactly what the demo's toSdkConfig emits for the local profile.
        let json = r#"{"provider":"local","api_key":"","model":"local","system_prompt":"x","response_language":"zh","max_tokens":8192,"max_tool_iterations":0,"context_engine":{},"shell_security":{},"local_llm":{}}"#;
        let config: crate::types::PlatformLlmConfig = serde_json::from_str(json)
            .expect("local wire config must parse");
        assert_eq!(config.provider, "local");
        assert!(config.local_llm.temperature - 0.1 < 1e-9);
        let route =
            crate::capabilities::resolve_llm_provider("local").expect("route resolves");
        assert!(matches!(route, crate::capabilities::LlmProviderRoute::Local));
    }

    #[test]
    #[ignore = "requires the Qwen2.5 q4_k_m GGUF staged under /tmp/napaxi-local-test"]
    fn host_warmup_then_two_turns_reuse_prefix() {
        set_files_dir("/tmp/napaxi-local-test/files");
        let mut config = PlatformLlmConfig::default();
        config.provider = "local".to_string();
        let tools = vec![crate::tool_registry::ToolDescriptor {
            name: "shell".to_string(),
            description: "run a shell command".to_string(),
            parameters: serde_json::json!({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}),
            effect: Default::default(),
        }];
        let prefix = build_prefix(&config, &tools);
        // Warm-up: prefill the prefix once.
        let warmed = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(warmup_prefix(&config, prefix.clone()))
            .expect("warmup");
        assert!(warmed > 0, "prefix token count: {warmed}");

        // Turn 1 (hits the warmed prefix) and turn 2 (hits turn 1's truncated
        // cache): both must produce a well-formed shell call.
        for prompt in [
            "请使用参数 command=\"uname -r\" 调用工具 shell。",
            "请使用参数 command=\"pwd\" 调用工具 shell。",
        ] {
            let messages = vec![serde_json::json!({
                "role": "user",
                "content": prompt,
            })];
            let turn = tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(complete_raw(&config, &messages, &tools))
                .expect("local turn");
            assert!(
                turn.tool_calls.iter().any(|call| call.name == "shell"),
                "expected shell call, got content={:?} calls={:?}",
                turn.content,
                turn.tool_calls,
            );
        }
    }

    #[test]
    #[ignore = "requires the Qwen2.5 q4_k_m GGUF staged under /tmp/napaxi-local-test"]
    fn host_tool_call_generation() {
        set_files_dir("/tmp/napaxi-local-test/files");
        let mut config = PlatformLlmConfig::default();
        config.provider = "local".to_string();
        let messages = vec![serde_json::json!({
            "role": "user",
            "content": "请使用参数 command=\"uname -r\" 调用工具 shell。",
        })];
        let tools = vec![
            crate::tool_registry::ToolDescriptor {
                name: "shell".to_string(),
                description: "run a shell command".to_string(),
                parameters: serde_json::json!({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}),
                effect: Default::default(),
            },
        ];
        let turn = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(complete_raw(&config, &messages, &tools))
            .expect("local turn");
        println!("content: {:?}", turn.content);
        println!("tool_calls: {:#?}", turn.tool_calls);
        assert!(!turn.content.trim().is_empty() || !turn.tool_calls.is_empty(), "empty turn");
        println!("output: {}", turn.content);
    }
}

#[cfg(test)]
mod kv_deadlock_tests {
    use super::*;

    /// Mimic the FFI thread calling the warm-up handle: a plain tokio runtime
    /// blocking on the warm-up, then a normal turn — reproduces the on-device
    /// deadlock setup (or proves it clean).
    #[test]
    #[ignore = "requires the Qwen2.5 q4_k_m GGUF staged under /tmp/napaxi-local-test"]
    fn warmup_then_turn_no_deadlock() {
        set_files_dir("/tmp/napaxi-local-test/files");
        let mut config = PlatformLlmConfig::default();
        config.provider = "local".to_string();
        let tools = vec![crate::tool_registry::ToolDescriptor {
            name: "shell".to_string(),
            description: "run".to_string(),
            parameters: serde_json::json!({}),
            effect: Default::default(),
        }];
        let prefix = build_prefix(&config, &tools);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let warmed = rt.block_on(warmup_prefix(&config, prefix)).expect("warm");
        assert!(warmed > 0);
        // Turn on the SAME runtime (like the FRB worker reusing one runtime).
        let messages = vec![serde_json::json!({"role":"user","content":"你好"})];
        let turn = rt
            .block_on(complete_raw(&config, &messages, &tools))
            .expect("turn after warmup");
        assert!(!turn.content.is_empty());
    }
}
