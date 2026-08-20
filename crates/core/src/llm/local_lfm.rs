//! On-device local LLM backend (CPU-only LFM2 / LFM2.5 GGUF via vendored `candle`).
//!
//! Renders the system prompt, message history, and tool catalogue into LFM
//! ChatML, asks the model to emit `<|tool_call_start|>[fn(...)]<|tool_call_end|>`
//! blocks, and parses those back into [`LlmToolCall`]s.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Result;
use candle_lfm::{GenerationConfig, Lfm};
use serde_json::Value;

use super::{LlmStreamEvent, LlmToolCall, LlmTurn};
use crate::tool_registry::ToolDescriptor;
use crate::types::PlatformLlmConfig;

static LOCAL_LFM: OnceLock<Mutex<Option<Lfm>>> = OnceLock::new();
static FILES_DIR: OnceLock<String> = OnceLock::new();

const DEFAULT_MODEL_FILENAME: &str = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf";
const DEFAULT_TOKENIZER_FILENAME: &str = "tokenizer.json";
const TOOL_CALL_OPEN: &str = "<|tool_call_start|>";
const TOOL_CALL_CLOSE: &str = "<|tool_call_end|>";

pub(crate) fn set_files_dir(files_dir: &str) {
    let _ = FILES_DIR.set(files_dir.to_string());
}

fn ensure_loaded(config: &PlatformLlmConfig) -> Result<()> {
    let mut guard = LOCAL_LFM
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.is_some() {
        return Ok(());
    }
    #[cfg(target_os = "android")]
    ensure_local_llm_files();
    let (model_path, tokenizer_path) = resolve_paths(config)?;
    *guard = Some(Lfm::from_files(&model_path, &tokenizer_path)?);
    Ok(())
}

#[cfg(target_os = "android")]
fn ensure_local_llm_files() {
    let Some(dir) = FILES_DIR.get() else {
        return;
    };
    let local_dir = std::path::Path::new(dir).join("local-llm");
    let _ = crate::android_assets::extract_asset_to_file(
        DEFAULT_MODEL_FILENAME,
        &local_dir.join(DEFAULT_MODEL_FILENAME),
    );
    let _ = crate::android_assets::extract_asset_to_file(
        DEFAULT_TOKENIZER_FILENAME,
        &local_dir.join(DEFAULT_TOKENIZER_FILENAME),
    );
}

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

fn resolve_one_path(configured: Option<&str>, fallback_name: &str) -> Option<String> {
    if let Some(p) = configured.filter(|s| !s.trim().is_empty())
        && std::path::Path::new(p).exists()
    {
        return Some(p.to_string());
    }
    let dir = FILES_DIR.get()?;
    let p = std::path::Path::new(dir).join("local-llm").join(fallback_name);
    p.exists().then(|| p.to_string_lossy().into_owned())
}

/// Load the model (if not yet loaded) and prefill the constant system+tools
/// prefix so its KV cache is warm before the first real turn. Returns the
/// cached prefix token length. Same blocking-thread pattern as [`generate`].
pub(crate) async fn warmup_prefix(
    config: &PlatformLlmConfig,
    prefix: String,
) -> Result<usize> {
    let config = config.clone();
    tokio::task::spawn_blocking(move || -> Result<usize> {
        ensure_loaded(&config)?;
        let mut guard = LOCAL_LFM
            .get_or_init(|| Mutex::new(None))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let lfm = guard.as_mut().expect("local model just loaded");
        lfm.prefill_prefix(&prefix)
    })
    .await?
}

async fn generate(
    config: &PlatformLlmConfig,
    prefix: String,
    suffix: String,
    max_new_tokens: usize,
) -> Result<String> {
    generate_streaming(config, prefix, suffix, max_new_tokens, |_| {}, || false).await
}

async fn generate_streaming<F, C>(
    config: &PlatformLlmConfig,
    prefix: String,
    suffix: String,
    max_new_tokens: usize,
    mut on_text: F,
    mut should_cancel: C,
) -> Result<String>
where
    F: FnMut(String),
    C: FnMut() -> bool,
{
    let mut gen_cfg = generation_config_from(config);
    gen_cfg.max_new_tokens = max_new_tokens;
    let config = config.clone();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_for_gen = Arc::clone(&cancel);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let join = tokio::task::spawn_blocking(move || -> Result<String> {
        ensure_loaded(&config)?;
        let mut guard = LOCAL_LFM
            .get_or_init(|| Mutex::new(None))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let lfm = guard.as_mut().expect("local model just loaded");
        lfm.generate_prompt_cached_with(
            &prefix,
            &suffix,
            &gen_cfg,
            |text| {
                let _ = tx.send(text.to_string());
            },
            || cancel_for_gen.load(Ordering::Relaxed),
        )
    });
    while let Some(chunk) = rx.recv().await {
        if should_cancel() {
            cancel.store(true, Ordering::Relaxed);
            let _ = join.await;
            anyhow::bail!("Chat cancelled");
        }
        on_text(chunk);
    }
    join.await?
}

pub(super) async fn complete_raw(
    config: &PlatformLlmConfig,
    messages: &[Value],
    tools: &[ToolDescriptor],
) -> Result<LlmTurn> {
    let prefix = build_prefix(config, tools);
    let suffix = format!("{}<|im_start|>assistant\n", render_history(messages));
    let text = generate(config, prefix, suffix, config.local_llm.max_new_tokens).await?;
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    Ok(parse_local_turn_with_names(&text, &tool_names))
}

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
    let prefix = build_prefix(config, tools);
    let suffix = format!("{}<|im_start|>assistant\n", render_history(messages));
    let mut gate = VisibleStreamGate::default();
    let text = generate_streaming(
        config,
        prefix,
        suffix,
        config.local_llm.max_new_tokens,
        |chunk| {
            gate.push(&chunk, |visible| {
                on_event(LlmStreamEvent::ResponseDelta(visible.to_string()));
            });
        },
        &mut should_cancel,
    )
    .await?;
    gate.finish(|visible| {
        on_event(LlmStreamEvent::ResponseDelta(visible.to_string()));
    });
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    Ok(parse_local_turn_with_names(&text, &tool_names))
}

/// Hold back early tokens until we know the reply is prose, not a tool call.
#[derive(Default)]
struct VisibleStreamGate {
    pending: String,
    live: bool,
    hidden: bool,
}

impl VisibleStreamGate {
    fn push(&mut self, chunk: &str, emit: impl FnOnce(&str)) {
        if self.hidden || chunk.is_empty() {
            return;
        }
        if self.live {
            emit(chunk);
            return;
        }
        self.pending.push_str(chunk);
        if looks_like_tool_call(&self.pending) {
            self.hidden = true;
            self.pending.clear();
            return;
        }
        if self.pending.len() >= 20 || self.pending.contains('\n') {
            self.live = true;
            let flush = std::mem::take(&mut self.pending);
            emit(&flush);
        }
    }

    fn finish(&mut self, emit: impl FnOnce(&str)) {
        if self.hidden || self.pending.is_empty() {
            return;
        }
        if looks_like_tool_call(&self.pending) {
            self.hidden = true;
            self.pending.clear();
            return;
        }
        emit(&self.pending);
        self.pending.clear();
        self.live = true;
    }
}

fn looks_like_tool_call(text: &str) -> bool {
    let trimmed = text.trim_start();
    trimmed.starts_with("<|tool_call_start|>")
        || trimmed.starts_with("<|tool_call")
        || trimmed.starts_with("<tool_call>")
        || trimmed.starts_with("<tool_call")
}

#[cfg(test)]
fn build_local_prompt(
    config: &PlatformLlmConfig,
    messages: &[Value],
    tools: &[ToolDescriptor],
) -> String {
    let mut s = String::new();
    s.push_str(&build_prefix(config, tools));
    s.push_str(&render_history(messages));
    s.push_str("<|im_start|>assistant\n");
    s
}

pub(crate) fn build_prefix(_config: &PlatformLlmConfig, tools: &[ToolDescriptor]) -> String {
    let mut s = String::from("<|startoftext|><|im_start|>system\n");
    // Deliberately short fixed persona, NOT the SDK's prepared system_prompt:
    // the 1.2B Instruct model's tool-call adherence collapses when any real
    // system text precedes the tool instructions (it obeys the persona text —
    // "confirm the user's intent first" — instead of emitting the JSON call).
    // The KV prefix cache absorbs what little context is lost.
    s.push_str("You are napaxi, a helpful on-device assistant. 回答简洁，使用中文。");
    // NOTE: the tool catalogue and the JSON format instruction below are
    // appended AFTER the budgeted system text, so a long SDK-compiled system
    // prompt can never truncate them away (that ordering bug left the on-device
    // model without any tool knowledge on real turns).
    if !tools.is_empty() {
        // Minimal catalogue: "name(param, ...)" one line per tool, descriptions
        // omitted. A full JSON descriptor per tool is ~150 tokens; with 30+
        // tools that inflates the prefix to thousands of tokens, and the KV
        // snapshot + prefill activations that follow blow past the device's
        // memory budget (2.4GB RSS at 32 full descriptors vs ~800MB with the
        // condensed list).
        // No tool-count cap: with the condensed one-line format all ~40
        // tools cost ~400 tokens total. A take(32) cap previously dropped
        // shell/read_file/apply_patch/http — registered late in the demo's
        // tool list — so the model literally could not call them.
        s.push_str("\n可用工具（需要时调用）：\n");
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
        // JSON function-call format (the Qwen-series trick): Liquid's docs say
        // models default to Pythonic calls but honor "Output function calls as
        // JSON" in the system prompt — and the smaller Instruct models follow
        // the JSON instruction far more reliably than the Pythonic template.
        // The worked example carries a REAL tool name and parameter (small
        // models copy the example verbatim).
        let first_tool = tools.first();
        let example = match first_tool {
            Some(t) => {
                let first_param = t
                    .parameters
                    .get("required")
                    .and_then(|r| r.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|v| v.as_str())
                    .unwrap_or("arg");
                format!(
                    "{{\"name\": \"{}\", \"arguments\": {{\"{}\": \"value\"}}}}",
                    t.name, first_param
                )
            }
            None => "{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}".to_string(),
        };
        s.push_str(&format!(
            "\n\nOutput function calls as JSON. When the user's request mentions or \
             implies calling a tool (使用/调用/call/use a tool), your ENTIRE reply must be \
             exactly one JSON object on a single line, nothing else — no explanations, no \
             markdown, no questions:\n\
             {example}\n\
             Rules: \"name\" must be a tool from the list above; \"arguments\" is REQUIRED and \
             must carry the parameter values the user asked for (e.g. \"command\" for a \
             command they named). Never emit an empty \"arguments\" when the user supplied \
             values. Only answer in plain text when no tool is relevant at all.\n",
        ));
    }
    s.push_str("<|im_end|>\n");
    s
}

fn render_history(messages: &[Value]) -> String {
    let mut s = String::new();
    let start = messages.len().saturating_sub(4);
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
                if let Some(c) = m.get("content").and_then(Value::as_str) {
                    s.push_str(c);
                }
                if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
                    let rendered: Vec<String> = calls
                        .iter()
                        .filter_map(|c| {
                            let name = c.pointer("/function/name").and_then(Value::as_str)?;
                            let args = c
                                .pointer("/function/arguments")
                                .and_then(Value::as_str)
                                .unwrap_or("{}");
                            Some(format!("{name}({})", json_args_to_python(args)))
                        })
                        .collect();
                    if !rendered.is_empty() {
                        if m.get("content").and_then(Value::as_str).is_some() {
                            s.push('\n');
                        }
                        s.push_str(TOOL_CALL_OPEN);
                        s.push('[');
                        s.push_str(&rendered.join(", "));
                        s.push(']');
                        s.push_str(TOOL_CALL_CLOSE);
                    }
                }
                s.push_str("<|im_end|>\n");
            }
            Some("tool") => {
                let content = m.get("content").and_then(Value::as_str).unwrap_or("");
                s.push_str(&format!("<|im_start|>tool\n{content}<|im_end|>\n"));
            }
            _ => {}
        }
    }
    s
}

#[cfg(test)]
fn parse_local_turn(raw: &str) -> LlmTurn {
    parse_local_turn_with_names(raw, &[])
}

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
        let parsed = parse_tool_call_body(body.trim(), tool_calls.len());
        tool_calls.extend(parsed);
        rest = after;
    }
    content.push_str(rest);
    if tool_calls.is_empty()
        && !tool_names.is_empty()
        && let Some((named, stripped)) = extract_named_tool_calls(&content, tool_names)
    {
        tool_calls = named;
        content = stripped;
    }
    if tool_calls.is_empty()
        && let Some((bare, stripped)) = extract_bare_tool_calls(&content)
    {
        tool_calls = bare;
        content = stripped;
    }
    // LFM2.5 is a reasoning model: strip a leading <think>…</think> block into
    // `reasoning_content` so it never leaks into the visible reply. An
    // unterminated block (length cap hit mid-think) is dropped as well.
    let content = content.trim().to_string();
    let (reasoning, content) = if let Some(rest) = content
        .strip_prefix("<think>")
    {
        match rest.find("</think>") {
            Some(end) => (Some(rest[..end].trim().to_string()), rest[end + 8..].trim().to_string()),
            None => (Some(String::new()), String::new()),
        }
    } else {
        (None, content)
    };
    LlmTurn {
        content,
        reasoning_content: reasoning,
        tool_calls,
        usage: None,
    }
}

fn parse_tool_call_body(body: &str, start_idx: usize) -> Vec<LlmToolCall> {
    let trimmed = body.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        let objects: Vec<&Value> = match &value {
            Value::Array(items) => items.iter().collect(),
            Value::Object(_) => vec![&value],
            _ => Vec::new(),
        };
        let mut calls = Vec::new();
        for (i, obj) in objects.iter().enumerate() {
            if let Some(call) = parse_tool_call_value(obj, start_idx + i) {
                calls.push(call);
            }
        }
        if !calls.is_empty() {
            return calls;
        }
    }
    parse_pythonic_calls(trimmed, start_idx)
}

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

fn parse_pythonic_calls(body: &str, start_idx: usize) -> Vec<LlmToolCall> {
    let mut inner = body.trim();
    if inner.starts_with('[') && inner.ends_with(']') && inner.len() >= 2 {
        inner = &inner[1..inner.len() - 1];
    }
    let chars: Vec<char> = inner.chars().collect();
    let mut i = 0;
    let mut calls = Vec::new();
    while i < chars.len() {
        while i < chars.len() && (chars[i].is_whitespace() || chars[i] == ',') {
            i += 1;
        }
        if i >= chars.len() {
            break;
        }
        if !(chars[i].is_ascii_alphabetic() || chars[i] == '_') {
            break;
        }
        let name_start = i;
        i += 1;
        while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
            i += 1;
        }
        let name: String = chars[name_start..i].iter().collect();
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= chars.len() || chars[i] != '(' {
            break;
        }
        i += 1;
        let args_start = i;
        let mut depth = 1i32;
        let mut in_str: Option<char> = None;
        let mut escaped = false;
        while i < chars.len() && depth > 0 {
            let c = chars[i];
            if let Some(q) = in_str {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == q {
                    in_str = None;
                }
            } else if c == '"' || c == '\'' {
                in_str = Some(c);
            } else if c == '(' || c == '[' || c == '{' {
                depth += 1;
            } else if c == ')' || c == ']' || c == '}' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            i += 1;
        }
        let args_str: String = chars[args_start..i].iter().collect();
        if i < chars.len() && chars[i] == ')' {
            i += 1;
        }
        let arguments = pythonic_args_to_json(&args_str).unwrap_or_else(|| "{}".to_string());
        let idx = start_idx + calls.len();
        calls.push(LlmToolCall {
            id: format!("call_local_{idx}"),
            name,
            arguments,
        });
    }
    calls
}

fn pythonic_args_to_json(args: &str) -> Option<String> {
    let args = args.trim();
    if args.is_empty() {
        return Some("{}".to_string());
    }
    if args.starts_with('{') {
        serde_json::from_str::<Value>(args).ok()?;
        return Some(args.to_string());
    }
    let mut map = serde_json::Map::new();
    for part in split_top_level(args, ',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let eq = part.find('=')?;
        let key = part[..eq].trim();
        if key.is_empty() {
            return None;
        }
        let val = parse_python_value(part[eq + 1..].trim())?;
        map.insert(key.to_string(), val);
    }
    Some(Value::Object(map).to_string())
}

fn parse_python_value(raw: &str) -> Option<Value> {
    let raw = raw.trim();
    if raw.eq_ignore_ascii_case("true") {
        return Some(Value::Bool(true));
    }
    if raw.eq_ignore_ascii_case("false") {
        return Some(Value::Bool(false));
    }
    if raw.eq_ignore_ascii_case("none") || raw.eq_ignore_ascii_case("null") {
        return Some(Value::Null);
    }
    if (raw.starts_with('"') && raw.ends_with('"') && raw.len() >= 2)
        || (raw.starts_with('\'') && raw.ends_with('\'') && raw.len() >= 2)
    {
        return Some(Value::String(raw[1..raw.len() - 1].to_string()));
    }
    if let Ok(n) = serde_json::from_str::<Value>(raw) {
        return Some(n);
    }
    Some(Value::String(raw.to_string()))
}

fn split_top_level(input: &str, sep: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut depth = 0i32;
    let mut in_str: Option<char> = None;
    let mut escaped = false;
    for c in input.chars() {
        if let Some(q) = in_str {
            buf.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                in_str = None;
            }
            continue;
        }
        if c == '"' || c == '\'' {
            in_str = Some(c);
            buf.push(c);
            continue;
        }
        if c == '(' || c == '[' || c == '{' {
            depth += 1;
            buf.push(c);
            continue;
        }
        if c == ')' || c == ']' || c == '}' {
            depth -= 1;
            buf.push(c);
            continue;
        }
        if c == sep && depth == 0 {
            out.push(std::mem::take(&mut buf));
            continue;
        }
        buf.push(c);
    }
    if !buf.trim().is_empty() {
        out.push(buf);
    }
    out
}

fn json_args_to_python(args: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(args) else {
        return String::new();
    };
    match value {
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| format!("{k}={}", python_literal(v)))
            .collect::<Vec<_>>()
            .join(", "),
        _ => String::new(),
    }
}

fn python_literal(v: &Value) -> String {
    match v {
        Value::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        Value::Number(n) => n.to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

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

fn extract_named_tool_calls(text: &str, tool_names: &[&str]) -> Option<(Vec<LlmToolCall>, String)> {
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
            let Some(rel) = text[search..].find(name) else {
                break;
            };
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
    let arguments = if inner.is_empty() {
        "{}".to_string()
    } else if open_ch == '{' {
        serde_json::from_str::<Value>(inner).ok()?;
        inner.to_string()
    } else {
        pythonic_args_to_json(inner)?
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
        max_new_tokens: config.local_llm.max_new_tokens,
        seed: config.local_llm.seed,
        ..GenerationConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_registry::ToolEffect;
    use crate::types::PlatformLlmConfig;

    #[test]
    fn parses_pythonic_tool_call() {
        let turn = parse_local_turn(
            "<|tool_call_start|>[read_file(path=\"/x\")]<|tool_call_end|>",
        );
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "read_file");
        assert_eq!(turn.tool_calls[0].arguments, r#"{"path":"/x"}"#);
        assert!(turn.content.is_empty());
    }

    #[test]
    fn parses_json_tool_call_inside_lfm_tags() {
        let turn = parse_local_turn(
            "<|tool_call_start|>{\"name\":\"read_file\",\"arguments\":{\"path\":\"/x\"}}<|tool_call_end|>",
        );
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "read_file");
        assert_eq!(turn.tool_calls[0].arguments, r#"{"path":"/x"}"#);
    }

    #[test]
    fn parses_multiple_pythonic_calls_and_text() {
        let turn = parse_local_turn(
            "Sure.\n<|tool_call_start|>[a(), b(q=1)]<|tool_call_end|>",
        );
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.tool_calls[0].name, "a");
        assert_eq!(turn.tool_calls[1].name, "b");
        assert_eq!(turn.tool_calls[1].arguments, r#"{"q":1}"#);
        assert_eq!(turn.content, "Sure.");
    }

    #[test]
    fn plain_text_with_no_tool_call_degrades_gracefully() {
        let turn = parse_local_turn("The answer is 42.");
        assert!(turn.tool_calls.is_empty());
        assert_eq!(turn.content, "The answer is 42.");
    }

    #[test]
    fn named_function_call_notation_is_lifted() {
        let turn = parse_local_turn_with_names(
            "set_alarm(time=\"08:00\")",
            &["set_alarm", "create_calendar_event"],
        );
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "set_alarm");
        assert_eq!(turn.tool_calls[0].arguments, r#"{"time":"08:00"}"#);
    }

    #[test]
    fn prompt_uses_short_persona_and_json_instructions() {
        let config = PlatformLlmConfig {
            system_prompt: "X".repeat(3000),
            ..PlatformLlmConfig::default()
        };
        let tools = vec![ToolDescriptor {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            parameters: serde_json::json!({"type":"object","properties":{"path":{"type":"string"}}}),
            effect: ToolEffect::Read,
        }];
        let prompt = build_local_prompt(&config, &[], &tools);
        assert!(prompt.starts_with("<|startoftext|>"));
        // The SDK's long system prompt is deliberately ignored: the 1.2B model
        // obeys persona text instead of the tool instructions when both are
        // present.
        assert!(!prompt.contains('X'));
        assert!(prompt.contains("napaxi"));
        assert!(prompt.contains("可用工具"));
        assert!(prompt.contains("- read_file"));
        // JSON function-call format: the example carries a real tool name and
        // the instruction asks for a bare JSON object (no special-token wrap —
        // the Instruct-tier models follow the JSON instruction far better).
        assert!(prompt.contains("\"name\": \"read_file\""));
        assert!(prompt.contains("Output function calls as JSON"));
        assert!(!prompt.contains("<|tool_call_start|>["));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn visible_gate_streams_prose_and_hides_tool_calls() {
        let mut gate = VisibleStreamGate::default();
        let mut out = String::new();
        gate.push("你好", |s| out.push_str(s));
        assert!(out.is_empty(), "short prefix is buffered");
        gate.push("，今天一起去散步吧。", |s| out.push_str(s));
        assert!(out.contains("你好"));
        assert!(out.contains("散步"));

        let mut tool_gate = VisibleStreamGate::default();
        let mut tool_out = String::new();
        tool_gate.push("<|tool_call_start|>[ls(dir=\"/\")]", |s| tool_out.push_str(s));
        tool_gate.finish(|s| tool_out.push_str(s));
        assert!(tool_out.is_empty());
    }

    #[test]
    fn prompt_renders_history_and_tool_results() {
        let messages = vec![
            serde_json::json!({"role":"user","content":"list"}),
            serde_json::json!({"role":"assistant","tool_calls":[
                {"id":"c1","function":{"name":"ls","arguments":"{\"dir\":\"/\"}"}}
            ]}),
            serde_json::json!({"role":"tool","tool_call_id":"c1","content":"a.txt"}),
        ];
        let prompt = build_local_prompt(&PlatformLlmConfig::default(), &messages, &[]);
        assert!(prompt.contains("<|tool_call_start|>[ls(dir=\"/\")]<|tool_call_end|>"));
        assert!(prompt.contains("<|im_start|>tool\na.txt<|im_end|>"));
    }
}

#[cfg(test)]
mod think_tests {
    use super::*;

    #[test]
    fn strips_think_block_into_reasoning() {
        let turn = parse_local_turn_with_names(
            "<think>pondering...</think>Hello!",
            &[],
        );
        assert_eq!(turn.content, "Hello!");
        assert_eq!(turn.reasoning_content.as_deref(), Some("pondering..."));
    }

    #[test]
    fn unterminated_think_is_dropped() {
        let turn = parse_local_turn_with_names("<think>cut off mid", &[]);
        assert_eq!(turn.content, "");
    }

    #[test]
    fn no_think_block_keeps_content() {
        let turn = parse_local_turn_with_names("plain reply", &[]);
        assert_eq!(turn.content, "plain reply");
        assert_eq!(turn.reasoning_content, None);
    }
}

#[cfg(test)]
mod host_model_tests {
    use super::*;

    #[test]
    #[ignore = "requires the LFM2.5-1.2B GGUF staged under /tmp/napaxi-lfm-test"]
    fn host_long_system_multi_tool() {
        set_files_dir("/tmp/napaxi-lfm-test/files");
        let mut config = PlatformLlmConfig::default();
        config.provider = "local".to_string();
        // Mimic the SDK-compiled system prompt: ~2500 chars of behavior text.
        config.system_prompt = "你是 napaxi，一个有帮助的 AI 助手。回答要专业、简洁、礼貌。在给出任何操作建议前请仔细确认用户意图。如果不确定，请先向用户澄清而不是直接执行。对于危险操作需要格外谨慎。请使用与用户相同的语言回复。".repeat(40);
        let tools: Vec<_> = (0..1)
            .map(|i| crate::tool_registry::ToolDescriptor {
                name: if i == 0 { "shell".to_string() } else { format!("tool_{i}") },
                description: format!("tool {i}"),
                parameters: serde_json::json!({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}),
                effect: Default::default(),
            })
            .collect();
        let messages = vec![
            serde_json::json!({
                "role": "user",
                "content": "你好",
            }),
            serde_json::json!({
                "role": "assistant",
                "content": "你好！我很高兴可以帮助您。",
            }),
            serde_json::json!({
                "role": "user",
                "content": "请使用参数 command=\"uname -r\" 调用工具 shell。",
            }),
        ];
        let turn = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(complete_raw(&config, &messages, &tools))
            .expect("local turn");
        println!("content: {:?}", turn.content);
        println!("tool_calls: {:#?}", turn.tool_calls);
        assert!(
            turn.tool_calls.iter().any(|call| call.name == "shell"),
            "expected shell call",
        );
    }

    #[test]
    #[ignore = "requires the LFM2.5-1.2B GGUF staged under /tmp/napaxi-lfm-test"]
    fn host_tool_call_generation() {
        set_files_dir("/tmp/napaxi-lfm-test/files");
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
        assert!(
            turn.tool_calls.iter().any(|call| call.name == "shell"),
            "expected shell call, got content={:?} calls={:?}",
            turn.content,
            turn.tool_calls,
        );
    }

    #[test]
    #[ignore = "requires the LFM2.5-1.2B GGUF staged under /tmp/napaxi-lfm-test"]
    fn host_warmup_then_two_turns_reuse_prefix() {
        set_files_dir("/tmp/napaxi-lfm-test/files");
        let mut config = PlatformLlmConfig::default();
        config.provider = "local".to_string();
        let tools = vec![crate::tool_registry::ToolDescriptor {
            name: "shell".to_string(),
            description: "run a shell command".to_string(),
            parameters: serde_json::json!({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}),
            effect: Default::default(),
        }];
        let prefix = build_prefix(&config, &tools);
        let warmed = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(warmup_prefix(&config, prefix.clone()))
            .expect("warmup");
        assert!(warmed > 0, "prefix token count: {warmed}");

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
}
