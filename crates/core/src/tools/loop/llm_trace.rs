//! Optional LLM request tracing for benchmarks and debugging.
//!
//! When the `NAPAXI_LLM_TRACE` environment variable is set to any non-empty
//! value, every tool-loop LLM call appends one JSON line describing the exact
//! request sent to the provider: the effective system prompt, the full message
//! list and the tool descriptors visible to the model for that call. The dump
//! lives next to the session storage (`<files_dir>/llm-trace/`), keyed by
//! thread id, so the host platform (e.g. the benchmark runner) can read it
//! back after the turn. Off by default; nothing is written unless the
//! variable is set.

use std::io::Write;
use std::path::PathBuf;

use crate::types::PlatformLlmConfig;
use crate::tool_registry::{ToolDescriptor, ToolExecutionContext};

pub(crate) fn dump_llm_request(
    context: Option<&ToolExecutionContext>,
    config: &PlatformLlmConfig,
    messages: &[serde_json::Value],
    active_descriptors: &[ToolDescriptor],
) {
    let enabled = std::env::var("NAPAXI_LLM_TRACE")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);
    if !enabled {
        return;
    }
    let Some(context) = context else {
        return;
    };
    let thread_id = context
        .session_key_json
        .as_deref()
        .and_then(extract_thread_id)
        .unwrap_or_else(|| "unknown-thread".to_string());
    let dir = PathBuf::from(&context.files_dir).join("llm-trace");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join(format!("llm-trace-{thread_id}.jsonl"));
    let entry = serde_json::json!({
        "ts": chrono::Utc::now().to_rfc3339(),
        "model": config.model,
        "system_prompt": config.system_prompt,
        "messages": messages,
        "tools": active_descriptors.iter().map(|descriptor| serde_json::json!({
            "name": descriptor.name,
            "description": descriptor.description,
            "parameters": descriptor.parameters,
        })).collect::<Vec<_>>(),
    });
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    else {
        return;
    };
    let _ = writeln!(file, "{entry}");
}

pub(crate) fn llm_trace_dir(files_dir: &str) -> Option<PathBuf> {
    let enabled = std::env::var("NAPAXI_LLM_TRACE")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);
    if !enabled {
        return None;
    }
    Some(PathBuf::from(files_dir).join("llm-trace"))
}

fn extract_thread_id(session_key_json: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(session_key_json).ok()?;
    value
        .get("thread_id")?
        .as_str()
        .map(|thread_id| thread_id.to_string())
}
