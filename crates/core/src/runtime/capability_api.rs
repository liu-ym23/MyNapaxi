//! Capability status query and config update over a runtime engine handle.

use super::handle::{handle_to_arc, invalid_handle_json, parse_config};
use crate::error::{CoreError, CoreResult};

pub fn capability_status_json_handle(
    handle: i64,
    profile_json: &str,
    selection_json: &str,
) -> String {
    // SAFETY: `handle` is a live engine handle produced by `create_engine_handle`; `handle_to_arc` returns `None` for a `0`/invalid handle rather than dereferencing it.
    let Some(engine) = (unsafe { handle_to_arc(handle) }) else {
        return crate::capabilities::status_json("unknown", profile_json, selection_json);
    };
    let profile = if is_blank_capability_json(profile_json) {
        engine.capability_profile()
    } else {
        crate::capabilities::profile_from_json(profile_json)
    };
    let selection = if is_blank_capability_json(selection_json) {
        engine.capability_selection()
    } else {
        crate::capabilities::selection_from_json(selection_json)
    };
    let platform = profile
        .platform
        .as_deref()
        .unwrap_or_else(|| engine.platform());
    serde_json::to_string(&crate::capabilities::status(
        platform,
        &serde_json::to_string(&profile).unwrap_or_else(|_| "{}".to_string()),
        &serde_json::to_string(&selection).unwrap_or_else(|_| "{}".to_string()),
    ))
    .unwrap_or_else(|_| "[]".to_string())
}

fn is_blank_capability_json(raw: &str) -> bool {
    let trimmed = raw.trim();
    trimmed.is_empty() || trimmed == "{}" || trimmed == "null"
}

pub fn update_config_handle(handle: i64, config_json: &str) -> bool {
    match update_config_handle_typed(handle, config_json) {
        Ok(()) => true,
        Err(error) => {
            tracing::warn!(
                error = %error,
                code = error.code(),
                handle,
                "update_config_handle failed"
            );
            false
        }
    }
}

/// Result-returning variant. Surfaces `invalid_handle` vs `config` errors
/// instead of collapsing both into `false`.
pub fn update_config_handle_typed(handle: i64, config_json: &str) -> CoreResult<()> {
    // SAFETY: `handle` is a live engine handle produced by `create_engine_handle`; `handle_to_arc` returns `None` for a `0`/invalid handle rather than dereferencing it.
    let engine = unsafe { handle_to_arc(handle) }.ok_or(CoreError::InvalidHandle(handle))?;
    let config = parse_config(config_json)?;
    if engine.update_config(config) {
        Ok(())
    } else {
        Err(CoreError::LockPoisoned("engine.config"))
    }
}

/// Warm up the on-device local LLM: load the model (if needed) and prefill
/// the constant system+tools prefix so the KV cache is hot before the first
/// turn. Enumerates the engine's tool registry and applies the same capability
/// filter the tool loop uses, so the warmed prefix matches the first turn's
/// prefix. Returns `{"ok":true,"prefix_tokens":N}` or
/// `{"ok":false,"error":"..."}`.
pub async fn warmup_local_llm_handle(handle: i64) -> String {
    // SAFETY: `handle` is a live engine handle produced by `create_engine_handle`; `handle_to_arc` returns `None` for a `0`/invalid handle rather than dereferencing it.
    let Some(engine) = (unsafe { handle_to_arc(handle) }) else {
        return invalid_handle_json().to_string();
    };
    let config = engine.config();
    if config.provider != "local" {
        return r#"{"ok":false,"error":"provider is not local"}"#.to_string();
    }
    let tools = crate::tool_loop::gather_tool_descriptors_for_config(
        &config,
        Some(&engine.tools()),
        Vec::new(),
    )
    .await;
    let prefix = crate::llm::local_lfm::build_prefix(&config, &tools);
    match crate::llm::local_lfm::warmup_prefix(&config, prefix).await {
        Ok(tokens) => format!(r#"{{"ok":true,"prefix_tokens":{tokens}}}"#),
        Err(error) => format!(r#"{{"ok":false,"error":{:?}}}"#, format!("{error:#}")),
    }
}

pub fn get_config_handle(handle: i64) -> String {
    // SAFETY: `handle` is a live engine handle produced by `create_engine_handle`; `handle_to_arc` returns `None` for a `0`/invalid handle rather than dereferencing it.
    let Some(engine) = (unsafe { handle_to_arc(handle) }) else {
        return invalid_handle_json();
    };
    engine.config_json()
}

/// Result-returning variant. Returns the engine's config JSON or a structured
/// `InvalidHandle` error.
pub fn get_config_handle_typed(handle: i64) -> CoreResult<String> {
    // SAFETY: `handle` is a live engine handle produced by `create_engine_handle`; `handle_to_arc` returns `None` for a `0`/invalid handle rather than dereferencing it.
    let engine = unsafe { handle_to_arc(handle) }.ok_or(CoreError::InvalidHandle(handle))?;
    Ok(engine.config_json())
}
