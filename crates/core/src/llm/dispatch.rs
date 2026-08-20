//! Provider routing: dispatches `complete` / `stream` calls to the correct
//! provider implementation based on the resolved `LlmProviderRoute`.

use std::future::Future;

use anyhow::Result;
use serde_json::Value;

use super::provider::resolve_provider_route;
use super::sse::{STREAM_RETRY_ATTEMPTS, is_retryable_transport_error, stream_retry_delay_for};
use super::{LlmStreamEvent, LlmTurn, anthropic, gemini, local_lfm, openai_compatible};
use crate::capabilities::LlmProviderRoute;
use crate::session::SessionMessage;
use crate::tool_registry::ToolDescriptor;
use crate::types::PlatformLlmConfig;

/// Test-only no-op LLM provider. Selecting `provider = "__test_noop__"` makes
/// every dispatch entry point return an empty, successful response without any
/// network call, so tests that drive a full agent turn (automation jobs, turn
/// orchestration, group delegation) run deterministically offline instead of
/// failing on a real HTTP request. Compiled out of release builds entirely —
/// in a shipped binary `__test_noop__` is just an unknown provider and errors
/// through the normal route resolution.
#[cfg(test)]
pub(crate) const TEST_NOOP_PROVIDER: &str = "__test_noop__";

#[cfg(test)]
fn is_test_noop(config: &PlatformLlmConfig) -> bool {
    config.provider == TEST_NOOP_PROVIDER
}

#[cfg(test)]
fn test_noop_turn() -> LlmTurn {
    LlmTurn {
        content: String::new(),
        reasoning_content: None,
        tool_calls: Vec::new(),
        usage: None,
    }
}

#[cfg(test)]
tokio::task_local! {
    /// Per-turn queue of pre-scripted LLM turns. When set (via
    /// [`with_scripted_turns`]), the `__test_noop__` provider pops the front of
    /// this queue instead of returning an empty turn, letting a test drive a
    /// full multi-iteration tool loop offline (turn 1 emits a tool call, the
    /// tool runs, turn 2 returns final text). Absent outside a scoped turn, so
    /// existing empty-turn tests fall back to [`test_noop_turn`].
    static SCRIPTED_TURNS: std::cell::RefCell<std::collections::VecDeque<LlmTurn>>;
}

/// Run `fut` with a queue of scripted LLM turns in scope. While active, each
/// `__test_noop__` dispatch call consumes the next scripted turn in order.
/// The queue is task-scoped (not thread-local) so it survives the turn loop's
/// `.await`s across tokio worker threads, and never leaks between tests.
#[cfg(test)]
pub(crate) async fn with_scripted_turns<F, T>(turns: Vec<LlmTurn>, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    SCRIPTED_TURNS
        .scope(std::cell::RefCell::new(turns.into_iter().collect()), fut)
        .await
}

/// Pop the next scripted turn if a queue is in scope. Returns `None` when no
/// queue is active (the common case outside the harness).
#[cfg(test)]
fn next_scripted_turn() -> Option<LlmTurn> {
    SCRIPTED_TURNS
        .try_with(|queue| queue.borrow_mut().pop_front())
        .ok()
        .flatten()
}

/// Retry a non-streaming LLM call on transient transport failures (connection
/// drops, timeouts, 429/5xx). Non-streaming calls have no partial output to
/// discard, so any retryable error simply replays the request after a backoff.
async fn with_retry<T, Fut, F>(mut attempt_fn: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut last_error: Option<anyhow::Error> = None;
    for attempt in 0..STREAM_RETRY_ATTEMPTS {
        match attempt_fn().await {
            Ok(value) => return Ok(value),
            Err(error)
                if attempt + 1 < STREAM_RETRY_ATTEMPTS && is_retryable_transport_error(&error) =>
            {
                let delay = stream_retry_delay_for(attempt, Some(&error));
                last_error = Some(error);
                tokio::time::sleep(delay).await;
            }
            Err(error) => return Err(error),
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("LLM request failed")))
}

pub async fn complete_with_history(
    config: &PlatformLlmConfig,
    messages: &[SessionMessage],
) -> Result<String> {
    #[cfg(test)]
    if is_test_noop(config) {
        return Ok(String::new());
    }
    let route = resolve_provider_route(config)?;
    // Local on-device model: no api_key / cloud model id, no transport, no retry.
    if route == LlmProviderRoute::Local {
        let messages = super::openai_messages_from_mobile_history(messages);
        let turn = local_lfm::complete_raw(config, &messages, &[]).await?;
        return Ok(turn.content);
    }
    if config.api_key.trim().is_empty() {
        anyhow::bail!("LLM API key is required");
    }
    if config.model.trim().is_empty() {
        anyhow::bail!("LLM model is required");
    }

    with_retry(|| async move {
        match route {
            LlmProviderRoute::Anthropic => anthropic::complete(config, messages).await,
            LlmProviderRoute::Gemini => gemini::complete(config, messages).await,
            LlmProviderRoute::OpenAiCompatible => {
                openai_compatible::complete(config, messages).await
            }
            LlmProviderRoute::Local => unreachable!("local route handled before retry path"),
        }
    })
    .await
}

pub async fn complete_turn_with_raw_messages(
    config: &PlatformLlmConfig,
    messages: &[Value],
    tools: &[ToolDescriptor],
) -> Result<LlmTurn> {
    #[cfg(test)]
    if is_test_noop(config) {
        return Ok(next_scripted_turn().unwrap_or_else(test_noop_turn));
    }
    let route = resolve_provider_route(config)?;
    // Local on-device model: no api_key / cloud model id, no transport, no retry.
    // Resolving the route first keeps local behind the same capability admission
    // gate as every cloud provider.
    if route == LlmProviderRoute::Local {
        return local_lfm::complete_raw(config, messages, tools).await;
    }
    if config.api_key.trim().is_empty() {
        anyhow::bail!("LLM API key is required");
    }
    if config.model.trim().is_empty() {
        anyhow::bail!("LLM model is required");
    }

    with_retry(|| async move {
        match route {
            LlmProviderRoute::OpenAiCompatible => {
                openai_compatible::complete_raw(config, messages, tools).await
            }
            LlmProviderRoute::Anthropic => anthropic::complete_raw(config, messages, tools).await,
            LlmProviderRoute::Gemini => gemini::complete_raw(config, messages, tools).await,
            // Pre-filtered above; local has no transport to retry.
            LlmProviderRoute::Local => unreachable!("local route handled before retry path"),
        }
    })
    .await
}

pub async fn stream_turn_with_raw_messages<F, C>(
    config: &PlatformLlmConfig,
    messages: &[Value],
    tools: &[ToolDescriptor],
    on_event: F,
    mut should_cancel: C,
) -> Result<LlmTurn>
where
    F: FnMut(LlmStreamEvent),
    C: FnMut() -> bool,
{
    #[cfg(test)]
    if is_test_noop(config) {
        return Ok(next_scripted_turn().unwrap_or_else(test_noop_turn));
    }
    let route = resolve_provider_route(config)?;
    // Local on-device model: no api_key / cloud model id, no transport.
    if route == LlmProviderRoute::Local {
        return local_lfm::stream_raw_cancelable(config, messages, tools, on_event, should_cancel)
            .await;
    }
    if config.api_key.trim().is_empty() {
        anyhow::bail!("LLM API key is required");
    }
    if config.model.trim().is_empty() {
        anyhow::bail!("LLM model is required");
    }
    if should_cancel() {
        anyhow::bail!("Chat cancelled");
    }

    match route {
        LlmProviderRoute::OpenAiCompatible => {
            openai_compatible::stream_raw_cancelable(
                config,
                messages,
                tools,
                on_event,
                should_cancel,
            )
            .await
        }
        LlmProviderRoute::Anthropic => {
            anthropic::stream_raw_cancelable(config, messages, tools, on_event, should_cancel).await
        }
        LlmProviderRoute::Gemini => {
            gemini::stream_raw_cancelable(config, messages, tools, on_event, should_cancel).await
        }
        LlmProviderRoute::Local => unreachable!("local route handled before cloud streaming path"),
    }
}

pub async fn stream_with_history_cancelable<F, C>(
    config: &PlatformLlmConfig,
    messages: &[SessionMessage],
    on_event: F,
    mut should_cancel: C,
) -> Result<String>
where
    F: FnMut(LlmStreamEvent),
    C: FnMut() -> bool,
{
    #[cfg(test)]
    if is_test_noop(config) {
        return Ok(String::new());
    }
    let route = resolve_provider_route(config)?;
    // Local on-device model: no api_key / cloud model id, no transport.
    if route == LlmProviderRoute::Local {
        let messages = super::openai_messages_from_mobile_history(messages);
        let turn =
            local_lfm::stream_raw_cancelable(config, &messages, &[], on_event, should_cancel)
                .await?;
        return Ok(turn.content);
    }
    if config.api_key.trim().is_empty() {
        anyhow::bail!("LLM API key is required");
    }
    if config.model.trim().is_empty() {
        anyhow::bail!("LLM model is required");
    }
    if should_cancel() {
        anyhow::bail!("Chat cancelled");
    }

    match route {
        LlmProviderRoute::OpenAiCompatible => {
            openai_compatible::stream_cancelable(config, messages, on_event, should_cancel).await
        }
        LlmProviderRoute::Anthropic => {
            anthropic::stream_cancelable(config, messages, on_event, should_cancel).await
        }
        LlmProviderRoute::Gemini => {
            gemini::stream_cancelable(config, messages, on_event, should_cancel).await
        }
        LlmProviderRoute::Local => unreachable!("local route handled before cloud streaming path"),
    }
}
