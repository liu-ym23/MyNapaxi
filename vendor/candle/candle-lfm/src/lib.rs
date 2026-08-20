//! CPU-only LFM2 / LFM2.5 GGUF inference library built on Candle.

mod token_stream;

use std::fs::File;
use std::path::Path;

use anyhow::{Error as E, Result};
use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_lfm2::{LayerCache, ModelWeights};
use token_stream::TokenOutputStream;
use tokenizers::Tokenizer;

/// LFM2.5 model size used when downloading from Hugging Face.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelSize {
    B1_2,
    #[default]
    B2_6,
}

impl ModelSize {
    #[cfg(feature = "hf-hub")]
    fn tokenizer_repo(self) -> &'static str {
        match self {
            Self::B1_2 => "LiquidAI/LFM2.5-1.2B-Instruct",
            Self::B2_6 => "LiquidAI/LFM2.5-2.6B",
        }
    }

    #[cfg(feature = "hf-hub")]
    fn model(self) -> (&'static str, &'static str) {
        match self {
            Self::B1_2 => (
                "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
                "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
            ),
            Self::B2_6 => ("LiquidAI/LFM2.5-2.6B-GGUF", "LFM2.5-2.6B-Q4_K_M.gguf"),
        }
    }
}

/// Text-generation sampling settings.
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub max_new_tokens: usize,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: None,
            top_k: None,
            max_new_tokens: 256,
            seed: 42,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

/// A loaded LFM2 / LFM2.5 model and tokenizer.
pub struct Lfm {
    model: ModelWeights,
    tokens: TokenOutputStream,
    device: Device,
    /// Cached state for a constant prompt prefix: `(prefix_token_ids, per-layer
    /// attention KV + short-conv snapshot)`. Built once on first use and reused
    /// across calls that share the same prefix.
    prefix_cache: Option<(Vec<u32>, Vec<LayerCache>)>,
}

impl Lfm {
    /// Loads a local GGUF model and matching `tokenizer.json`.
    pub fn from_files(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let device = Device::Cpu;
        let model = load_model(model_path.as_ref(), &device)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref()).map_err(E::msg)?;
        Ok(Self {
            model,
            tokens: TokenOutputStream::new(tokenizer),
            device,
            prefix_cache: None,
        })
    }

    /// Downloads (or reuses cached) model files from Hugging Face, then loads them.
    #[cfg(feature = "hf-hub")]
    pub fn from_hugging_face(size: ModelSize, cache_dir: impl AsRef<Path>) -> Result<Self> {
        let cache_dir = cache_dir.as_ref();
        std::fs::create_dir_all(cache_dir)?;
        let api = hf_api(cache_dir)?;
        let tokenizer_path = api
            .model(size.tokenizer_repo().to_owned())
            .get("tokenizer.json")?;
        let (repo, filename) = size.model();
        let model_path = api
            .repo(hf_hub::Repo::with_revision(
                repo.to_owned(),
                hf_hub::RepoType::Model,
                "main".to_owned(),
            ))
            .get(filename)?;
        Self::from_files(model_path, tokenizer_path)
    }

    /// Generates one response. Each call starts with a fresh cache.
    /// The prompt is wrapped in a single-turn LFM ChatML user turn.
    pub fn generate(&mut self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        self.generate_prompt(&format_prompt(prompt), config)
    }

    /// Generates one response from a caller-supplied, pre-formatted prompt.
    pub fn generate_prompt(&mut self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        self.model.clear_kv_cache();
        self.tokens.clear();
        if config.max_new_tokens == 0 {
            return Ok(String::new());
        }

        let prompt_tokens = self
            .tokens
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let eos = eos_token_id(self.tokens.tokenizer())?;

        let mut sampler = make_sampler(config);
        let mut generated_tokens = Vec::new();
        let mut output = String::new();
        let mut next_input = None;

        for index in 0..config.max_new_tokens {
            let logits = if let Some(token) = next_input {
                let input = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
                self.model
                    .forward(&input, prompt_tokens.len() + index - 1)?
                    .squeeze(0)?
            } else {
                let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                self.model.forward(&input, 0)?.squeeze(0)?
            };
            let logits = if config.repeat_penalty == 1.0 {
                logits
            } else {
                let start = generated_tokens.len().saturating_sub(config.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    config.repeat_penalty,
                    &generated_tokens[start..],
                )?
            };

            let token = sampler.sample(&logits)?;
            if token == eos {
                break;
            }
            generated_tokens.push(token);
            next_input = Some(token);
            if let Some(text) = self.tokens.next_token(token)? {
                output.push_str(&text);
            }
        }

        if let Some(text) = self.tokens.decode_rest()? {
            output.push_str(&text);
        }
        Ok(output)
    }

    /// Forward `prefix` through the model and record it as the cached prefix
    /// (no generation). Warm-up entry point: lets a caller pay the prefix
    /// prefill cost ahead of the first real turn. Returns the cached prefix
    /// token length; a no-op returning the existing length on a cache hit.
    pub fn prefill_prefix(&mut self, prefix: &str) -> Result<usize> {
        Ok(self.ensure_prefix_cache(prefix)?.len())
    }

    fn ensure_prefix_cache(&mut self, prefix: &str) -> Result<Vec<u32>> {
        let prefix_tokens = self
            .tokens
            .tokenizer()
            .encode(prefix, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let needs_rebuild = match &self.prefix_cache {
            None => true,
            Some((cached_tokens, _)) => cached_tokens != &prefix_tokens,
        };
        if needs_rebuild {
            self.model.clear_kv_cache();
            if !prefix_tokens.is_empty() {
                let input =
                    Tensor::new(prefix_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                self.model.forward(&input, 0)?.squeeze(0)?;
            }
            let snapshot = self.model.snapshot_kv_cache();
            self.prefix_cache = Some((prefix_tokens.clone(), snapshot));
        }
        Ok(prefix_tokens)
    }

    /// Generate from a prompt split into a constant `prefix` and a per-call
    /// `suffix`. Reuses the prefix cache across calls.
    pub fn generate_prompt_cached(
        &mut self,
        prefix: &str,
        suffix: &str,
        config: &GenerationConfig,
    ) -> Result<String> {
        self.generate_prompt_cached_with(prefix, suffix, config, |_| {}, || false)
    }

    /// Same as [`Self::generate_prompt_cached`], with a text callback for
    /// streaming and a stop flag checked after each generated token.
    pub fn generate_prompt_cached_with<F, S>(
        &mut self,
        prefix: &str,
        suffix: &str,
        config: &GenerationConfig,
        mut on_text: F,
        mut should_stop: S,
    ) -> Result<String>
    where
        F: FnMut(&str),
        S: FnMut() -> bool,
    {
        self.tokens.clear();
        let prefix_tokens = self.ensure_prefix_cache(prefix)?;
        if let Some((_, snapshot)) = &self.prefix_cache {
            self.model.restore_kv_cache(snapshot);
        }
        if config.max_new_tokens == 0 {
            return Ok(String::new());
        }

        let suffix_tokens = self
            .tokens
            .tokenizer()
            .encode(suffix, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let start_pos = prefix_tokens.len() + suffix_tokens.len();
        let eos = eos_token_id(self.tokens.tokenizer())?;

        let mut sampler = make_sampler(config);
        let mut generated_tokens = Vec::new();
        let mut output = String::new();
        let mut next_input = None;
        for index in 0..config.max_new_tokens {
            if should_stop() {
                break;
            }
            let logits = if let Some(token) = next_input {
                let input = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
                self.model
                    .forward(&input, start_pos + index - 1)?
                    .squeeze(0)?
            } else {
                let input =
                    Tensor::new(suffix_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                self.model
                    .forward(&input, prefix_tokens.len())?
                    .squeeze(0)?
            };
            let logits = if config.repeat_penalty == 1.0 {
                logits
            } else {
                let start = generated_tokens.len().saturating_sub(config.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    config.repeat_penalty,
                    &generated_tokens[start..],
                )?
            };
            let token = sampler.sample(&logits)?;
            if token == eos {
                break;
            }
            generated_tokens.push(token);
            next_input = Some(token);
            if let Some(text) = self.tokens.next_token(token)? {
                output.push_str(&text);
                on_text(&text);
            }
        }
        if let Some(text) = self.tokens.decode_rest()? {
            output.push_str(&text);
            on_text(&text);
        }
        Ok(output)
    }
}

#[cfg(feature = "hf-hub")]
fn hf_api(cache_dir: &Path) -> Result<hf_hub::api::sync::Api> {
    let hub = cache_dir.join("hub");
    Ok(hf_hub::api::sync::ApiBuilder::from_cache(hf_hub::Cache::new(hub)).build()?)
}

fn load_model(path: &Path, device: &Device) -> Result<ModelWeights> {
    let mut file = File::open(path)?;
    let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path))?;
    Ok(ModelWeights::from_gguf(content, &mut file, device)?)
}

fn format_prompt(prompt: &str) -> String {
    format!("<|startoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

fn eos_token_id(tokenizer: &Tokenizer) -> Result<u32> {
    let vocab = tokenizer.get_vocab(true);
    for name in ["<|im_end|>", "<|endoftext|>", "</s>"] {
        if let Some(id) = vocab.get(name) {
            return Ok(*id);
        }
    }
    anyhow::bail!("missing LFM EOS token (<|im_end|>)")
}

fn make_sampler(config: &GenerationConfig) -> LogitsProcessor {
    let sampling = if config.temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        match (config.top_k, config.top_p) {
            (None, None) => Sampling::All {
                temperature: config.temperature,
            },
            (Some(k), None) => Sampling::TopK {
                k,
                temperature: config.temperature,
            },
            (None, Some(p)) => Sampling::TopP {
                p,
                temperature: config.temperature,
            },
            (Some(k), Some(p)) => Sampling::TopKThenTopP {
                k,
                p,
                temperature: config.temperature,
            },
        }
    };
    LogitsProcessor::from_sampling(config.seed, sampling)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_defaults_are_stable() {
        let config = GenerationConfig::default();
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.max_new_tokens, 256);
        assert_eq!(config.seed, 42);
        assert_eq!(config.repeat_penalty, 1.1);
        assert_eq!(config.repeat_last_n, 64);
    }

    #[test]
    fn prompt_uses_lfm_chat_template() {
        assert_eq!(
            format_prompt("你好"),
            "<|startoftext|><|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
