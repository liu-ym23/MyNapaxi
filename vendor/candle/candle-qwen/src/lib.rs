//! CPU-only Qwen2 GGUF inference library built on Candle.

mod token_stream;

use std::fs::File;
use std::path::Path;

use anyhow::{Error as E, Result};
use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use token_stream::TokenOutputStream;
use tokenizers::Tokenizer;

/// Qwen2 model size used when downloading from Hugging Face.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelSize {
    #[default]
    B0_5,
    B1_5,
    B7,
    B72,
}

impl ModelSize {
    #[cfg(feature = "download")]
    fn tokenizer_repo(self) -> &'static str {
        match self {
            Self::B0_5 => "Qwen/Qwen2-0.5B-Instruct",
            Self::B1_5 => "Qwen/Qwen2-1.5B-Instruct",
            Self::B7 => "Qwen/Qwen2-7B-Instruct",
            Self::B72 => "Qwen/Qwen2-72B-Instruct",
        }
    }

    #[cfg(feature = "download")]
    fn model(self) -> (&'static str, &'static str) {
        match self {
            Self::B0_5 => (
                "Qwen/Qwen2-0.5B-Instruct-GGUF",
                "qwen2-0_5b-instruct-q4_0.gguf",
            ),
            Self::B1_5 => (
                "Qwen/Qwen2-1.5B-Instruct-GGUF",
                "qwen2-1_5b-instruct-q4_0.gguf",
            ),
            Self::B7 => ("Qwen/Qwen2-7B-Instruct-GGUF", "qwen2-7b-instruct-q4_0.gguf"),
            Self::B72 => (
                "Qwen/Qwen2-72B-Instruct-GGUF",
                "qwen2-72b-instruct-q4_0.gguf",
            ),
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

/// A loaded Qwen2 model and tokenizer.
pub struct Qwen {
    model: ModelWeights,
    tokens: TokenOutputStream,
    device: Device,
}

impl Qwen {
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
        })
    }

    /// Downloads (or reuses cached) model files from Hugging Face, then loads them.
    #[cfg(feature = "download")]
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

    /// Generates one response from a plain user prompt, wrapping it in the
    /// Qwen ChatML template first. Each call starts with a fresh KV cache.
    pub fn generate(&mut self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        self.generate_raw(&format_prompt(prompt), config)
    }

    /// Generates one response from an already-formatted prompt (e.g. a full
    /// ChatML transcript with system / history turns already rendered). No chat
    /// template is applied, unlike [`generate`](Self::generate); callers are
    /// responsible for the full prompt including the trailing
    /// `<|im_start|>assistant\n` primer. Each call starts with a fresh KV cache.
    pub fn generate_raw(&mut self, prompt: &str, config: &GenerationConfig) -> Result<String> {
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
        let eos = *self
            .tokens
            .tokenizer()
            .get_vocab(true)
            .get("<|im_end|>")
            .ok_or_else(|| anyhow::anyhow!("missing <|im_end|> token"))?;

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
}

#[cfg(feature = "download")]
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
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
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
    fn generation_defaults_match_the_previous_cli() {
        let config = GenerationConfig::default();
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.max_new_tokens, 256);
        assert_eq!(config.seed, 42);
        assert_eq!(config.repeat_penalty, 1.1);
        assert_eq!(config.repeat_last_n, 64);
    }

    #[test]
    fn prompt_uses_qwen_chat_template() {
        assert_eq!(
            format_prompt("你好"),
            "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    #[cfg(feature = "download")]
    #[ignore = "requires a cached or downloadable Qwen model"]
    fn cached_model_smoke_test() {
        let cache_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("cache/huggingface");
        let mut qwen = Qwen::from_hugging_face(ModelSize::B0_5, cache_dir).unwrap();
        let config = GenerationConfig {
            temperature: 0.0,
            max_new_tokens: 16,
            ..Default::default()
        };
        let output = qwen.generate("你好", &config).unwrap();
        assert!(!output.trim().is_empty());
    }
}
