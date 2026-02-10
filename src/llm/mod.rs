use anyhow::{anyhow, Result};
use encoding_rs::UTF_8;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

pub trait Llm {
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String>;
}

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub model_path: PathBuf,
    pub n_ctx: Option<u32>,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_gpu_layers: Option<u32>,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub seed: u32,
}

impl LlmConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            n_ctx: Some(8192),
            n_batch: 512,
            n_ubatch: 512,
            n_gpu_layers: None,
            temperature: 0.2,
            top_k: 40,
            top_p: 0.9,
            seed: 0,
        }
    }

    pub fn from_env(model_path: impl Into<PathBuf>) -> Self {
        let mut cfg = Self::new(model_path);

        if let Ok(v) = std::env::var("VARCTX_N_CTX") {
            cfg.n_ctx = v.parse::<u32>().ok();
        }
        if let Ok(v) = std::env::var("VARCTX_N_BATCH") {
            if let Ok(n) = v.parse::<u32>() {
                cfg.n_batch = n.max(1);
            }
        }
        if let Ok(v) = std::env::var("VARCTX_N_UBATCH") {
            if let Ok(n) = v.parse::<u32>() {
                cfg.n_ubatch = n.max(1);
            }
        }
        if let Ok(v) = std::env::var("VARCTX_N_GPU_LAYERS") {
            cfg.n_gpu_layers = v.parse::<u32>().ok();
        }
        if let Ok(v) = std::env::var("VARCTX_TEMP") {
            if let Ok(t) = v.parse::<f32>() {
                cfg.temperature = t;
            }
        }
        if let Ok(v) = std::env::var("VARCTX_TOP_K") {
            if let Ok(k) = v.parse::<i32>() {
                cfg.top_k = k;
            }
        }
        if let Ok(v) = std::env::var("VARCTX_TOP_P") {
            if let Ok(p) = v.parse::<f32>() {
                cfg.top_p = p;
            }
        }
        if let Ok(v) = std::env::var("VARCTX_SEED") {
            if let Ok(s) = v.parse::<u32>() {
                cfg.seed = s;
            }
        }

        cfg
    }
}

pub struct LlamaCppLlm {
    _backend: LlamaBackend,
    model: &'static LlamaModel,
    ctx: llama_cpp_2::context::LlamaContext<'static>,
    config: LlmConfig,
}

impl LlamaCppLlm {
    pub fn load(config: LlmConfig) -> Result<Self> {
        if !Path::new(&config.model_path).exists() {
            return Err(anyhow!(
                "model not found at {}",
                config.model_path.display()
            ));
        }

        let backend = LlamaBackend::init()?;

        let mut model_params = LlamaModelParams::default();
        if let Some(n_gpu_layers) = config.n_gpu_layers {
            model_params = model_params.with_n_gpu_layers(n_gpu_layers);
        }

        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)?;
        let model = Box::new(model);
        let model: &'static LlamaModel = Box::leak(model);

        let mut ctx_params = LlamaContextParams::default()
            .with_n_batch(config.n_batch)
            .with_n_ubatch(config.n_ubatch);
        if let Some(n_ctx) = config.n_ctx {
            ctx_params = ctx_params.with_n_ctx(NonZeroU32::new(n_ctx));
        }

        let ctx = model.new_context(&backend, ctx_params)?;

        Ok(Self {
            _backend: backend,
            model,
            ctx,
            config,
        })
    }

    fn decode_prompt(&mut self, tokens: &[llama_cpp_2::token::LlamaToken]) -> Result<i32> {
        let mut n_past: i32 = 0;
        let batch_cap = self.config.n_batch.max(1) as usize;

        let mut i = 0;
        while i < tokens.len() {
            let end = (i + batch_cap).min(tokens.len());
            let mut batch = LlamaBatch::new(end - i, 1);

            for (j, token) in tokens[i..end].iter().enumerate() {
                let pos = n_past + j as i32;
                let logits = j == (end - i - 1);
                batch.add(*token, pos, &[0], logits)?;
            }

            self.ctx.decode(&mut batch)?;
            n_past += (end - i) as i32;
            i = end;
        }

        Ok(n_past)
    }
}

impl Llm for LlamaCppLlm {
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.ctx.clear_kv_cache();

        let tokens = self.model.str_to_token(prompt, AddBos::Always)?;
        if tokens.is_empty() {
            return Ok(String::new());
        }

        let mut n_past = self.decode_prompt(&tokens)?;

        let mut sampler = if self.config.temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::top_k(self.config.top_k),
                LlamaSampler::top_p(self.config.top_p, 1),
                LlamaSampler::temp(self.config.temperature),
                LlamaSampler::dist(self.config.seed),
            ])
        };

        let mut decoder = UTF_8.new_decoder();
        let mut out = String::new();

        for _ in 0..max_tokens {
            let token = sampler.sample(&self.ctx, 0);
            sampler.accept(token);

            if token == self.model.token_eos() {
                break;
            }

            let piece = self.model.token_to_piece(token, &mut decoder, false, None)?;
            out.push_str(&piece);

            let mut batch = LlamaBatch::new(1, 1);
            batch.add(token, n_past, &[0], true)?;
            self.ctx.decode(&mut batch)?;
            n_past += 1;
        }

        Ok(out)
    }
}
