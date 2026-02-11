use anyhow::{anyhow, Result};
use encoding_rs::UTF_8;
use std::mem::ManuallyDrop;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

#[derive(Debug, Clone)]
pub struct Prompt {
    pub system: Option<String>,
    pub user: String,
}

impl Prompt {
    pub fn new(user: impl Into<String>) -> Self {
        Self {
            system: None,
            user: user.into(),
        }
    }
}

pub trait Llm {
    fn generate(&mut self, prompt: &Prompt, max_tokens: usize) -> Result<String>;

    fn generate_stream(
        &mut self,
        prompt: &Prompt,
        max_tokens: usize,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let out = self.generate(prompt, max_tokens)?;
        if !out.is_empty() {
            on_token(&out);
        }
        Ok(out)
    }
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
    pub use_chat_template: bool,
    pub system_prompt: Option<String>,
    pub silence_logs: bool,
    pub cancel_flag: Option<Arc<AtomicBool>>,
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
            use_chat_template: true,
            system_prompt: None,
            silence_logs: false,
            cancel_flag: None,
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
        if let Ok(v) = std::env::var("VARCTX_USE_CHAT_TEMPLATE") {
            let v = v.to_lowercase();
            if v == "1" || v == "true" || v == "yes" {
                cfg.use_chat_template = true;
            } else if v == "0" || v == "false" || v == "no" {
                cfg.use_chat_template = false;
            }
        }
        if let Ok(v) = std::env::var("VARCTX_SYSTEM_PROMPT") {
            if !v.trim().is_empty() {
                cfg.system_prompt = Some(v);
            }
        }
        if let Ok(v) = std::env::var("VARCTX_SILENCE_LOGS") {
            let v = v.to_lowercase();
            if v == "1" || v == "true" || v == "yes" {
                cfg.silence_logs = true;
            } else if v == "0" || v == "false" || v == "no" {
                cfg.silence_logs = false;
            }
        }

        cfg
    }
}

pub struct LlamaCppLlm {
    _backend: LlamaBackend,
    model_ptr: *mut LlamaModel,
    ctx: ManuallyDrop<llama_cpp_2::context::LlamaContext<'static>>,
    config: LlmConfig,
    cancel_flag: Option<Arc<AtomicBool>>,
}

impl LlamaCppLlm {
    pub fn load(config: LlmConfig) -> Result<Self> {
        if !Path::new(&config.model_path).exists() {
            return Err(anyhow!(
                "model not found at {}",
                config.model_path.display()
            ));
        }

        let mut backend = LlamaBackend::init()?;
        if config.silence_logs {
            backend.void_logs();
        }

        let mut model_params = LlamaModelParams::default();
        if let Some(n_gpu_layers) = config.n_gpu_layers {
            model_params = model_params.with_n_gpu_layers(n_gpu_layers);
        }

        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)?;
        let model_ptr = Box::into_raw(Box::new(model));
        let model_ref: &'static LlamaModel = unsafe { &*model_ptr };

        let mut ctx_params = LlamaContextParams::default()
            .with_n_batch(config.n_batch)
            .with_n_ubatch(config.n_ubatch);
        if let Some(n_ctx) = config.n_ctx {
            ctx_params = ctx_params.with_n_ctx(NonZeroU32::new(n_ctx));
        }

        let ctx = model_ref.new_context(&backend, ctx_params)?;

        Ok(Self {
            _backend: backend,
            model_ptr,
            ctx: ManuallyDrop::new(ctx),
            cancel_flag: config.cancel_flag.clone(),
            config,
        })
    }

    fn model(&self) -> &LlamaModel {
        unsafe { &*self.model_ptr }
    }

    fn decode_prompt(
        &mut self,
        tokens: &[llama_cpp_2::token::LlamaToken],
    ) -> Result<(i32, i32)> {
        let mut n_past: i32 = 0;
        let batch_cap = self.config.n_batch.max(1) as usize;
        let mut last_logits_idx: i32 = 0;

        let mut i = 0;
        while i < tokens.len() {
            let end = (i + batch_cap).min(tokens.len());
            let batch_len = end - i;
            let mut batch = LlamaBatch::new(end - i, 1);

            for (j, token) in tokens[i..end].iter().enumerate() {
                let pos = n_past + j as i32;
                let logits = j + 1 == batch_len;
                batch.add(*token, pos, &[0], logits)?;
            }

            self.ctx.decode(&mut batch)?;
            n_past += (end - i) as i32;
            last_logits_idx = (batch_len.saturating_sub(1)) as i32;
            i = end;
        }

        Ok((n_past, last_logits_idx))
    }
}

impl Llm for LlamaCppLlm {
    fn generate(&mut self, prompt: &Prompt, max_tokens: usize) -> Result<String> {
        self.generate_stream(prompt, max_tokens, &mut |_| {})
    }

    fn generate_stream(
        &mut self,
        prompt: &Prompt,
        max_tokens: usize,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String> {
        self.ctx.clear_kv_cache();

        let system_msg = prompt
            .system
            .as_deref()
            .or(self.config.system_prompt.as_deref())
            .map(str::to_string);

        let (prompt_text, add_bos) = if self.config.use_chat_template {
            if let Ok(tmpl) = self.model().chat_template(None) {
                let mut msgs = Vec::new();
                if let Some(sys) = system_msg.as_deref() {
                    if !sys.trim().is_empty() {
                        msgs.push(LlamaChatMessage::new("system".to_string(), sys.to_string())?);
                    }
                }
                msgs.push(LlamaChatMessage::new(
                    "user".to_string(),
                    prompt.user.to_string(),
                )?);
                if let Ok(rendered) = self.model().apply_chat_template(&tmpl, &msgs, true) {
                    (rendered, AddBos::Never)
                } else {
                    (fallback_prompt(system_msg.as_deref(), &prompt.user), AddBos::Always)
                }
            } else {
                (fallback_prompt(system_msg.as_deref(), &prompt.user), AddBos::Always)
            }
        } else {
            (fallback_prompt(system_msg.as_deref(), &prompt.user), AddBos::Always)
        };

        let tokens = self.model().str_to_token(&prompt_text, add_bos)?;
        if tokens.is_empty() {
            return Ok(String::new());
        }

        let (mut n_past, mut sample_idx) = self.decode_prompt(&tokens)?;

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
            if let Some(flag) = &self.cancel_flag {
                if flag.load(Ordering::SeqCst) {
                    break;
                }
            }
            let token = sampler.sample(&self.ctx, sample_idx);
            sampler.accept(token);
            sample_idx = 0;

            if token == self.model().token_eos() {
                break;
            }

            let piece = self.model().token_to_piece(token, &mut decoder, false, None)?;
            if !piece.is_empty() {
                on_token(&piece);
            }
            out.push_str(&piece);

            let mut batch = LlamaBatch::new(1, 1);
            batch.add(token, n_past, &[0], true)?;
            self.ctx.decode(&mut batch)?;
            n_past += 1;
        }

        Ok(out)
    }
}

fn fallback_prompt(system: Option<&str>, user: &str) -> String {
    let mut out = String::new();
    if let Some(sys) = system {
        if !sys.trim().is_empty() {
            out.push_str("SYSTEM:\n");
            out.push_str(sys);
            out.push_str("\n\n");
        }
    }
    out.push_str("USER:\n");
    out.push_str(user);
    out
}

impl Drop for LlamaCppLlm {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.ctx);
            drop(Box::from_raw(self.model_ptr));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_generate_smoke() -> Result<()> {
        if std::env::var("VARCTX_RUN_LLM_TESTS").ok().as_deref() != Some("1") {
            return Ok(());
        }

        let model_path = std::env::var("VARCTX_MODEL_PATH").ok().or_else(|| {
            let default = "Qwen2.5-Coder-3B-Instruct-F16.gguf";
            if Path::new(default).exists() {
                Some(default.to_string())
            } else {
                None
            }
        });

        let Some(model_path) = model_path else {
            return Ok(());
        };

        let mut cfg = LlmConfig::new(model_path);
        cfg.n_ctx = Some(1024);
        cfg.n_batch = 128;
        cfg.n_ubatch = 128;
        cfg.temperature = 0.0;
        cfg.top_k = 1;
        cfg.top_p = 1.0;

        let mut llm = LlamaCppLlm::load(cfg)?;
        let out = llm.generate(&Prompt::new("Say hello in 3 words."), 8)?;
        assert!(!out.trim().is_empty());
        Ok(())
    }
}
