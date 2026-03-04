use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub enum Backend {
    #[default]
    Local,
    OpenRouter,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ExperimentConfig {
    pub user_turn_only: bool,
    pub disable_plan_phase: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Config {
    pub backend: Backend,
    pub openrouter_api_key: Option<String>,
    pub openrouter_model: Option<String>,
    pub local_model_path: Option<String>,
    pub n_ctx: Option<u32>,
    pub skills_dir: Option<String>,
    pub exec_mode: String,
    pub plan_tokens: usize,
    pub answer_tokens: usize,
    pub experiments: ExperimentConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backend: Backend::Local,
            openrouter_api_key: None,
            openrouter_model: None,
            local_model_path: None,
            n_ctx: Some(8192),
            skills_dir: None,
            exec_mode: "yolo".to_string(),
            plan_tokens: 256,
            answer_tokens: 512,
            experiments: ExperimentConfig::default(),
        }
    }
}

impl Config {
    pub fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("equicode").join("config.toml"))
    }

    pub fn load() -> Self {
        let Some(path) = Self::config_path() else {
            return Self::default();
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        toml::from_str(&text).unwrap_or_default()
    }

    pub fn save(&self) -> Result<()> {
        let Some(path) = Self::config_path() else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(self)?;
        std::fs::write(&path, text)?;
        Ok(())
    }

    pub fn is_configured(&self) -> bool {
        match self.backend {
            Backend::Local => self.local_model_path.is_some(),
            Backend::OpenRouter => {
                self.openrouter_api_key.as_ref().map(|k| !k.is_empty()).unwrap_or(false)
                    && self.openrouter_model.as_ref().map(|m| !m.is_empty()).unwrap_or(false)
            }
        }
    }
}
