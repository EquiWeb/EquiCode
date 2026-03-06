use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

fn default_max_todo_interventions() -> Option<usize> { Some(2) }

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
    /// When true: if the agent tries to produce a FINAL answer while there are
    /// pending todos, inject a System reminder and force it to continue.
    #[serde(default)]
    pub enforce_todos: bool,
    /// Max consecutive TODO-enforcement interventions per run before giving up.
    /// None = unlimited (not recommended). Default = 2.
    #[serde(default = "default_max_todo_interventions")]
    pub max_todo_interventions: Option<usize>,
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
    /// Max conversation turns to keep in history (None = unlimited).
    /// Each turn = 1 user + 1 assistant message. Oldest turns are dropped first.
    pub max_history_turns: Option<usize>,
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
            max_history_turns: Some(20),
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

    /// Returns the effective OpenRouter API key: config value if set, else $OPENROUTER_API_KEY.
    pub fn effective_openrouter_key(&self) -> Option<String> {
        self.openrouter_api_key
            .clone()
            .filter(|k| !k.is_empty())
            .or_else(|| std::env::var("OPENROUTER_API_KEY").ok().filter(|k| !k.is_empty()))
    }

    pub fn is_configured(&self) -> bool {
        match self.backend {
            Backend::Local => self.local_model_path.as_ref().map(|p| !p.is_empty()).unwrap_or(false),
            Backend::OpenRouter => {
                self.effective_openrouter_key().is_some()
                    && self.openrouter_model.as_ref().map(|m| !m.is_empty()).unwrap_or(false)
            }
        }
    }
}
