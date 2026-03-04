use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::io::{BufRead, BufReader};

use super::{ConvMessage, Llm, Prompt};

pub struct OpenRouterLlm {
    client: reqwest::blocking::Client,
    api_key: String,
    model: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenRouterModel {
    pub id: String,
    pub name: String,
    pub context_length: Option<u64>,
}

impl OpenRouterLlm {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            api_key,
            model,
        }
    }

    pub fn fetch_models(api_key: &str) -> Result<Vec<OpenRouterModel>> {
        let client = reqwest::blocking::Client::new();
        let resp = client
            .get("https://openrouter.ai/api/v1/models")
            .header("Authorization", format!("Bearer {}", api_key))
            .send()?;

        if !resp.status().is_success() {
            return Err(anyhow!("OpenRouter models API error: {}", resp.status()));
        }

        let json: serde_json::Value = resp.json()?;
        let models = json["data"]
            .as_array()
            .ok_or_else(|| anyhow!("unexpected models response"))?
            .iter()
            .filter_map(|v| serde_json::from_value::<OpenRouterModel>(v.clone()).ok())
            .collect();
        Ok(models)
    }
}

impl Llm for OpenRouterLlm {
    fn generate(&mut self, prompt: &Prompt, max_tokens: usize) -> Result<String> {
        self.generate_stream(prompt, max_tokens, &mut |_| {})
    }

    fn generate_stream(
        &mut self,
        prompt: &Prompt,
        max_tokens: usize,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let mut msgs = Vec::new();
        if let Some(sys) = prompt.system.as_deref() {
            msgs.push(ConvMessage {
                role: "system".to_string(),
                content: sys.to_string(),
            });
        }
        msgs.push(ConvMessage {
            role: "user".to_string(),
            content: prompt.user.clone(),
        });
        self.generate_messages_stream(&msgs, max_tokens, on_token)
    }

    fn generate_messages_stream(
        &mut self,
        messages: &[ConvMessage],
        max_tokens: usize,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let messages_json: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();

        let body = serde_json::json!({
            "model": self.model,
            "messages": messages_json,
            "max_tokens": max_tokens,
            "stream": true,
        });

        let response = self
            .client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().unwrap_or_default();
            return Err(anyhow!("OpenRouter API error {}: {}", status, text));
        }

        let reader = BufReader::new(response);
        let mut out = String::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim().to_string();
            if line.is_empty() || line == "data: [DONE]" {
                continue;
            }
            let Some(json_str) = line.strip_prefix("data: ") else {
                continue;
            };
            let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) else {
                continue;
            };
            if let Some(content) = val["choices"][0]["delta"]["content"].as_str() {
                if !content.is_empty() {
                    on_token(content);
                    out.push_str(content);
                }
            }
        }

        Ok(out)
    }
}
