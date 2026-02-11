use std::collections::HashSet;

use anyhow::Result;
use uuid::Uuid;

use crate::agent::{CodingAgent, CodingAgentConfig};
use crate::llm::Llm;
use crate::llm::Prompt;
use crate::retrieval::OverlapRetriever;
use crate::store::ContextStore;

#[derive(Debug, Clone)]
pub struct ContextBuildConfig {
    pub max_snippets: usize,
    pub snippet_chars: usize,
}

impl Default for ContextBuildConfig {
    fn default() -> Self {
        Self {
            max_snippets: 6,
            snippet_chars: 800,
        }
    }
}

pub struct ContextAgent<'a> {
    llm: &'a mut dyn Llm,
    store: &'a ContextStore,
    retriever: OverlapRetriever,
    agent_config: CodingAgentConfig,
    context_config: ContextBuildConfig,
}

impl<'a> ContextAgent<'a> {
    pub fn new(
        llm: &'a mut dyn Llm,
        store: &'a ContextStore,
        retriever: OverlapRetriever,
        agent_config: CodingAgentConfig,
        context_config: ContextBuildConfig,
    ) -> Self {
        Self {
            llm,
            store,
            retriever,
            agent_config,
            context_config,
        }
    }

    pub fn run(&mut self, task: &str, active_vars: &[String]) -> Result<crate::agent::AgentResult> {
        let candidates = resolve_candidate_chunks(self.store, active_vars)?;
        let retrieved = self
            .retriever
            .retrieve(self.store, task, &candidates)?;
        let context = build_context(
            self.store,
            active_vars,
            &retrieved,
            &self.context_config,
        )?;

        let mut agent = CodingAgent::new(self.llm, self.agent_config.clone());
        agent.run(task, Some(&context))
    }
}

pub fn resolve_candidate_chunks(
    store: &ContextStore,
    active_vars: &[String],
) -> Result<Vec<Uuid>> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();

    for v in active_vars {
        if let Some(binding) = store.get_var_binding_latest_lossy(v)? {
            for cid in binding.chunk_ids {
                if seen.insert(cid) {
                    out.push(cid);
                }
            }
        }
    }

    Ok(out)
}

pub fn build_context(
    store: &ContextStore,
    active_vars: &[String],
    retrieved: &[(Uuid, f32)],
    config: &ContextBuildConfig,
) -> Result<String> {
    let mut out = String::new();

    out.push_str("VARIABLES_AVAILABLE:\n");
    for v in active_vars {
        let summary = store
            .get_var_binding_latest_lossy(v)?
            .map(|b| b.summary)
            .unwrap_or_default();
        if summary.trim().is_empty() {
            out.push_str(&format!("- {}\n", v));
        } else {
            out.push_str(&format!("- {}: {}\n", v, summary.trim()));
        }
    }
    out.push('\n');

    out.push_str("RETRIEVED_EXCERPTS:\n");
    for (cid, score) in retrieved.iter().take(config.max_snippets) {
        if let Some(txt) = store.get_chunk_text_lossy(*cid)? {
            let snippet = txt.chars().take(config.snippet_chars).collect::<String>();
            out.push_str(&format!(
                "\n[chunk {} score {:.3}]\n{}\n",
                cid, score, snippet
            ));
        }
    }

    Ok(out)
}

pub fn summarize_var(
    llm: &mut dyn Llm,
    store: &ContextStore,
    var_name: &str,
    config: &ContextBuildConfig,
    max_tokens: usize,
) -> Result<String> {
    let Some(binding) = store.get_var_binding_latest_lossy(var_name)? else {
        return Err(anyhow::anyhow!(
            "var binding missing or corrupt: {var_name}. Rebind or re-ingest the store."
        ));
    };

    let retrieved: Vec<(Uuid, f32)> = binding
        .chunk_ids
        .iter()
        .take(config.max_snippets)
        .map(|cid| (*cid, 1.0))
        .collect();

    let context = build_context(store, &[var_name.to_string()], &retrieved, config)?;
    let prompt = build_summary_prompt(var_name, &context);
    let summary = llm.generate(&prompt, max_tokens)?;
    Ok(summary.trim().to_string())
}

pub(crate) fn build_summary_prompt(var_name: &str, context: &str) -> Prompt {
    let system = "Summarize the variable in 1-2 sentences. Focus on its content. Avoid lists."
        .to_string();
    let mut user = String::new();
    user.push_str("VARIABLE:\n");
    user.push_str(var_name);
    user.push_str("\n\nCONTEXT:\n");
    user.push_str(context);
    user.push_str("\n\nINSTRUCTION:\nReturn only the summary text.\n");

    Prompt {
        system: Some(system),
        user,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir(name: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!("varctx_proto_ctx_{name}_{}", Uuid::new_v4()));
        p.to_string_lossy().to_string()
    }

    #[test]
    fn build_context_includes_vars_and_snippets() -> Result<()> {
        let path = temp_dir("build_context_includes_vars_and_snippets");
        let store = ContextStore::open(&path)?;

        let ids = store.put_doc_chunked("doc:ctx", "alpha beta gamma", 200)?;
        store.bind_var_with_summary("V:ctx", ids.clone(), Some("ctx summary".to_string()))?;

        let retrieved = vec![(ids[0], 0.9)];
        let context = build_context(
            &store,
            &["V:ctx".to_string()],
            &retrieved,
            &ContextBuildConfig::default(),
        )?;

        assert!(context.contains("VARIABLES_AVAILABLE"));
        assert!(context.contains("V:ctx"));
        assert!(context.contains("ctx summary"));
        assert!(context.contains("alpha beta gamma"));

        fs::remove_dir_all(path)?;
        Ok(())
    }

    #[test]
    fn build_summary_prompt_includes_var_and_context() {
        let prompt = build_summary_prompt("V:test", "ctx here");
        let sys = prompt.system.as_deref().unwrap_or("");
        assert!(sys.contains("Summarize"));
        assert!(prompt.user.contains("V:test"));
        assert!(prompt.user.contains("ctx here"));
    }
}
