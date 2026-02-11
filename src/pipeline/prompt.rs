use anyhow::Result;
use uuid::Uuid;

use crate::llm::Prompt;
use crate::pipeline::assertions::Attempt;
use crate::store::ContextStore;

pub struct PromptAssembler<'a> {
    pub store: &'a ContextStore,
}

impl<'a> PromptAssembler<'a> {
    pub fn build(
        &self,
        question: &str,
        active_vars: &[String],
        retrieved: &[(Uuid, f32)],
        attempts: &[Attempt],
    ) -> Result<Prompt> {
        let system = "You answer using provided variables and retrieved excerpts.\n\
If constraints fail, revise accordingly."
            .to_string();

        let mut user = String::new();

        // Retry feedback (paper-style)
        if !attempts.is_empty() {
            user.push_str("RETRY_FEEDBACK:\n");
            for (i, a) in attempts.iter().enumerate() {
                user.push_str(&format!("- Attempt {} output:\n{}\n", i + 1, a.output));
                user.push_str(&format!("  Error:\n{}\n", a.error_msg));
            }
            user.push('\n');
        }

        user.push_str("QUESTION:\n");
        user.push_str(question);
        user.push_str("\n\n");

        user.push_str("VARIABLES_AVAILABLE:\n");
        for v in active_vars {
            let summary = self
                .store
                .get_var_binding_latest_lossy(v)?
                .map(|b| b.summary)
                .unwrap_or_default();
            if summary.trim().is_empty() {
                user.push_str(&format!("- {}\n", v));
            } else {
                user.push_str(&format!("- {}: {}\n", v, summary.trim()));
            }
        }
        user.push('\n');

        user.push_str("RETRIEVED_EXCERPTS:\n");
        for (cid, score) in retrieved.iter().take(6) {
            if let Some(txt) = self.store.get_chunk_text_lossy(*cid)? {
                let snippet = txt.chars().take(800).collect::<String>();
                user.push_str(&format!(
                    "\n[chunk {} score {:.3}]\n{}\n",
                    cid, score, snippet
                ));
            }
        }

        user.push_str("\nINSTRUCTION:\nRespond clearly. If you reference a variable, name it.\n");
        Ok(Prompt {
            system: Some(system),
            user,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::ContextStore;
    use std::fs;
    use uuid::Uuid;

    fn temp_dir(name: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!("varctx_proto_prompt_{name}_{}", Uuid::new_v4()));
        p.to_string_lossy().to_string()
    }

    #[test]
    fn prompt_includes_attempts_vars_and_snippets() -> Result<()> {
        let path = temp_dir("prompt_includes_attempts_vars_and_snippets");
        let store = ContextStore::open(&path)?;

        let ids = store.put_doc_chunked("doc:prompt", "foo bar baz", 200)?;
        store.bind_var_with_summary("V:test", ids.clone(), Some("test summary".to_string()))?;
        let retrieved = vec![(ids[0], 0.9)];
        let attempts = vec![Attempt {
            output: "bad output".to_string(),
            error_msg: "must be concise".to_string(),
        }];

        let assembler = PromptAssembler { store: &store };
        let prompt = assembler.build(
            "What is foo?",
            &["V:test".to_string()],
            &retrieved,
            &attempts,
        )?;

        let sys = prompt.system.as_deref().unwrap_or("");
        assert!(sys.contains("retrieved excerpts"));
        assert!(prompt.user.contains("RETRY_FEEDBACK"));
        assert!(prompt.user.contains("V:test"));
        assert!(prompt.user.contains("test summary"));
        assert!(prompt.user.contains("foo bar baz"));

        fs::remove_dir_all(path)?;
        Ok(())
    }
}
