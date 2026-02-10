use anyhow::Result;
use uuid::Uuid;

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
    ) -> Result<String> {
        let mut p = String::new();

        p.push_str("SYSTEM: You answer using provided variables and retrieved excerpts.\n");
        p.push_str("If constraints fail, revise accordingly.\n\n");

        // Retry feedback (paper-style)
        if !attempts.is_empty() {
            p.push_str("RETRY_FEEDBACK:\n");
            for (i, a) in attempts.iter().enumerate() {
                p.push_str(&format!("- Attempt {} output:\n{}\n", i + 1, a.output));
                p.push_str(&format!("  Error:\n{}\n", a.error_msg));
            }
            p.push('\n');
        }

        p.push_str("QUESTION:\n");
        p.push_str(question);
        p.push_str("\n\n");

        p.push_str("VARIABLES_AVAILABLE:\n");
        for v in active_vars {
            p.push_str(&format!("- {}\n", v));
        }
        p.push('\n');

        p.push_str("RETRIEVED_EXCERPTS:\n");
        for (cid, score) in retrieved.iter().take(6) {
            if let Some(txt) = self.store.get_chunk_text(*cid)? {
                let snippet = txt.chars().take(800).collect::<String>();
                p.push_str(&format!(
                    "\n[chunk {} score {:.3}]\n{}\n",
                    cid, score, snippet
                ));
            }
        }

        p.push_str("\nINSTRUCTION:\nRespond clearly. If you reference a variable, name it.\n");
        Ok(p)
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

        assert!(prompt.contains("RETRY_FEEDBACK"));
        assert!(prompt.contains("V:test"));
        assert!(prompt.contains("foo bar baz"));

        fs::remove_dir_all(path)?;
        Ok(())
    }
}
