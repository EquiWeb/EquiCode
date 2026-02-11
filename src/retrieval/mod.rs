use std::collections::HashSet;
use uuid::Uuid;

use anyhow::Result;
use crate::store::ContextStore;

/// Very cheap baseline: rank by token overlap against chunk text.
/// Replace with BM25 once the plumbing works.
pub struct OverlapRetriever {
    pub top_k: usize,
}

impl OverlapRetriever {
    pub fn retrieve(
        &self,
        store: &ContextStore,
        query: &str,
        candidate_chunk_ids: &[Uuid],
    ) -> Result<Vec<(Uuid, f32)>> {
        let q = tokenize(query);
        let mut scored = Vec::new();

        for cid in candidate_chunk_ids {
            if let Some(txt) = store.get_chunk_text_lossy(*cid)? {
                let t = tokenize(&txt);
                let inter = q.intersection(&t).count() as f32;
                if inter > 0.0 {
                    let score = inter / (q.len() as f32).max(1.0);
                    scored.push((*cid, score));
                }
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(self.top_k);
        Ok(scored)
    }
}

fn tokenize(s: &str) -> HashSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3)
        .map(|w| w.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::ContextStore;
    use std::fs;
    use uuid::Uuid;

    fn temp_dir(name: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!("varctx_proto_retrieval_{name}_{}", Uuid::new_v4()));
        p.to_string_lossy().to_string()
    }

    #[test]
    fn overlap_retriever_scores_expected_chunk() -> Result<()> {
        let path = temp_dir("overlap_retriever_scores_expected_chunk");
        let store = ContextStore::open(&path)?;

        let doc = "alpha beta gamma ".repeat(40) + "foo bar baz ".repeat(40).as_str();
        let chunk_ids = store.put_doc_chunked("doc:retrieval", &doc, 200)?;
        assert!(chunk_ids.len() >= 2);

        let retriever = OverlapRetriever { top_k: 2 };
        let scored = retriever.retrieve(&store, "alpha gamma", &chunk_ids)?;
        assert!(!scored.is_empty());

        let top_text = store.get_chunk_text(scored[0].0)?.unwrap();
        assert!(top_text.contains("alpha"));

        fs::remove_dir_all(path)?;
        Ok(())
    }
}
