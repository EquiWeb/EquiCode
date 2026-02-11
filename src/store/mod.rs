use anyhow::{anyhow, Result};
use chrono::Utc;
use uuid::Uuid;

use crate::store::types::{Chunk, VarBinding};

pub mod types;

fn zstd_compress(s: &str) -> Result<Vec<u8>> {
    Ok(zstd::encode_all(s.as_bytes(), 3)?)
}

fn zstd_decompress(bytes: &[u8]) -> Result<String> {
    let raw = zstd::decode_all(bytes)?;
    Ok(String::from_utf8(raw)?)
}

pub struct ContextStore {
    db: sled::Db,
    chunks: sled::Tree,
    vars: sled::Tree,
    docs: sled::Tree, // doc_id -> Vec<Uuid>
}

impl ContextStore {
    pub fn open(path: &str) -> Result<Self> {
        let db = sled::open(path)?;
        Ok(Self {
            chunks: db.open_tree("chunks")?,
            vars: db.open_tree("vars")?,
            docs: db.open_tree("docs")?,
            db,
        })
    }

    pub fn put_doc_chunked(
        &self,
        doc_id: &str,
        text: &str,
        chunk_chars: usize,
    ) -> Result<Vec<Uuid>> {
        // naive chunker for prototype; replace with token-aware chunking later
        let parts = text
            .chars()
            .collect::<Vec<_>>()
            .chunks(chunk_chars.max(256))
            .map(|c| c.iter().collect::<String>())
            .collect::<Vec<_>>();

        // tombstone existing chunks for doc
        if let Some(prev) = self.docs.get(doc_id.as_bytes())? {
            let prev_ids: Vec<Uuid> = bincode::deserialize(&prev)?;
            for cid in prev_ids {
                if let Some(old) = self.get_chunk(cid)? {
                    let mut old2 = old.clone();
                    old2.tombstone = true;
                    old2.updated_at = Utc::now();
                    self.put_chunk(&old2)?;
                }
            }
        }

        let mut new_ids = Vec::with_capacity(parts.len());
        for (i, p) in parts.iter().enumerate() {
            let cid = Uuid::new_v4();
            let chunk = Chunk {
                chunk_id: cid,
                doc_id: doc_id.to_string(),
                version: (i as u32) + 1,
                tombstone: false,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                text_zstd: zstd_compress(p)?,
                summary: String::new(),
            };
            self.put_chunk(&chunk)?;
            new_ids.push(cid);
        }

        self.docs
            .insert(doc_id.as_bytes(), bincode::serialize(&new_ids)?)?;
        Ok(new_ids)
    }

    pub fn put_chunk(&self, chunk: &Chunk) -> Result<()> {
        self.chunks
            .insert(chunk.chunk_id.as_bytes(), bincode::serialize(chunk)?)?;
        Ok(())
    }

    pub fn get_chunk(&self, chunk_id: Uuid) -> Result<Option<Chunk>> {
        match self.chunks.get(chunk_id.as_bytes())? {
            Some(v) => Ok(Some(bincode::deserialize(&v)?)),
            None => Ok(None),
        }
    }

    pub fn get_chunk_text(&self, chunk_id: Uuid) -> Result<Option<String>> {
        let Some(c) = self.get_chunk(chunk_id)? else { return Ok(None); };
        if c.tombstone {
            return Ok(None);
        }
        Ok(Some(zstd_decompress(&c.text_zstd)?))
    }

    pub fn get_chunk_text_lossy(&self, chunk_id: Uuid) -> Result<Option<String>> {
        let Some(raw) = self.chunks.get(chunk_id.as_bytes())? else {
            return Ok(None);
        };
        let c: Chunk = match bincode::deserialize(&raw) {
            Ok(c) => c,
            Err(_) => return Ok(None),
        };
        if c.tombstone {
            return Ok(None);
        }
        Ok(zstd_decompress(&c.text_zstd).ok())
    }

    pub fn bind_var(&self, var_name: &str, chunk_ids: Vec<Uuid>) -> Result<VarBinding> {
        self.bind_var_with_summary(var_name, chunk_ids, None)
    }

    pub fn bind_var_with_summary(
        &self,
        var_name: &str,
        chunk_ids: Vec<Uuid>,
        summary: Option<String>,
    ) -> Result<VarBinding> {
        let now = Utc::now();
        let (next_ver, _) = self
            .get_var_binding_latest(var_name)?
            .map(|b| (b.binding_version + 1, b))
            .unwrap_or((
                1,
                VarBinding {
                    binding_id: Uuid::nil(),
                    var_name: var_name.to_string(),
                    binding_version: 0,
                    chunk_ids: vec![],
                    summary: String::new(),
                    created_at: now,
                },
            ));

        let binding = VarBinding {
            binding_id: Uuid::new_v4(),
            var_name: var_name.to_string(),
            binding_version: next_ver,
            chunk_ids,
            summary: summary.unwrap_or_default(),
            created_at: now,
        };

        // key: var_name -> latest binding (overwrite), keep history separately if you want
        self.vars
            .insert(var_name.as_bytes(), bincode::serialize(&binding)?)?;
        Ok(binding)
    }

    pub fn get_var_binding_latest(&self, var_name: &str) -> Result<Option<VarBinding>> {
        match self.vars.get(var_name.as_bytes())? {
            Some(v) => Ok(Some(bincode::deserialize(&v)?)),
            None => Ok(None),
        }
    }

    pub fn get_var_binding_latest_lossy(&self, var_name: &str) -> Result<Option<VarBinding>> {
        let Some(raw) = self.vars.get(var_name.as_bytes())? else {
            return Ok(None);
        };
        Ok(bincode::deserialize(&raw).ok())
    }

    pub fn update_var_summary(&self, var_name: &str, summary: &str) -> Result<VarBinding> {
        let Some(binding) = self.get_var_binding_latest(var_name)? else {
            return Err(anyhow!("unknown var: {var_name}"));
        };
        self.bind_var_with_summary(var_name, binding.chunk_ids, Some(summary.to_string()))
    }

    pub fn materialize_var(&self, var_name: &str, max_chars: usize) -> Result<String> {
        let Some(b) = self.get_var_binding_latest(var_name)? else {
            return Err(anyhow!("unknown var: {var_name}"));
        };
        let mut out = String::new();
        for cid in b.chunk_ids {
            if let Some(t) = self.get_chunk_text(cid)? {
                out.push_str(&t);
                out.push_str("\n");
                if out.len() >= max_chars {
                    break;
                }
            }
        }
        Ok(out)
    }

    pub fn list_doc_chunks(&self, doc_id: &str) -> Result<Vec<Uuid>> {
        match self.docs.get(doc_id.as_bytes())? {
            Some(v) => Ok(bincode::deserialize(&v)?),
            None => Ok(vec![]),
        }
    }

    pub fn db(&self) -> &sled::Db {
        &self.db
    }
}

#[cfg(test)]
fn test_temp_dir(name: &str) -> String {
    let mut p = std::env::temp_dir();
    p.push(format!("varctx_proto_{name}_{}", Uuid::new_v4()));
    p.to_string_lossy().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn put_doc_and_get_text() -> Result<()> {
        let path = test_temp_dir("put_doc_and_get_text");
        let store = ContextStore::open(&path)?;

        let doc = "hello world ".repeat(1000);
        let chunk_ids = store.put_doc_chunked("doc:1", &doc, 200)?;
        assert!(!chunk_ids.is_empty());

        let first = store.get_chunk_text(chunk_ids[0])?.unwrap();
        assert!(first.contains("hello"));

        fs::remove_dir_all(path)?;
        Ok(())
    }

    #[test]
    fn upsert_tombstones_old_chunks() -> Result<()> {
        let path = test_temp_dir("upsert_tombstones_old_chunks");
        let store = ContextStore::open(&path)?;

        let doc1 = "alpha beta gamma ".repeat(200);
        let old_ids = store.put_doc_chunked("doc:2", &doc1, 180)?;
        assert!(!old_ids.is_empty());

        let doc2 = "delta epsilon ".repeat(200);
        let _new_ids = store.put_doc_chunked("doc:2", &doc2, 180)?;

        for cid in old_ids {
            let c = store.get_chunk(cid)?.unwrap();
            assert!(c.tombstone);
            let txt = store.get_chunk_text(cid)?;
            assert!(txt.is_none());
        }

        fs::remove_dir_all(path)?;
        Ok(())
    }

    #[test]
    fn bind_var_versions_increment() -> Result<()> {
        let path = test_temp_dir("bind_var_versions_increment");
        let store = ContextStore::open(&path)?;

        let ids1 = store.put_doc_chunked("doc:3", "hello world", 200)?;
        let b1 = store.bind_var("V:test", ids1)?;
        assert_eq!(b1.binding_version, 1);
        assert!(b1.summary.is_empty());

        let ids2 = store.put_doc_chunked("doc:4", "another doc", 200)?;
        let b2 = store.bind_var("V:test", ids2)?;
        assert_eq!(b2.binding_version, 2);
        assert!(b2.summary.is_empty());

        fs::remove_dir_all(path)?;
        Ok(())
    }

    #[test]
    fn update_var_summary_bumps_version() -> Result<()> {
        let path = test_temp_dir("update_var_summary_bumps_version");
        let store = ContextStore::open(&path)?;

        let ids = store.put_doc_chunked("doc:summary", "hello world", 200)?;
        let b1 = store.bind_var("V:sum", ids.clone())?;

        let b2 = store.update_var_summary("V:sum", "short summary")?;
        assert_eq!(b2.binding_version, b1.binding_version + 1);
        assert_eq!(b2.summary, "short summary");
        assert_eq!(b2.chunk_ids, ids);

        fs::remove_dir_all(path)?;
        Ok(())
    }

    #[test]
    fn materialize_var_respects_limit() -> Result<()> {
        let path = test_temp_dir("materialize_var_respects_limit");
        let store = ContextStore::open(&path)?;

        let text = "abc ".repeat(200);
        let ids = store.put_doc_chunked("doc:5", &text, 50)?;
        store.bind_var("V:mat", ids)?;

        let out = store.materialize_var("V:mat", 120)?;
        assert!(!out.is_empty());
        assert!(out.len() >= 120);

        fs::remove_dir_all(path)?;
        Ok(())
    }
}
