use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_id: Uuid,
    pub doc_id: String,
    pub version: u32,
    pub tombstone: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// zstd-compressed UTF-8
    pub text_zstd: Vec<u8>,
    /// optional short summary (can be empty)
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarBinding {
    pub binding_id: Uuid,
    pub var_name: String,
    pub binding_version: u32,
    pub chunk_ids: Vec<Uuid>,
    pub created_at: DateTime<Utc>,
}
