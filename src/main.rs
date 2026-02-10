use anyhow::{anyhow, Result};

use varctx_proto::llm::{Llm, LlamaCppLlm, LlmConfig};
use varctx_proto::pipeline::assertions::{run_with_retry, Constraint, ConstraintKind};
use varctx_proto::pipeline::prompt::PromptAssembler;
use varctx_proto::retrieval::OverlapRetriever;
use varctx_proto::store::ContextStore;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).init();

    let model_path = parse_model_path()
        .ok_or_else(|| anyhow!("missing model path. set VARCTX_MODEL_PATH or pass --model <path>"))?;

    let config = LlmConfig::from_env(model_path);
    let mut llm = LlamaCppLlm::load(config)?;

    let store = ContextStore::open("./varctx_db")?;

    // 1) Ingest something big (prototype)
    let doc_chunks = store.put_doc_chunked("doc:demo", &"hello world ".repeat(2000), 1200)?;
    store.bind_var("V:demo_doc", doc_chunks.clone())?;

    // 2) Candidate search space: chunks bound to active vars
    let active_vars = vec!["V:demo_doc".to_string()];
    let candidates = doc_chunks;

    // 3) Retriever
    let retriever = OverlapRetriever { top_k: 8 };

    // 4) Assertions (Suggest): enforce format/length/etc.
    let constraints = vec![
        Constraint {
            kind: ConstraintKind::Suggest,
            message: "Output must be < 160 chars.".to_string(),
            check: |s| s.chars().count() < 160,
            max_retries: 2,
        },
        Constraint {
            kind: ConstraintKind::Suggest,
            message: "Output must contain the word 'prompt'.".to_string(),
            check: |s| s.to_lowercase().contains("prompt"),
            max_retries: 2,
        },
    ];

    let question = "Generate a concise retrieval query about the demo doc.";

    let (out, attempts) = run_with_retry(&constraints, |attempts| {
        let retrieved = retriever.retrieve(&store, question, &candidates)?;
        let assembler = PromptAssembler { store: &store };
        let prompt = assembler.build(question, &active_vars, &retrieved, attempts)?;
        llm.generate(&prompt, 128)
    })?;

    println!("FINAL:\n{out}\n\nAttempts: {}", attempts.len());
    Ok(())
}

fn parse_model_path() -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--model" {
            return args.next();
        }
    }
    if let Ok(p) = std::env::var("VARCTX_MODEL_PATH") {
        return Some(p);
    }
    let default = "Qwen2.5-Coder-3B-Instruct-F16.gguf";
    if std::path::Path::new(default).exists() {
        return Some(default.to_string());
    }
    None
}
