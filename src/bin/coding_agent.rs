use anyhow::{anyhow, Result};
use std::fs;

use varctx_proto::agent::context::{
    build_context, resolve_candidate_chunks, summarize_var, ContextBuildConfig,
};
use varctx_proto::agent::{CodingAgent, CodingAgentConfig};
use varctx_proto::llm::{LlamaCppLlm, LlmConfig};
use varctx_proto::retrieval::OverlapRetriever;
use varctx_proto::store::ContextStore;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).init();

    let args = Args::from_env()?;

    if args.ingest_file.is_some() {
        ingest_var_from_file(&args)?;
    }

    let needs_llm = args.task.is_some() || !args.summarize_vars.is_empty();
    if !needs_llm {
        return Ok(());
    }

    let model_path = parse_model_path(args.model_override.clone())
        .ok_or_else(|| anyhow!("missing model path. set VARCTX_MODEL_PATH or pass --model <path>"))?;

    let config = LlmConfig::from_env(model_path);
    let mut llm = LlamaCppLlm::load(config)?;

    let mut agent_cfg = CodingAgentConfig::default();
    if let Some(tokens) = args.plan_tokens {
        agent_cfg.max_plan_tokens = tokens.max(1);
    }
    if let Some(tokens) = args.answer_tokens {
        agent_cfg.max_answer_tokens = tokens.max(1);
    }

    if !args.summarize_vars.is_empty() {
        update_var_summaries(&mut llm, &args)?;
    }

    let context = if let Some(task) = args.task.as_deref() {
        build_runtime_context(&args, task)?
    } else {
        None
    };

    if let Some(task) = args.task.as_deref() {
        let mut agent = CodingAgent::new(&mut llm, agent_cfg);
        let result = agent.run(task, context.as_deref())?;

        println!("PLAN:\n{}\n", result.plan.trim());
        println!("ANSWER:\n{}\n", result.answer.trim());
    }
    Ok(())
}

struct Args {
    model_override: Option<String>,
    task: Option<String>,
    context: Option<String>,
    store_path: Option<String>,
    vars: Vec<String>,
    top_k: Option<usize>,
    max_snippets: Option<usize>,
    snippet_chars: Option<usize>,
    summarize_vars: Vec<String>,
    summarize_max_tokens: Option<usize>,
    plan_tokens: Option<usize>,
    answer_tokens: Option<usize>,
    ingest_file: Option<String>,
    ingest_doc_id: Option<String>,
    ingest_var: Option<String>,
    ingest_chunk_chars: Option<usize>,
    rebuild_store: bool,
}

impl Args {
    fn from_env() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut model_override = None;
        let mut task = None;
        let mut context = None;
        let mut context_file = None;
        let mut store_path = None;
        let mut vars: Vec<String> = Vec::new();
        let mut top_k = None;
        let mut max_snippets = None;
        let mut snippet_chars = None;
        let mut summarize_vars: Vec<String> = Vec::new();
        let mut summarize_max_tokens = None;
        let mut plan_tokens = None;
        let mut answer_tokens = None;
        let mut ingest_file = None;
        let mut ingest_doc_id = None;
        let mut ingest_var = None;
        let mut ingest_chunk_chars = None;
        let mut rebuild_store = false;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => model_override = args.next(),
                "--task" => task = args.next(),
                "--context" => context = args.next(),
                "--context-file" => context_file = args.next(),
                "--store" => store_path = args.next(),
                "--vars" => {
                    if let Some(v) = args.next() {
                        vars.extend(
                            v.split(',')
                                .map(|s| s.trim().to_string())
                                .filter(|s| !s.is_empty()),
                        );
                    }
                }
                "--summarize-var" => {
                    if let Some(v) = args.next() {
                        summarize_vars.push(v);
                    }
                }
                "--summarize-max-tokens" => {
                    if let Some(v) = args.next() {
                        summarize_max_tokens = v.parse::<usize>().ok();
                    }
                }
                "--top-k" => {
                    if let Some(v) = args.next() {
                        top_k = v.parse::<usize>().ok();
                    }
                }
                "--max-snippets" => {
                    if let Some(v) = args.next() {
                        max_snippets = v.parse::<usize>().ok();
                    }
                }
                "--snippet-chars" => {
                    if let Some(v) = args.next() {
                        snippet_chars = v.parse::<usize>().ok();
                    }
                }
                "--plan-tokens" => {
                    if let Some(v) = args.next() {
                        plan_tokens = v.parse::<usize>().ok();
                    }
                }
                "--answer-tokens" => {
                    if let Some(v) = args.next() {
                        answer_tokens = v.parse::<usize>().ok();
                    }
                }
                "--ingest-file" => ingest_file = args.next(),
                "--ingest-doc-id" => ingest_doc_id = args.next(),
                "--bind-var" => ingest_var = args.next(),
                "--chunk-chars" => {
                    if let Some(v) = args.next() {
                        ingest_chunk_chars = v.parse::<usize>().ok();
                    }
                }
                "--rebuild-store" => rebuild_store = true,
                _ => {}
            }
        }

        if task.is_none() && summarize_vars.is_empty() && ingest_file.is_none() {
            return Err(anyhow!(
                "missing --task <text>, --summarize-var <V:...>, or --ingest-file <path>"
            ));
        }

        let mut combined_context = context;
        if let Some(path) = context_file {
            let file_ctx = fs::read_to_string(path)?;
            combined_context = match combined_context {
                Some(mut existing) => {
                    existing.push_str("\n\n");
                    existing.push_str(&file_ctx);
                    Some(existing)
                }
                None => Some(file_ctx),
            };
        }

        Ok(Self {
            model_override,
            task,
            context: combined_context,
            store_path,
            vars,
            top_k,
            max_snippets,
            snippet_chars,
            summarize_vars,
            summarize_max_tokens,
            plan_tokens,
            answer_tokens,
            ingest_file,
            ingest_doc_id,
            ingest_var,
            ingest_chunk_chars,
            rebuild_store,
        })
    }
}

fn build_runtime_context(args: &Args, task: &str) -> Result<Option<String>> {
    let mut context = args.context.clone();

    let Some(store_path) = args.store_path.as_deref() else {
        return Ok(context);
    };

    if args.vars.is_empty() {
        return Err(anyhow!("--store requires --vars <V:...,...>"));
    }

    let store = ContextStore::open(store_path)?;
    let candidates = resolve_candidate_chunks(&store, &args.vars)?;
    let retriever = OverlapRetriever {
        top_k: args.top_k.unwrap_or(8),
    };
    let retrieved = retriever.retrieve(&store, task, &candidates)?;

    let cfg = ContextBuildConfig {
        max_snippets: args.max_snippets.unwrap_or(6),
        snippet_chars: args.snippet_chars.unwrap_or(800),
    };
    let auto_ctx = build_context(&store, &args.vars, &retrieved, &cfg)?;

    context = match context {
        Some(mut existing) => {
            existing.push_str("\n\n");
            existing.push_str(&auto_ctx);
            Some(existing)
        }
        None => Some(auto_ctx),
    };

    Ok(context)
}

fn update_var_summaries(llm: &mut LlamaCppLlm, args: &Args) -> Result<()> {
    let store_path = args
        .store_path
        .as_deref()
        .ok_or_else(|| anyhow!("--summarize-var requires --store <path>"))?;

    let store = ContextStore::open(store_path)?;
    let cfg = ContextBuildConfig {
        max_snippets: args.max_snippets.unwrap_or(6),
        snippet_chars: args.snippet_chars.unwrap_or(800),
    };
    let max_tokens = args.summarize_max_tokens.unwrap_or(128);

    for var in &args.summarize_vars {
        let summary = summarize_var(llm, &store, var, &cfg, max_tokens)?;
        store.update_var_summary(var, &summary)?;
        println!("SUMMARY[{}]: {}", var, summary);
    }

    Ok(())
}

fn ingest_var_from_file(args: &Args) -> Result<()> {
    let Some(path) = args.ingest_file.as_deref() else {
        return Ok(());
    };
    let store_path = args
        .store_path
        .as_deref()
        .ok_or_else(|| anyhow!("--ingest-file requires --store <path>"))?;
    let var_name = args
        .ingest_var
        .as_deref()
        .ok_or_else(|| anyhow!("--ingest-file requires --bind-var <V:...>"))?;

    let doc_id = args
        .ingest_doc_id
        .clone()
        .unwrap_or_else(|| {
            let fname = std::path::Path::new(path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("ingest");
            format!("doc:{fname}")
        });

    let text = fs::read_to_string(path)?;
    if args.rebuild_store {
        remove_store_path(store_path)?;
    }

    let store = ContextStore::open(store_path)?;
    let chunk_chars = args.ingest_chunk_chars.unwrap_or(1200);
    let chunks = store.put_doc_chunked(&doc_id, &text, chunk_chars)?;
    store.bind_var(var_name, chunks.clone())?;

    println!(
        "INGESTED[{}] -> {} ({} chunks)",
        doc_id,
        var_name,
        chunks.len()
    );
    Ok(())
}

fn remove_store_path(path: &str) -> Result<()> {
    let p = std::path::Path::new(path);
    if !p.exists() {
        return Ok(());
    }
    let meta = std::fs::metadata(p)?;
    if meta.is_file() {
        fs::remove_file(p)?;
    } else {
        fs::remove_dir_all(p)?;
    }
    Ok(())
}

fn parse_model_path(override_path: Option<String>) -> Option<String> {
    if let Some(p) = override_path {
        return Some(p);
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
