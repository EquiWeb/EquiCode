use anyhow::Result;
use serde_json::json;
use std::path::PathBuf;

use varctx_proto::tools::monty::run_code_exec;
use varctx_proto::tools::{ExecMode, SafetyPolicy};

fn main() -> Result<()> {
    let policy = SafetyPolicy {
        mode: ExecMode::Paranoid,
        workspace_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        allow_network: false,
    };

    let args = json!({
        "code": "print('Hello, World!')",
        "inputs": {},
        "input_names": []
    });

    let output = run_code_exec(&args, &policy)?;
    println!("{output}");
    Ok(())
}
