use anyhow::{anyhow, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const MAX_OUTPUT_CHARS: usize = 120_000;
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

pub fn run_shell_exec(args: &Value, root: &Path) -> Result<String> {
    let cmd = args
        .get("cmd")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("shell.exec requires cmd"))?;
    if cmd.trim().is_empty() {
        return Err(anyhow!("shell.exec cmd is empty"));
    }
    if cmd.contains('/') {
        return Err(anyhow!("cmd must be a bare command (no path separators)"));
    }

    let mut command = Command::new(cmd);
    if let Some(arr) = args.get("args").and_then(|v| v.as_array()) {
        for v in arr {
            let s = v
                .as_str()
                .ok_or_else(|| anyhow!("shell.exec args must be strings"))?;
            if s.contains('\n') || s.contains('\0') {
                return Err(anyhow!("shell.exec args cannot contain control chars"));
            }
            command.arg(s);
        }
    }

    let cwd = args.get("cwd").and_then(|v| v.as_str());
    if let Some(cwd) = cwd {
        let path = Path::new(cwd);
        let joined = if path.is_absolute() {
            path.to_path_buf()
        } else {
            root.join(path)
        };
        let normalized = normalize_path(&joined);
        if !normalized.starts_with(root) {
            return Err(anyhow!("cwd outside workspace"));
        }
        command.current_dir(normalized);
    } else {
        command.current_dir(root);
    }

    command.env("VARCTX_WORKSPACE_ROOT", root);
    command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());

    let timeout = args
        .get("timeout_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_TIMEOUT_MS);
    let timeout = Duration::from_millis(timeout.max(100));

    let mut child = command.spawn()?;
    let start = Instant::now();
    loop {
        if start.elapsed() > timeout {
            let _ = child.kill();
            return Err(anyhow!("shell.exec timeout after {} ms", timeout.as_millis()));
        }
        if let Some(status) = child.try_wait()? {
            let output = child.wait_with_output()?;
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let mut out = String::new();
            if !stdout.trim().is_empty() {
                out.push_str(stdout.trim_end());
            }
            if !stderr.trim().is_empty() {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str("STDERR:\n");
                out.push_str(stderr.trim_end());
            }
            out = truncate_output(&out, MAX_OUTPUT_CHARS);
            if !status.success() {
                return Err(anyhow!("shell.exec failed: {}", out));
            }
            return Ok(out);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn truncate_output(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text.chars().take(max_chars).collect::<String>();
    out.push_str("\n...output truncated...");
    out
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::RootDir => out.push(comp.as_os_str()),
            std::path::Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            std::path::Component::Normal(part) => out.push(part),
        }
    }
    out
}
