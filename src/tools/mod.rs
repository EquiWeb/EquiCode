use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use walkdir::WalkDir;

pub mod monty;
pub mod builtin_fs;
pub mod builtin_shell;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: u64,
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub id: u64,
    pub ok: bool,
    pub stdout: String,
    pub stderr: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecMode {
    Yolo,
    Confirm,
    Paranoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Risk {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone)]
pub struct PolicyDecision {
    pub allowed: bool,
    pub needs_approval: bool,
    pub risk: Risk,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SafetyPolicy {
    pub mode: ExecMode,
    pub workspace_root: PathBuf,
    pub allow_network: bool,
}

impl SafetyPolicy {
    pub fn classify(&self, call: &ToolCall) -> PolicyDecision {
        let (risk, class) = classify_tool(&call.name);
        if !self.allow_network && class == ToolClass::Network {
            return PolicyDecision {
                allowed: false,
                needs_approval: false,
                risk: Risk::High,
                reason: Some("network disabled".to_string()),
            };
        }

        if !paths_within_root(&call.args, &self.workspace_root) {
            return PolicyDecision {
                allowed: false,
                needs_approval: false,
                risk: Risk::High,
                reason: Some("path outside workspace".to_string()),
            };
        }

        if call.name.starts_with("shell.exec") {
            if let Some(reason) = shell_call_denied(&call.args, self.allow_network) {
                return PolicyDecision {
                    allowed: false,
                    needs_approval: false,
                    risk: Risk::High,
                    reason: Some(reason),
                };
            }
        }

        match self.mode {
            ExecMode::Paranoid => PolicyDecision {
                allowed: class == ToolClass::ReadOnly,
                needs_approval: false,
                risk,
                reason: if class == ToolClass::ReadOnly {
                    None
                } else {
                    Some("paranoid mode blocks non-read tools".to_string())
                },
            },
            ExecMode::Confirm => PolicyDecision {
                allowed: true,
                needs_approval: true,
                risk,
                reason: None,
            },
            ExecMode::Yolo => PolicyDecision {
                allowed: true,
                needs_approval: class != ToolClass::ReadOnly,
                risk,
                reason: None,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolClass {
    ReadOnly,
    Write,
    Exec,
    Network,
    Unknown,
}

fn classify_tool(name: &str) -> (Risk, ToolClass) {
    if name.starts_with("todo.") {
        return (Risk::Low, ToolClass::ReadOnly);
    }
    if name.starts_with("filesystem.read")
        || name.starts_with("filesystem.list")
        || name.starts_with("filesystem.grep")
        || name.starts_with("fs.read")
        || name.starts_with("fs.list")
    {
        return (Risk::Low, ToolClass::ReadOnly);
    }
    if name.starts_with("filesystem.write")
        || name.starts_with("filesystem.patch")
        || name.starts_with("fs.edit")
    {
        return (Risk::Medium, ToolClass::Write);
    }
    if name.starts_with("shell.exec") || name.starts_with("git.") {
        return (Risk::High, ToolClass::Exec);
    }
    if name.starts_with("code.exec") || name.starts_with("monty.exec") {
        return (Risk::High, ToolClass::Exec);
    }
    if name.starts_with("net.") {
        return (Risk::High, ToolClass::Network);
    }
    (Risk::Medium, ToolClass::Unknown)
}

fn shell_call_denied(args: &Value, allow_network: bool) -> Option<String> {
    let cmd = args.get("cmd").and_then(|v| v.as_str()).unwrap_or("").trim();
    if cmd.is_empty() {
        return Some("shell.exec requires cmd".to_string());
    }
    if cmd.contains('/') {
        return Some("cmd must be a bare command (no path separators)".to_string());
    }
    let cmd_lower = cmd.to_lowercase();
    let denied = [
        "rm",
        "sudo",
        "dd",
        "mkfs",
        "shutdown",
        "reboot",
        "chmod",
        "chown",
        "kill",
        "pkill",
        "killall",
    ];
    if denied.contains(&cmd_lower.as_str()) {
        return Some(format!("cmd blocked: {}", cmd));
    }
    if !allow_network {
        let net = ["curl", "wget", "scp", "ssh"];
        if net.contains(&cmd_lower.as_str()) {
            return Some(format!("network cmd blocked: {}", cmd));
        }
    }
    if (cmd_lower == "bash" || cmd_lower == "sh")
        && args
            .get("args")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().any(|v| v.as_str() == Some("-c") || v.as_str() == Some("-lc")))
            .unwrap_or(false)
    {
        return Some("shell.exec disallows bash/sh -c".to_string());
    }
    None
}

fn paths_within_root(args: &Value, root: &Path) -> bool {
    let mut ok = true;
    visit_paths(args, &mut |p| {
        if ok && !is_within_root(p, root) {
            ok = false;
        }
    });
    ok
}

fn visit_paths(value: &Value, f: &mut dyn FnMut(&Path)) {
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                if k == "path" {
                    if let Some(s) = v.as_str() {
                        f(Path::new(s));
                    }
                } else if k == "paths" {
                    if let Some(arr) = v.as_array() {
                        for item in arr {
                            if let Some(s) = item.as_str() {
                                f(Path::new(s));
                            }
                        }
                    }
                } else {
                    visit_paths(v, f);
                }
            }
        }
        Value::Array(arr) => {
            for v in arr {
                visit_paths(v, f);
            }
        }
        _ => {}
    }
}

fn is_within_root(path: &Path, root: &Path) -> bool {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    joined.starts_with(root)
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct SkillManifest {
    name: String,
    version: String,
    description: Option<String>,
    schema: SkillSchema,
    entrypoint: SkillEntrypoint,
}

#[derive(Debug, Clone, Deserialize)]
struct SkillSchema {
    tools: Vec<ToolSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct SkillEntrypoint {
    #[serde(rename = "type")]
    kind: String,
    command: String,
}

#[derive(Debug, Clone)]
struct Skill {
    manifest: SkillManifest,
    root: PathBuf,
}

pub struct SkillHost {
    skills: Vec<Skill>,
    tools: HashMap<String, Skill>,
}

impl SkillHost {
    const MAX_OUTPUT_CHARS: usize = 120_000;

    pub fn load(skills_root: &Path) -> Result<Self> {
        let mut skills = Vec::new();
        for entry in WalkDir::new(skills_root).into_iter().filter_map(Result::ok) {
            if entry.file_type().is_file() && entry.file_name() == "skill.json" {
                let manifest_text = std::fs::read_to_string(entry.path())?;
                let manifest: SkillManifest = serde_json::from_str(&manifest_text)?;
                if manifest.entrypoint.kind != "subprocess" {
                    return Err(anyhow!(
                        "unsupported entrypoint type: {}",
                        manifest.entrypoint.kind
                    ));
                }
                let root = entry
                    .path()
                    .parent()
                    .ok_or_else(|| anyhow!("skill.json missing parent dir"))?
                    .to_path_buf();
                skills.push(Skill { manifest, root });
            }
        }

        let mut tools = HashMap::new();
        for skill in &skills {
            for tool in &skill.manifest.schema.tools {
                tools.insert(tool.name.clone(), skill.clone());
            }
        }

        Ok(Self { skills, tools })
    }

    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        let mut out = Vec::new();
        for skill in &self.skills {
            for tool in &skill.manifest.schema.tools {
                out.push(tool.clone());
            }
        }
        out
    }

    pub fn run_tool(
        &self,
        call: &ToolCall,
        timeout: Duration,
        workspace_root: &Path,
    ) -> Result<ToolResult> {
        let Some(skill) = self.tools.get(&call.name) else {
            return Ok(ToolResult {
                id: call.id,
                ok: false,
                stdout: String::new(),
                stderr: String::new(),
                error: Some(format!("unknown tool: {}", call.name)),
            });
        };

        let cmd = &skill.manifest.entrypoint.command;
        let mut command = if cmd.contains(' ') {
            let mut c = Command::new("sh");
            c.arg("-c").arg(cmd);
            c
        } else {
            Command::new(cmd)
        };
        command.current_dir(&skill.root);
        command.env("VARCTX_WORKSPACE_ROOT", workspace_root);
        command.env("VARCTX_SKILL_ROOT", &skill.root);
        command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut child = command.spawn()?;
        if let Some(stdin) = child.stdin.as_mut() {
            let payload = serde_json::to_vec(call)?;
            use std::io::Write;
            stdin.write_all(&payload)?;
        }

        let start = Instant::now();
        loop {
            if start.elapsed() > timeout {
                let _ = child.kill();
                return Ok(ToolResult {
                    id: call.id,
                    ok: false,
                    stdout: String::new(),
                    stderr: String::new(),
                    error: Some("tool timeout".to_string()),
                });
            }
            if let Some(status) = child.try_wait()? {
                let output = child.wait_with_output()?;
                let mut ok = status.success();
                let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let mut stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let mut error = if ok {
                    None
                } else {
                    Some(format!("tool exited {}", status))
                };

                if let Ok(value) = serde_json::from_str::<Value>(&stdout) {
                    if let Some(obj) = value.as_object() {
                        if let Some(v) = obj.get("ok").and_then(|v| v.as_bool()) {
                            ok = v;
                        }
                        if let Some(v) = obj.get("stdout").and_then(|v| v.as_str()) {
                            stdout = v.to_string();
                        }
                        if let Some(v) = obj.get("stderr").and_then(|v| v.as_str()) {
                            stderr = v.to_string();
                        }
                        if let Some(v) = obj.get("error").and_then(|v| v.as_str()) {
                            error = Some(v.to_string());
                        }
                    }
                }

                stdout = truncate_output(&stdout, Self::MAX_OUTPUT_CHARS);
                stderr = truncate_output(&stderr, Self::MAX_OUTPUT_CHARS);

                return Ok(ToolResult {
                    id: call.id,
                    ok,
                    stdout,
                    stderr,
                    error,
                });
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

fn truncate_output(text: &str, max_chars: usize) -> String {
    let mut out = String::new();
    let mut count = 0usize;
    for ch in text.chars() {
        if count >= max_chars {
            break;
        }
        out.push(ch);
        count += 1;
    }
    if text.chars().count() > max_chars {
        out.push_str("\n...[truncated]");
    }
    out
}
