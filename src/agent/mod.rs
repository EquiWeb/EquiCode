use anyhow::{anyhow, Result};
use serde_json::Value;
use std::collections::HashSet;

use crate::llm::{ConvMessage, Llm, Prompt};
use crate::pipeline::assertions::{run_with_retry, Attempt, Constraint};

pub mod context;

// ─── Built-in tool registry ───────────────────────────────────────────────────
// Single source of truth for every tool the agent can call natively.
// Adding a new built-in tool = add one entry here; the action prompt is built
// automatically.

pub struct BuiltinToolDef {
    pub name: &'static str,
    /// Short args sketch shown to the model (JSON-like hint, not strict schema).
    pub args_hint: &'static str,
    /// Optional one-line note appended in parentheses after the hint.
    pub note: Option<&'static str>,
}

pub const BUILTIN_TOOLS: &[BuiltinToolDef] = &[
    BuiltinToolDef {
        name: "fs.list",
        args_hint: r#"{"path": "...", "recursive": bool?, "max_entries": int?}"#,
        note: None,
    },
    BuiltinToolDef {
        name: "fs.read",
        args_hint: r#"{"path": "...", "start_line": int?, "end_line": int?, "head": int?, "tail": int?}"#,
        note: None,
    },
    BuiltinToolDef {
        name: "fs.edit",
        args_hint: r#"{"path": "...", "old": "...", "new": "..."}"#,
        note: Some("call fs.read on the file first"),
    },
    BuiltinToolDef {
        name: "shell.exec",
        args_hint: r#"{"cmd": "...", "args": [...], "cwd": "..."?, "timeout_ms": int?}"#,
        note: Some("cmd must be a bare command with no spaces"),
    },
    BuiltinToolDef {
        name: "code.exec",
        args_hint: r#"{"code": "...", "inputs": {...}, "input_names": [...]}"#,
        note: Some("runs Monty (Python subset); helpers: read(path), write(path,content), list(path), grep(path,pattern), exists(path)"),
    },
    BuiltinToolDef {
        name: "todo.create",
        args_hint: r#"{"task": "description"}"#,
        note: Some("create a TODO to track pending work; returns the assigned id"),
    },
    BuiltinToolDef {
        name: "todo.complete",
        args_hint: r#"{"id": <number>}"#,
        note: Some("mark a TODO item as completed"),
    },
];

/// Renders the tools section appended to the action system prompt.
/// `tool_names` is the full list of available tool names (builtin + skill).
/// `skill_schema_text` is optional extra schema detail for external/skill tools.
fn render_tools_section(tool_names: &[String], skill_schema_text: Option<&str>) -> String {
    if tool_names.is_empty() { return String::new(); }
    let mut out = String::from("\n\nAvailable tools:\n");
    for name in tool_names {
        if let Some(def) = BUILTIN_TOOLS.iter().find(|d| d.name == name.as_str()) {
            out.push_str(&format!("- {} {}", def.name, def.args_hint));
            if let Some(note) = def.note {
                out.push_str(&format!("  ({})", note));
            }
        } else {
            out.push_str(&format!("- {}", name));
        }
        out.push('\n');
    }
    if let Some(schema) = skill_schema_text {
        if !schema.trim().is_empty() {
            out.push_str("\nExternal tool schemas:\n");
            out.push_str(schema.trim());
            out.push('\n');
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CodingAgentConfig {
    pub plan_system: String,
    pub answer_system: String,
    pub action_system: String,
    pub max_plan_tokens: usize,
    pub max_answer_tokens: usize,
    pub max_action_tokens: usize,
    pub max_tool_iters: usize,
    pub plan_constraints: Vec<Constraint>,
    pub answer_constraints: Vec<Constraint>,
    pub action_constraints: Vec<Constraint>,
}

impl Default for CodingAgentConfig {
    fn default() -> Self {
        Self {
            plan_system: "You are a planning module. Produce a short, actionable plan.".to_string(),
            answer_system: "You are a coding agent. Use the plan and context to respond."
                .to_string(),
            action_system: "\
You are a coding agent with tool access. Output either:\n\
- FINAL: <final response>\n\
- a JSON object: {\"tool\": \"tool.name\", \"args\": {...}}\n\
Return only one of those.\n\n\
If the user asks to run/execute a command, read/write/edit files, or otherwise perform an \
action, you MUST output a tool JSON call (not FINAL). Do not reply with instructions for \
the user to run tools. Use one tool call per turn; after results are returned, you may \
call more tools or output FINAL.\n\
Only use tools listed under Available tools. Do not invent tool names."
                .to_string(),
            max_plan_tokens: 256,
            max_answer_tokens: 512,
            max_action_tokens: 256,
            max_tool_iters: 6,
            plan_constraints: Vec::new(),
            answer_constraints: Vec::new(),
            action_constraints: Vec::new(),
        }
    }
}

pub struct CodingAgent<'a> {
    llm: &'a mut dyn Llm,
    config: CodingAgentConfig,
}

#[derive(Debug, Clone, Default)]
pub struct ContextBreakdown {
    pub system_tokens: usize,
    pub history_user_tokens: usize,
    pub history_assistant_tokens: usize,
    pub task_tokens: usize,
    pub context_chunks_tokens: usize,
    pub tool_log_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct AgentResult {
    pub plan: String,
    pub answer: String,
    pub plan_attempts: Vec<Attempt>,
    pub answer_attempts: Vec<Attempt>,
    pub context_breakdown: ContextBreakdown,
}

#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Clone)]
enum AgentAction {
    Final(String),
    Tool(ToolCallRequest),
}

#[derive(Debug, Clone, Copy)]
pub enum StreamTarget {
    Plan,
    Answer,
    Tool,
}

#[derive(Debug, Clone, Copy)]
pub enum StreamEvent {
    Start,
    Chunk,
    End,
}

impl<'a> CodingAgent<'a> {
    pub fn new(llm: &'a mut dyn Llm, config: CodingAgentConfig) -> Self {
        Self { llm, config }
    }

    pub fn run(&mut self, task: &str, context: Option<&str>) -> Result<AgentResult> {
        let (plan, plan_attempts) = run_with_retry(&self.config.plan_constraints, |attempts| {
            let prompt = build_plan_prompt(task, context, attempts, &self.config);
            self.llm.generate(&prompt, self.config.max_plan_tokens)
        })?;

        let (answer, answer_attempts) =
            run_with_retry(&self.config.answer_constraints, |attempts| {
                let prompt = build_answer_prompt(task, context, &plan, attempts, &self.config);
                self.llm.generate(&prompt, self.config.max_answer_tokens)
            })?;

        Ok(AgentResult {
            plan,
            answer,
            plan_attempts,
            answer_attempts,
            context_breakdown: ContextBreakdown::default(),
        })
    }

    pub fn run_streaming(
        &mut self,
        task: &str,
        context: Option<&str>,
        on_stream: &mut dyn FnMut(StreamTarget, StreamEvent, &str),
    ) -> Result<AgentResult> {
        let (plan, plan_attempts) = run_with_retry(&self.config.plan_constraints, |attempts| {
            let prompt = build_plan_prompt(task, context, attempts, &self.config);
            on_stream(StreamTarget::Plan, StreamEvent::Start, "");
            let out = self.llm.generate_stream(
                &prompt,
                self.config.max_plan_tokens,
                &mut |chunk| on_stream(StreamTarget::Plan, StreamEvent::Chunk, chunk),
            )?;
            on_stream(StreamTarget::Plan, StreamEvent::End, "");
            Ok(out)
        })?;

        let (answer, answer_attempts) =
            run_with_retry(&self.config.answer_constraints, |attempts| {
                let prompt = build_answer_prompt(task, context, &plan, attempts, &self.config);
                on_stream(StreamTarget::Answer, StreamEvent::Start, "");
                let out = self.llm.generate_stream(
                    &prompt,
                    self.config.max_answer_tokens,
                    &mut |chunk| on_stream(StreamTarget::Answer, StreamEvent::Chunk, chunk),
                )?;
                on_stream(StreamTarget::Answer, StreamEvent::End, "");
                Ok(out)
            })?;

        Ok(AgentResult {
            plan,
            answer,
            plan_attempts,
            answer_attempts,
            context_breakdown: ContextBreakdown::default(),
        })
    }

    pub fn run_with_tools_streaming(
        &mut self,
        task: &str,
        context: Option<&str>,
        tool_names: &[String],
        tool_schema_text: Option<&str>,
        exec_tool: &mut dyn FnMut(&ToolCallRequest) -> Result<String>,
        on_final_check: &mut dyn FnMut() -> Option<String>,
        on_stream: &mut dyn FnMut(StreamTarget, StreamEvent, &str),
    ) -> Result<AgentResult> {
        if tool_names.is_empty() {
            return self.run_streaming(task, context, on_stream);
        }
        let (plan, plan_attempts) = run_with_retry(&self.config.plan_constraints, |attempts| {
            let prompt = build_plan_prompt(task, context, attempts, &self.config);
            on_stream(StreamTarget::Plan, StreamEvent::Start, "");
            let out = self.llm.generate_stream(
                &prompt,
                self.config.max_plan_tokens,
                &mut |chunk| on_stream(StreamTarget::Plan, StreamEvent::Chunk, chunk),
            )?;
            on_stream(StreamTarget::Plan, StreamEvent::End, "");
            Ok(out)
        })?;

        let mut attempts: Vec<Attempt> = Vec::new();
        let mut tool_context = String::new();
        let mut final_hint: Option<String> = None;
        let required_tools = required_tools_from_task(task, tool_names);
        let mut remaining_required: HashSet<String> =
            required_tools.iter().cloned().collect();
        if !required_tools.is_empty() {
            tool_context.push_str("REQUIRED_TOOLS:\n");
            for tool in &required_tools {
                tool_context.push_str("- ");
                tool_context.push_str(tool);
                tool_context.push('\n');
            }
            tool_context.push('\n');
        }

        for _ in 0..self.config.max_tool_iters.max(1) {
            let prompt = build_action_prompt(
                task,
                context,
                &plan,
                &tool_context,
                tool_names,
                tool_schema_text,
                &attempts,
                &self.config,
            );
            on_stream(StreamTarget::Tool, StreamEvent::Start, "");
            let action_out = self.llm.generate_stream(
                &prompt,
                self.config.max_action_tokens,
                &mut |chunk| on_stream(StreamTarget::Tool, StreamEvent::Chunk, chunk),
            )?;
            on_stream(StreamTarget::Tool, StreamEvent::End, "");

            if let Some(msg) = first_constraint_error(&self.config.action_constraints, &action_out) {
                attempts.push(Attempt {
                    output: action_out,
                    error_msg: msg.clone(),
                });
                tool_context.push_str("\nACTION_FAILED:\n");
                tool_context.push_str(&msg);
                tool_context.push('\n');
                continue;
            }

            match parse_action(&action_out) {
                Ok(AgentAction::Final(answer)) => {
                    if !remaining_required.is_empty() {
                        let mut remaining: Vec<String> =
                            remaining_required.iter().cloned().collect();
                        remaining.sort();
                        let msg = format!(
                            "must call required tools before FINAL: {}",
                            remaining.join(", ")
                        );
                        attempts.push(Attempt {
                            output: action_out,
                            error_msg: msg.clone(),
                        });
                        tool_context.push_str("\nTOOL_ERROR:\n");
                        tool_context.push_str(&msg);
                        tool_context.push('\n');
                        continue;
                    }
                    if let Some(reminder) = on_final_check() {
                        tool_context.push_str("\nSYSTEM_REMINDER:\n");
                        tool_context.push_str(&reminder);
                        tool_context.push('\n');
                        continue;
                    }
                    final_hint = Some(answer);
                    break;
                }
                Ok(AgentAction::Tool(call)) => {
                    if !tool_names.iter().any(|name| name == &call.name) {
                        let msg = format!(
                            "unknown tool: {}. Use one of: {}",
                            call.name,
                            tool_names.join(", ")
                        );
                        attempts.push(Attempt {
                            output: action_out,
                            error_msg: msg.clone(),
                        });
                        tool_context.push_str("\nTOOL_ERROR:\n");
                        tool_context.push_str(&msg);
                        tool_context.push('\n');
                        continue;
                    }
                    remaining_required.remove(&call.name);
                    match exec_tool(&call) {
                        Ok(result) => {
                            tool_context.push_str("\nTOOL_CALL:\n");
                            tool_context.push_str(&format!("{} {}\n", call.name, call.args));
                            tool_context.push_str("TOOL_RESULT:\n");
                            tool_context.push_str(&result);
                            tool_context.push('\n');
                        }
                        Err(err) => {
                            attempts.push(Attempt {
                                output: action_out,
                                error_msg: err.to_string(),
                            });
                            tool_context.push_str("\nTOOL_ERROR:\n");
                            tool_context.push_str(&err.to_string());
                            tool_context.push('\n');
                        }
                    }
                }
                Err(err) => {
                    attempts.push(Attempt {
                        output: action_out,
                        error_msg: err.to_string(),
                    });
                }
            }
        }

        let Some(final_hint) = final_hint else {
            return Err(anyhow!(
                "tool loop exceeded max iterations ({})",
                self.config.max_tool_iters
            ));
        };

        let (answer, answer_attempts) =
            run_with_retry(&self.config.answer_constraints, |attempts| {
                let prompt = build_answer_prompt_with_tools(
                    task,
                    context,
                    &plan,
                    &tool_context,
                    Some(&final_hint),
                    attempts,
                    &self.config,
                );
                on_stream(StreamTarget::Answer, StreamEvent::Start, "");
                let out = self.llm.generate_stream(
                    &prompt,
                    self.config.max_answer_tokens,
                    &mut |chunk| on_stream(StreamTarget::Answer, StreamEvent::Chunk, chunk),
                )?;
                on_stream(StreamTarget::Answer, StreamEvent::End, "");
                Ok(out)
            })?;

        Ok(AgentResult {
            plan,
            answer,
            plan_attempts,
            answer_attempts,
            context_breakdown: ContextBreakdown::default(),
        })
    }

    /// Multi-turn variant: accepts conversation history and user_turn_only flag.
    /// Builds prompts as message arrays for proper multi-turn context.
    pub fn run_with_history_streaming(
        &mut self,
        task: &str,
        context: Option<&str>,
        conv_history: &[ConvMessage],
        user_turn_only: bool,
        tool_names: &[String],
        tool_schema_text: Option<&str>,
        exec_tool: &mut dyn FnMut(&ToolCallRequest) -> Result<String>,
        on_final_check: &mut dyn FnMut() -> Option<String>,
        on_stream: &mut dyn FnMut(StreamTarget, StreamEvent, &str),
    ) -> Result<AgentResult> {
        if conv_history.is_empty() {
            return self.run_with_tools_streaming(
                task,
                context,
                tool_names,
                tool_schema_text,
                exec_tool,
                on_final_check,
                on_stream,
            );
        }

        // Estimate breakdown for ContextViz
        let history_user_tokens: usize = conv_history
            .iter()
            .filter(|m| m.role == "user")
            .map(|m| estimate_tokens(&m.content))
            .sum();
        let history_assistant_tokens: usize = if user_turn_only {
            0
        } else {
            conv_history
                .iter()
                .filter(|m| m.role == "assistant")
                .map(|m| estimate_tokens(&m.content))
                .sum()
        };

        let (plan, plan_attempts) = run_with_retry(&self.config.plan_constraints, |attempts| {
            let messages = build_plan_messages(
                task,
                context,
                attempts,
                &self.config,
                conv_history,
                user_turn_only,
            );
            on_stream(StreamTarget::Plan, StreamEvent::Start, "");
            let out = self.llm.generate_messages_stream(
                &messages,
                self.config.max_plan_tokens,
                &mut |chunk| on_stream(StreamTarget::Plan, StreamEvent::Chunk, chunk),
            )?;
            on_stream(StreamTarget::Plan, StreamEvent::End, "");
            Ok(out)
        })?;

        if tool_names.is_empty() {
            let (answer, answer_attempts) =
                run_with_retry(&self.config.answer_constraints, |attempts| {
                    let messages = build_answer_messages(
                        task,
                        context,
                        &plan,
                        None,
                        None,
                        attempts,
                        &self.config,
                        conv_history,
                        user_turn_only,
                    );
                    on_stream(StreamTarget::Answer, StreamEvent::Start, "");
                    let out = self.llm.generate_messages_stream(
                        &messages,
                        self.config.max_answer_tokens,
                        &mut |chunk| on_stream(StreamTarget::Answer, StreamEvent::Chunk, chunk),
                    )?;
                    on_stream(StreamTarget::Answer, StreamEvent::End, "");
                    Ok(out)
                })?;

            return Ok(AgentResult {
                plan,
                answer,
                plan_attempts,
                answer_attempts,
                context_breakdown: ContextBreakdown {
                    system_tokens: estimate_tokens(&self.config.answer_system),
                    history_user_tokens,
                    history_assistant_tokens,
                    task_tokens: estimate_tokens(task),
                    context_chunks_tokens: context.map(estimate_tokens).unwrap_or(0),
                    tool_log_tokens: 0,
                },
            });
        }

        let mut attempts: Vec<Attempt> = Vec::new();
        let mut tool_context = String::new();
        let mut final_hint: Option<String> = None;
        let required_tools = required_tools_from_task(task, tool_names);
        let mut remaining_required: HashSet<String> = required_tools.iter().cloned().collect();
        if !required_tools.is_empty() {
            tool_context.push_str("REQUIRED_TOOLS:\n");
            for tool in &required_tools {
                tool_context.push_str("- ");
                tool_context.push_str(tool);
                tool_context.push('\n');
            }
            tool_context.push('\n');
        }

        for _ in 0..self.config.max_tool_iters.max(1) {
            let messages = build_action_messages(
                task,
                context,
                &plan,
                &tool_context,
                tool_names,
                tool_schema_text,
                &attempts,
                &self.config,
                conv_history,
                user_turn_only,
            );
            on_stream(StreamTarget::Tool, StreamEvent::Start, "");
            let action_out = self.llm.generate_messages_stream(
                &messages,
                self.config.max_action_tokens,
                &mut |chunk| on_stream(StreamTarget::Tool, StreamEvent::Chunk, chunk),
            )?;
            on_stream(StreamTarget::Tool, StreamEvent::End, "");

            if let Some(msg) = first_constraint_error(&self.config.action_constraints, &action_out) {
                attempts.push(Attempt { output: action_out, error_msg: msg.clone() });
                tool_context.push_str("\nACTION_FAILED:\n");
                tool_context.push_str(&msg);
                tool_context.push('\n');
                continue;
            }

            match parse_action(&action_out) {
                Ok(AgentAction::Final(answer)) => {
                    if !remaining_required.is_empty() {
                        let mut remaining: Vec<String> = remaining_required.iter().cloned().collect();
                        remaining.sort();
                        let msg = format!("must call required tools before FINAL: {}", remaining.join(", "));
                        attempts.push(Attempt { output: action_out, error_msg: msg.clone() });
                        tool_context.push_str("\nTOOL_ERROR:\n");
                        tool_context.push_str(&msg);
                        tool_context.push('\n');
                        continue;
                    }
                    if let Some(reminder) = on_final_check() {
                        tool_context.push_str("\nSYSTEM_REMINDER:\n");
                        tool_context.push_str(&reminder);
                        tool_context.push('\n');
                        continue;
                    }
                    final_hint = Some(answer);
                    break;
                }
                Ok(AgentAction::Tool(call)) => {
                    if !tool_names.iter().any(|name| name == &call.name) {
                        let msg = format!("unknown tool: {}. Use one of: {}", call.name, tool_names.join(", "));
                        attempts.push(Attempt { output: action_out, error_msg: msg.clone() });
                        tool_context.push_str("\nTOOL_ERROR:\n");
                        tool_context.push_str(&msg);
                        tool_context.push('\n');
                        continue;
                    }
                    remaining_required.remove(&call.name);
                    match exec_tool(&call) {
                        Ok(result) => {
                            tool_context.push_str("\nTOOL_CALL:\n");
                            tool_context.push_str(&format!("{} {}\n", call.name, call.args));
                            tool_context.push_str("TOOL_RESULT:\n");
                            tool_context.push_str(&result);
                            tool_context.push('\n');
                        }
                        Err(err) => {
                            attempts.push(Attempt { output: action_out, error_msg: err.to_string() });
                            tool_context.push_str("\nTOOL_ERROR:\n");
                            tool_context.push_str(&err.to_string());
                            tool_context.push('\n');
                        }
                    }
                }
                Err(err) => {
                    attempts.push(Attempt { output: action_out, error_msg: err.to_string() });
                }
            }
        }

        let Some(final_hint) = final_hint else {
            return Err(anyhow!("tool loop exceeded max iterations ({})", self.config.max_tool_iters));
        };

        let tool_log_tokens = estimate_tokens(&tool_context);
        let (answer, answer_attempts) =
            run_with_retry(&self.config.answer_constraints, |attempts| {
                let messages = build_answer_messages(
                    task,
                    context,
                    &plan,
                    Some(&tool_context),
                    Some(&final_hint),
                    attempts,
                    &self.config,
                    conv_history,
                    user_turn_only,
                );
                on_stream(StreamTarget::Answer, StreamEvent::Start, "");
                let out = self.llm.generate_messages_stream(
                    &messages,
                    self.config.max_answer_tokens,
                    &mut |chunk| on_stream(StreamTarget::Answer, StreamEvent::Chunk, chunk),
                )?;
                on_stream(StreamTarget::Answer, StreamEvent::End, "");
                Ok(out)
            })?;

        Ok(AgentResult {
            plan,
            answer,
            plan_attempts,
            answer_attempts,
            context_breakdown: ContextBreakdown {
                system_tokens: estimate_tokens(&self.config.answer_system),
                history_user_tokens,
                history_assistant_tokens,
                task_tokens: estimate_tokens(task),
                context_chunks_tokens: context.map(estimate_tokens).unwrap_or(0),
                tool_log_tokens,
            },
        })
    }
}

fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    if chars == 0 { 0 } else { (chars + 3) / 4 }
}

fn required_tools_from_task(task: &str, tool_names: &[String]) -> Vec<String> {
    let task_lc = task.to_lowercase();
    let mut out = Vec::new();
    for name in tool_names {
        let name_lc = name.to_lowercase();
        if task_lc.contains(&name_lc) {
            out.push(name.clone());
        }
    }

    let has_shell_hint = task_lc.contains("shell")
        || task_lc.contains("terminal")
        || task_lc.contains("bash")
        || task_lc.contains("run command")
        || task_lc.contains("execute command");
    if has_shell_hint && tool_names.iter().any(|t| t == "shell.exec") {
        if !out.iter().any(|t| t == "shell.exec") {
            out.push("shell.exec".to_string());
        }
    }

    let has_code_hint = task_lc.contains("code tool")
        || task_lc.contains("code tools")
        || task_lc.contains("code.exec")
        || task_lc.contains("monty")
        || task_lc.contains("python")
        || task_lc.contains("run code");
    if has_code_hint && tool_names.iter().any(|t| t == "code.exec") {
        if !out.iter().any(|t| t == "code.exec") {
            out.push("code.exec".to_string());
        }
    }

    out
}

fn render_retry_feedback(attempts: &[Attempt]) -> String {
    let mut out = String::new();
    if attempts.is_empty() {
        return out;
    }
    out.push_str("RETRY_FEEDBACK:\n");
    for (i, a) in attempts.iter().enumerate() {
        out.push_str(&format!("- Attempt {} output:\n{}\n", i + 1, a.output));
        out.push_str(&format!("  Error:\n{}\n", a.error_msg));
    }
    out.push('\n');
    out
}

fn build_action_prompt(
    task: &str,
    context: Option<&str>,
    plan: &str,
    tool_context: &str,
    tool_names: &[String],
    tool_schema_text: Option<&str>,
    attempts: &[Attempt],
    config: &CodingAgentConfig,
) -> Prompt {
    let mut system = config.action_system.clone();
    system.push_str(&render_tools_section(tool_names, tool_schema_text));

    let mut user = String::new();
    user.push_str("TASK:\n");
    user.push_str(task);
    user.push_str("\n\nPLAN:\n");
    user.push_str(plan.trim());
    user.push('\n');

    if let Some(ctx) = context {
        user.push_str("\nCONTEXT:\n");
        user.push_str(ctx.trim());
        user.push('\n');
    }

    if !tool_context.trim().is_empty() {
        user.push_str("\nTOOL_LOG:\n");
        user.push_str(tool_context.trim());
        user.push('\n');
    }

    let retry = render_retry_feedback(attempts);
    if !retry.is_empty() {
        user.push('\n');
        user.push_str(&retry);
    }

    user.push_str("\nINSTRUCTION:\nOutput FINAL: <answer> or a tool JSON object only.\n");

    Prompt {
        system: Some(system),
        user,
    }
}

fn parse_action(output: &str) -> Result<AgentAction> {
    let trimmed = output.trim();
    if let Some(final_idx) = trimmed.find("FINAL:") {
        let answer = trimmed[final_idx + "FINAL:".len()..].trim().to_string();
        if !answer.is_empty() {
            return Ok(AgentAction::Final(answer));
        }
    }

    if let Some(json) = extract_json_object(trimmed) {
        let value: Value = serde_json::from_str(&json)?;
        let name = value
            .get("tool")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("tool json missing 'tool' field"))?
            .to_string();
        let args = value.get("args").cloned().unwrap_or(Value::Null);
        return Ok(AgentAction::Tool(ToolCallRequest { name, args }));
    }

    Err(anyhow!("output did not contain FINAL or tool json"))
}

fn extract_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let mut depth = 0;
    for (i, ch) in text[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let end = start + i + 1;
                    return Some(text[start..end].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn first_constraint_error(constraints: &[Constraint], output: &str) -> Option<String> {
    for c in constraints {
        let ok = (c.check)(output);
        if !ok && matches!(c.kind, crate::pipeline::assertions::ConstraintKind::Assert) {
            return Some(c.message.clone());
        }
    }
    for c in constraints {
        let ok = (c.check)(output);
        if !ok && matches!(c.kind, crate::pipeline::assertions::ConstraintKind::Suggest) {
            return Some(c.message.clone());
        }
    }
    None
}

fn build_plan_messages(
    task: &str,
    context: Option<&str>,
    attempts: &[Attempt],
    config: &CodingAgentConfig,
    conv_history: &[ConvMessage],
    user_turn_only: bool,
) -> Vec<ConvMessage> {
    let mut messages = vec![ConvMessage {
        role: "system".to_string(),
        content: config.plan_system.clone(),
    }];
    for msg in conv_history {
        if user_turn_only && msg.role == "assistant" {
            continue;
        }
        messages.push(msg.clone());
    }
    let prompt = build_plan_prompt(task, context, attempts, config);
    messages.push(ConvMessage { role: "user".to_string(), content: prompt.user });
    messages
}

fn build_action_messages(
    task: &str,
    context: Option<&str>,
    plan: &str,
    tool_context: &str,
    tool_names: &[String],
    tool_schema_text: Option<&str>,
    attempts: &[Attempt],
    config: &CodingAgentConfig,
    conv_history: &[ConvMessage],
    user_turn_only: bool,
) -> Vec<ConvMessage> {
    let action_prompt = build_action_prompt(
        task, context, plan, tool_context, tool_names, tool_schema_text, attempts, config,
    );
    let mut messages = vec![ConvMessage {
        role: "system".to_string(),
        content: action_prompt.system.unwrap_or_default(),
    }];
    for msg in conv_history {
        if user_turn_only && msg.role == "assistant" {
            continue;
        }
        messages.push(msg.clone());
    }
    messages.push(ConvMessage { role: "user".to_string(), content: action_prompt.user });
    messages
}

fn build_answer_messages(
    task: &str,
    context: Option<&str>,
    plan: &str,
    tool_context: Option<&str>,
    draft: Option<&str>,
    attempts: &[Attempt],
    config: &CodingAgentConfig,
    conv_history: &[ConvMessage],
    user_turn_only: bool,
) -> Vec<ConvMessage> {
    let mut messages = vec![ConvMessage {
        role: "system".to_string(),
        content: config.answer_system.clone(),
    }];
    for msg in conv_history {
        if user_turn_only && msg.role == "assistant" {
            continue;
        }
        messages.push(msg.clone());
    }
    let prompt = if let Some(tc) = tool_context {
        build_answer_prompt_with_tools(task, context, plan, tc, draft, attempts, config)
    } else {
        build_answer_prompt(task, context, plan, attempts, config)
    };
    messages.push(ConvMessage { role: "user".to_string(), content: prompt.user });
    messages
}

pub(crate) fn build_plan_prompt(
    task: &str,
    context: Option<&str>,
    attempts: &[Attempt],
    config: &CodingAgentConfig,
) -> Prompt {
    let mut user = String::new();
    user.push_str(&render_retry_feedback(attempts));
    user.push_str("TASK:\n");
    user.push_str(task);
    if let Some(ctx) = context {
        if !ctx.trim().is_empty() {
            user.push_str("\n\nCONTEXT:\n");
            user.push_str(ctx);
        }
    }
    user.push_str("\n\nINSTRUCTION:\nWrite a short plan with 3-6 bullets.\n");

    Prompt {
        system: Some(config.plan_system.clone()),
        user,
    }
}

pub(crate) fn build_answer_prompt(
    task: &str,
    context: Option<&str>,
    plan: &str,
    attempts: &[Attempt],
    config: &CodingAgentConfig,
) -> Prompt {
    let mut user = String::new();
    user.push_str(&render_retry_feedback(attempts));
    user.push_str("TASK:\n");
    user.push_str(task);
    user.push_str("\n\nPLAN:\n");
    user.push_str(plan);
    if let Some(ctx) = context {
        if !ctx.trim().is_empty() {
            user.push_str("\n\nCONTEXT:\n");
            user.push_str(ctx);
        }
    }
    user.push_str("\n\nINSTRUCTION:\nProduce the best next response.\n");

    Prompt {
        system: Some(config.answer_system.clone()),
        user,
    }
}

fn build_answer_prompt_with_tools(
    task: &str,
    context: Option<&str>,
    plan: &str,
    tool_context: &str,
    draft: Option<&str>,
    attempts: &[Attempt],
    config: &CodingAgentConfig,
) -> Prompt {
    let mut user = String::new();
    user.push_str(&render_retry_feedback(attempts));
    user.push_str("TASK:\n");
    user.push_str(task);
    user.push_str("\n\nPLAN:\n");
    user.push_str(plan);
    if let Some(ctx) = context {
        if !ctx.trim().is_empty() {
            user.push_str("\n\nCONTEXT:\n");
            user.push_str(ctx);
        }
    }
    if !tool_context.trim().is_empty() {
        user.push_str("\n\nTOOL_LOG:\n");
        user.push_str(tool_context.trim());
    }
    if let Some(draft) = draft {
        if !draft.trim().is_empty() {
            user.push_str("\n\nDRAFT:\n");
            user.push_str(draft.trim());
        }
    }
    user.push_str("\n\nINSTRUCTION:\nProduce the best next response.\n");

    Prompt {
        system: Some(config.answer_system.clone()),
        user,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_plan_prompt_includes_sections() {
        let config = CodingAgentConfig::default();
        let attempts = vec![Attempt {
            output: "bad".to_string(),
            error_msg: "try again".to_string(),
        }];
        let prompt = build_plan_prompt("Do X", Some("ctx"), &attempts, &config);

        let sys = prompt.system.as_deref().unwrap_or("");
        assert!(sys.contains("planning"));
        assert!(prompt.user.contains("RETRY_FEEDBACK"));
        assert!(prompt.user.contains("TASK:"));
        assert!(prompt.user.contains("CONTEXT:"));
        assert!(prompt.user.contains("INSTRUCTION:"));
    }

    #[test]
    fn build_answer_prompt_includes_plan() {
        let config = CodingAgentConfig::default();
        let prompt = build_answer_prompt("Do Y", None, "Plan here", &[], &config);

        let sys = prompt.system.as_deref().unwrap_or("");
        assert!(sys.contains("coding agent"));
        assert!(prompt.user.contains("TASK:"));
        assert!(prompt.user.contains("PLAN:"));
        assert!(prompt.user.contains("Plan here"));
    }
}
