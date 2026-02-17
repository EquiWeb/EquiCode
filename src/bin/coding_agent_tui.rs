use anyhow::{anyhow, Result};
use arboard::Clipboard;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers,
    MouseEvent, MouseEventKind,
};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::{execute, ExecutableCommand};
use ratatui::prelude::*;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use std::io;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use varctx_proto::agent::context::{build_context, resolve_candidate_chunks, ContextBuildConfig};
use varctx_proto::agent::{CodingAgent, CodingAgentConfig, StreamEvent, StreamTarget};
use varctx_proto::llm::{LlamaCppLlm, LlmConfig};
use varctx_proto::retrieval::OverlapRetriever;
use varctx_proto::store::ContextStore;
use varctx_proto::tools::{
    builtin_fs::{edit_file_at, list_dir, read_file_at, resolve_path_arg},
    builtin_shell::run_shell_exec,
    monty::run_code_exec,
    ExecMode, PolicyDecision, SafetyPolicy, SkillHost, ToolCall, ToolSpec,
};

fn main() -> Result<()> {
    let args = TuiArgs::from_env()?;

    let model_path = parse_model_path(args.model_override.clone())
        .ok_or_else(|| anyhow!("missing model path. set VARCTX_MODEL_PATH or pass --model <path>"))?;

    let mut agent_cfg = CodingAgentConfig::default();
    if let Some(tokens) = args.plan_tokens {
        agent_cfg.max_plan_tokens = tokens.max(1);
    }
    if let Some(tokens) = args.answer_tokens {
        agent_cfg.max_answer_tokens = tokens.max(1);
    }

    let mut llm_cfg = LlmConfig::from_env(model_path);
    llm_cfg.silence_logs = true;
    let cancel_flag = Arc::new(AtomicBool::new(false));
    llm_cfg.cancel_flag = Some(cancel_flag.clone());
    let model_ctx = llm_cfg.n_ctx.unwrap_or(0);

    let skills_dir = resolve_skills_dir(args.skills_dir.as_deref());
    let mut tool_names = Vec::new();
    let mut tool_specs: Vec<ToolSpec> = Vec::new();
    if let Some(dir) = skills_dir.as_ref() {
        let host = SkillHost::load(dir)?;
        tool_specs = host.tool_specs();
        tool_names = tool_specs.iter().map(|t| t.name.clone()).collect();
    }
    if !tool_names.iter().any(|t| t == "code.exec") {
        tool_names.push("code.exec".to_string());
    }
    for builtin in ["fs.list", "fs.read", "fs.edit", "shell.exec"] {
        if !tool_names.iter().any(|t| t == builtin) {
            tool_names.push(builtin.to_string());
        }
    }

    let policy = SafetyPolicy {
        mode: parse_exec_mode(args.exec_mode.clone()),
        workspace_root: std::env::current_dir()?,
        allow_network: args.allow_network,
    };

    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (evt_tx, evt_rx) = mpsc::channel();
    let worker_cfg = llm_cfg;
    let worker_agent_cfg = agent_cfg.clone();
    let worker_settings = WorkerSettings {
        skills_dir,
        policy,
        tool_timeout: Duration::from_secs(30),
    };
    let worker_handle =
        thread::spawn(move || worker_loop(cmd_rx, evt_tx, worker_cfg, worker_agent_cfg, worker_settings));

    let mut terminal = init_terminal()?;
    let mut app = App::new(args, agent_cfg, model_ctx, cancel_flag);
    app.tool_names = tool_names;
    app.tool_schema_text = format_tool_schemas(&tool_specs);
    let res = run_app(&mut terminal, &mut app, cmd_tx.clone(), evt_rx);
    restore_terminal(&mut terminal)?;

    let _ = cmd_tx.send(WorkerCommand::Shutdown);
    let _ = worker_handle.join();
    res
}

struct TuiArgs {
    model_override: Option<String>,
    store_path: Option<String>,
    vars: Vec<String>,
    top_k: Option<usize>,
    max_snippets: Option<usize>,
    snippet_chars: Option<usize>,
    plan_tokens: Option<usize>,
    answer_tokens: Option<usize>,
    preset_task: Option<String>,
    skills_dir: Option<String>,
    exec_mode: Option<String>,
    allow_network: bool,
}

impl TuiArgs {
    fn from_env() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut model_override = None;
        let mut store_path = None;
        let mut vars = Vec::new();
        let mut top_k = None;
        let mut max_snippets = None;
        let mut snippet_chars = None;
        let mut plan_tokens = None;
        let mut answer_tokens = None;
        let mut preset_task = None;
        let mut skills_dir = None;
        let mut exec_mode = None;
        let mut allow_network = false;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => model_override = args.next(),
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
                "--skills-dir" => skills_dir = args.next(),
                "--mode" => exec_mode = args.next(),
                "--allow-network" => allow_network = true,
                "--task" => preset_task = args.next(),
                _ => {}
            }
        }

        Ok(Self {
            model_override,
            store_path,
            vars,
            top_k,
            max_snippets,
            snippet_chars,
            plan_tokens,
            answer_tokens,
            preset_task,
            skills_dir,
            exec_mode,
            allow_network,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Focus {
    Input,
    Chat,
    Overlay,
    Approval,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum OverlayKind {
    Context,
    Plan,
    Tools,
    Help,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Pane {
    Chat,
    Overlay,
}

enum WorkerCommand {
    Run {
        task: String,
        context: Option<String>,
        plan_tokens: usize,
        answer_tokens: usize,
        tool_names: Vec<String>,
        tool_schema_text: String,
    },
    ToolDecision {
        id: u64,
        allow: bool,
    },
    Shutdown,
}

enum WorkerEvent {
    Stream {
        target: StreamTarget,
        event: StreamEvent,
        chunk: String,
    },
    ToolApprovalRequired {
        id: u64,
        call: ToolCall,
        decision: PolicyDecision,
    },
    ToolLog {
        line: String,
    },
    Done {
        plan: String,
        answer: String,
        cancelled: bool,
        elapsed_ms: u128,
    },
    Error(String),
}

struct WorkerSettings {
    skills_dir: Option<PathBuf>,
    policy: SafetyPolicy,
    tool_timeout: Duration,
}

impl Focus {
    fn next(self) -> Self {
        match self {
            Self::Input => Self::Chat,
            Self::Chat => Self::Overlay,
            Self::Overlay => Self::Input,
            Self::Approval => Self::Input,
        }
    }

    fn prev(self) -> Self {
        match self {
            Self::Input => Self::Overlay,
            Self::Chat => Self::Input,
            Self::Overlay => Self::Chat,
            Self::Approval => Self::Input,
        }
    }
}

struct App {
    args: TuiArgs,
    agent_cfg: CodingAgentConfig,
    focus: Focus,
    overlay: Option<OverlayKind>,
    status: String,
    task_input: String,
    context: String,
    plan: String,
    answer: String,
    tools_log: String,
    history: Vec<ChatEntry>,
    current_turn: Option<usize>,
    context_tokens_est: usize,
    last_context_tokens_est: usize,
    current_context_delta: i64,
    current_context_pct: f32,
    var_stats: String,
    model_ctx: u32,
    is_generating: bool,
    cancel_flag: Arc<AtomicBool>,
    tool_names: Vec<String>,
    tool_schema_text: String,
    scroll_chat: u16,
    scroll_overlay: u16,
    chat_width: u16,
    chat_height: u16,
    overlay_width: u16,
    overlay_height: u16,
    chat_rect: Rect,
    overlay_rect: Rect,
    input_rect: Rect,
    auto_follow_chat: bool,
    auto_follow_overlay: bool,
    clipboard: Option<Clipboard>,
    selection: Option<Selection>,
    last_mouse: Option<Position>,
    pending_tool: Option<PendingTool>,
    approval_prompt_active: bool,
    approval_prompt: String,
    pending_followup_prompt: Option<String>,
    exec_mode: ExecMode,
    spinner_idx: usize,
}

impl App {
    fn new(
        args: TuiArgs,
        agent_cfg: CodingAgentConfig,
        model_ctx: u32,
        cancel_flag: Arc<AtomicBool>,
    ) -> Self {
        let task_input = args.preset_task.clone().unwrap_or_default();
        let exec_mode = parse_exec_mode(args.exec_mode.clone());
        let mut app = Self {
            args,
            agent_cfg,
            focus: Focus::Input,
            overlay: None,
            status: "Ready".to_string(),
            task_input,
            context: "No context loaded.".to_string(),
            plan: String::new(),
            answer: String::new(),
            tools_log: String::new(),
            history: Vec::new(),
            current_turn: None,
            context_tokens_est: 0,
            last_context_tokens_est: 0,
            current_context_delta: 0,
            current_context_pct: 0.0,
            var_stats: "none".to_string(),
            model_ctx,
            is_generating: false,
            cancel_flag,
            tool_names: Vec::new(),
            tool_schema_text: String::new(),
            scroll_chat: 0,
            scroll_overlay: 0,
            chat_width: 80,
            chat_height: 1,
            overlay_width: 60,
            overlay_height: 1,
            chat_rect: Rect::default(),
            overlay_rect: Rect::default(),
            input_rect: Rect::default(),
            auto_follow_chat: true,
            auto_follow_overlay: true,
            clipboard: Clipboard::new().ok(),
            selection: None,
            last_mouse: None,
            pending_tool: None,
            approval_prompt_active: false,
            approval_prompt: String::new(),
            pending_followup_prompt: None,
            exec_mode,
            spinner_idx: 0,
        };
        app.history.push(ChatEntry {
            user: "Quick Start".to_string(),
            answer: QUICK_START_TEXT.to_string(),
            tools: String::new(),
            stats: String::new(),
        });
        app.update_history_render();
        app.scroll_chat = max_scroll(&app.answer, app.chat_width, app.chat_height);
        app
    }

    fn set_error(&mut self, err: &anyhow::Error) {
        self.status = format!("Error: {}", err);
    }

    fn update_history_render(&mut self) {
        let mut out = String::new();
        for (i, entry) in self.history.iter().enumerate() {
            if i > 0 {
                out.push_str("\n\n");
            }
            out.push_str("USER:\n");
            out.push_str(&entry.user);
            out.push_str("\n\nASSISTANT:\n");
            out.push_str(&entry.answer);
            if !entry.tools.trim().is_empty() {
                out.push_str("\n\nTOOLS:\n");
                out.push_str(entry.tools.trim_end());
            }
            if !entry.stats.trim().is_empty() {
                out.push_str("\n\n");
                out.push_str(entry.stats.trim());
            }
        }
        self.answer = out;
    }

    fn copy_selection(&mut self) {
        let Some(text) = self.selection_text() else {
            self.status = "Nothing selected".to_string();
            return;
        };
        if text.is_empty() {
            self.status = "Nothing selected".to_string();
            return;
        }
        match self.clipboard.as_mut() {
            Some(clipboard) => match clipboard.set_text(text.clone()) {
                Ok(_) => {
                    self.status = format!("Copied {} chars", text.chars().count());
                }
                Err(err) => {
                    self.status = format!("Clipboard error: {}", err);
                }
            },
            None => {
                self.status = "Clipboard unavailable".to_string();
            }
        }
    }

    fn selection_text(&self) -> Option<String> {
        let sel = self.selection.as_ref()?;
        if sel.is_empty() {
            return None;
        }
        let (text, width) = match sel.target {
            Pane::Chat => (self.answer.as_str(), self.chat_width),
            Pane::Overlay => {
                let Some(overlay_text) = overlay_text(self) else {
                    return None;
                };
                (overlay_text, self.overlay_width)
            }
        };
        Some(extract_selection_text(text, width, sel))
    }

    fn scroll_active(&mut self, delta: i16) {
        match self.focus {
            Focus::Chat => {
                self.scroll_chat = scroll_offset(
                    self.scroll_chat,
                    delta,
                    &self.answer,
                    self.chat_width,
                    self.chat_height,
                );
                self.auto_follow_chat =
                    self.scroll_chat >= max_scroll(&self.answer, self.chat_width, self.chat_height);
            }
            Focus::Overlay => {
                let text = match self.overlay {
                    Some(OverlayKind::Context) => self.context.as_str(),
                    Some(OverlayKind::Plan) => self.plan.as_str(),
                    Some(OverlayKind::Tools) => self.tools_log.as_str(),
                    Some(OverlayKind::Help) => HELP_TEXT,
                    None => return,
                };
                let next = scroll_offset(
                    self.scroll_overlay,
                    delta,
                    text,
                    self.overlay_width,
                    self.overlay_height,
                );
                self.scroll_overlay = next;
                self.auto_follow_overlay =
                    self.scroll_overlay >= max_scroll(text, self.overlay_width, self.overlay_height);
            }
            Focus::Input | Focus::Approval => {}
        }
    }
}

#[derive(Debug, Clone)]
struct ChatEntry {
    user: String,
    answer: String,
    tools: String,
    stats: String,
}

#[derive(Debug, Clone)]
struct PendingTool {
    id: u64,
    call: ToolCall,
    decision: PolicyDecision,
}

#[derive(Debug, Clone)]
struct Selection {
    target: Pane,
    start_line: usize,
    start_col: usize,
    end_line: usize,
    end_col: usize,
    active: bool,
}

impl Selection {
    fn is_empty(&self) -> bool {
        self.start_line == self.end_line && self.start_col == self.end_col
    }

    fn normalized(&self) -> (usize, usize, usize, usize) {
        if (self.start_line, self.start_col) <= (self.end_line, self.end_col) {
            (
                self.start_line,
                self.start_col,
                self.end_line,
                self.end_col,
            )
        } else {
            (
                self.end_line,
                self.end_col,
                self.start_line,
                self.start_col,
            )
        }
    }
}

fn scroll_offset(current: u16, delta: i16, text: &str, width: u16, height: u16) -> u16 {
    let lines = wrapped_line_count(text, width).max(1);
    let visible = height.max(1) as i16;
    let max_scroll = (lines as i16).saturating_sub(visible).max(0);
    let mut next = current as i16 + delta;
    if next < 0 {
        next = 0;
    }
    if next > max_scroll {
        next = max_scroll;
    }
    next as u16
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    cmd_tx: Sender<WorkerCommand>,
    evt_rx: Receiver<WorkerEvent>,
) -> Result<()> {
    let mut last_draw = Instant::now();
    loop {
        let mut needs_draw = false;
        while let Ok(evt) = evt_rx.try_recv() {
            handle_worker_event(app, evt)?;
            needs_draw = true;
        }
        if needs_draw || last_draw.elapsed() >= Duration::from_millis(100) {
            terminal.draw(|f| draw_ui(f, app))?;
            last_draw = Instant::now();
        }

        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) => {
                    if handle_key(terminal, app, key, &cmd_tx)? {
                        return Ok(());
                    }
                }
                Event::Mouse(me) => {
                    if handle_mouse(app, me) {
                        terminal.draw(|f| draw_ui(f, app))?;
                    }
                }
                Event::Resize(_, _) => {
                    terminal.draw(|f| draw_ui(f, app))?;
                }
                Event::FocusGained | Event::FocusLost | Event::Paste(_) => {}
            }
        }
    }
}

const HELP_TEXT: &str = "\
Commands:
/context  Toggle context overlay
/plan     Toggle plan overlay
/tools    Toggle tools overlay
/help     Toggle this help
/close    Close overlay

Keys:
Tab / Shift+Tab  Cycle focus
Esc             Clear selection / close overlay / exit
Ctrl+C          Cancel run (or quit)

Mouse:
Drag to select text, release to copy

Tool approvals:
a / Enter = approve
n / Backspace = decline
e = deny with prompt
";

const QUICK_START_TEXT: &str = "\
Welcome to EquiCode.

Quick start:
- Type a task and press Enter.
- /context, /plan, /tools to open overlays.
- Tab switches focus, Esc closes overlays.

Built-in tools:
- fs.list, fs.read, fs.edit (read before edit)
- shell.exec (requires approval)
- code.exec (Monty, requires approval)
";

fn format_tool_schemas(specs: &[ToolSpec]) -> String {
    if specs.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for spec in specs {
        let schema = serde_json::to_string(&spec.input_schema).unwrap_or_else(|_| "{}".to_string());
        out.push_str("- ");
        out.push_str(&spec.name);
        out.push(' ');
        out.push_str(&schema);
        out.push('\n');
    }
    out
}

fn overlay_title(kind: OverlayKind) -> &'static str {
    match kind {
        OverlayKind::Context => "Context",
        OverlayKind::Plan => "Plan",
        OverlayKind::Tools => "Tools",
        OverlayKind::Help => "Help",
    }
}

fn overlay_text(app: &App) -> Option<&str> {
    match app.overlay {
        Some(OverlayKind::Context) => Some(&app.context),
        Some(OverlayKind::Plan) => Some(&app.plan),
        Some(OverlayKind::Tools) => Some(&app.tools_log),
        Some(OverlayKind::Help) => Some(HELP_TEXT),
        None => None,
    }
}

fn toggle_overlay(app: &mut App, kind: OverlayKind) {
    if app.overlay == Some(kind) {
        app.overlay = None;
        if app.focus == Focus::Overlay {
            app.focus = Focus::Chat;
        }
        app.status = format!("{} closed", overlay_title(kind));
    } else {
        app.overlay = Some(kind);
        app.focus = Focus::Overlay;
        app.scroll_overlay = 0;
        app.auto_follow_overlay = true;
        app.status = format!("{} open", overlay_title(kind));
    }
    app.selection = None;
}

fn handle_command(app: &mut App, input: &str) -> bool {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return false;
    }
    let cmd = trimmed.trim_start_matches('/').split_whitespace().next().unwrap_or("");
    match cmd {
        "context" => toggle_overlay(app, OverlayKind::Context),
        "plan" => toggle_overlay(app, OverlayKind::Plan),
        "tools" => toggle_overlay(app, OverlayKind::Tools),
        "help" | "?" => toggle_overlay(app, OverlayKind::Help),
        "close" | "hide" => {
            app.overlay = None;
            if app.focus == Focus::Overlay {
                app.focus = Focus::Chat;
            }
            app.status = "Overlay closed".to_string();
        }
        "clear" => {
            app.history.clear();
            app.answer.clear();
            app.status = "Chat cleared".to_string();
        }
        _ => {
            app.status = format!("Unknown command: {}", cmd);
        }
    }
    true
}

fn handle_key(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &Sender<WorkerCommand>,
) -> Result<bool> {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        if app.is_generating {
            app.cancel_flag.store(true, Ordering::SeqCst);
            app.status = "Cancelling...".to_string();
            return Ok(false);
        }
        return Ok(true);
    }

    if let Some(pending) = app.pending_tool.clone() {
        if app.approval_prompt_active {
            match key.code {
                KeyCode::Enter => {
                    if app.approval_prompt.trim().is_empty() {
                        app.status = "Enter a prompt or Esc to cancel".to_string();
                        return Ok(false);
                    }
                    cmd_tx.send(WorkerCommand::ToolDecision {
                        id: pending.id,
                        allow: false,
                    })?;
                    app.pending_followup_prompt =
                        Some(app.approval_prompt.trim().to_string());
                    app.approval_prompt.clear();
                    app.approval_prompt_active = false;
                    app.pending_tool = None;
                    app.focus = Focus::Input;
                    app.status = "Tool denied. Prompt ready.".to_string();
                    return Ok(false);
                }
                KeyCode::Esc => {
                    app.approval_prompt_active = false;
                    app.approval_prompt.clear();
                    app.status = "Tool approval required".to_string();
                    return Ok(false);
                }
                KeyCode::Backspace => {
                    app.approval_prompt.pop();
                    return Ok(false);
                }
                KeyCode::Char(c) => {
                    if !key.modifiers.contains(KeyModifiers::CONTROL) {
                        app.approval_prompt.push(c);
                    }
                    return Ok(false);
                }
                _ => return Ok(false),
            }
        } else {
            match key.code {
                KeyCode::Enter | KeyCode::Char('a') => {
                    cmd_tx.send(WorkerCommand::ToolDecision {
                        id: pending.id,
                        allow: true,
                    })?;
                    app.status = "Tool approved".to_string();
                    app.pending_tool = None;
                    app.focus = Focus::Chat;
                    return Ok(false);
                }
                KeyCode::Char('n') | KeyCode::Backspace => {
                    cmd_tx.send(WorkerCommand::ToolDecision {
                        id: pending.id,
                        allow: false,
                    })?;
                    app.status = "Tool rejected".to_string();
                    app.pending_tool = None;
                    app.focus = Focus::Input;
                    return Ok(false);
                }
                KeyCode::Char('e') => {
                    app.approval_prompt_active = true;
                    app.approval_prompt.clear();
                    app.focus = Focus::Approval;
                    app.status = "Enter denial prompt (Enter to send, Esc to cancel)".to_string();
                    return Ok(false);
                }
                _ => return Ok(false),
            }
        }
    }

    match key.code {
        KeyCode::Esc => {
            if app.selection.is_some() {
                app.selection = None;
                app.status = "Selection cleared".to_string();
                return Ok(false);
            }
            if app.overlay.is_some() {
                app.overlay = None;
                if app.focus == Focus::Overlay {
                    app.focus = Focus::Chat;
                }
                app.scroll_overlay = 0;
                app.auto_follow_overlay = true;
                app.status = "Overlay closed".to_string();
                return Ok(false);
            }
            if app.focus == Focus::Input && !app.task_input.is_empty() {
                app.task_input.clear();
            } else {
                return Ok(true);
            }
        }
        KeyCode::Char('q') if app.focus != Focus::Input => return Ok(true),
        KeyCode::Tab => {
            if app.overlay.is_some() {
                app.focus = app.focus.next();
            } else {
                app.focus = if app.focus == Focus::Input {
                    Focus::Chat
                } else {
                    Focus::Input
                };
            }
            app.selection = None;
        }
        KeyCode::BackTab => {
            if app.overlay.is_some() {
                app.focus = app.focus.prev();
            } else {
                app.focus = if app.focus == Focus::Input {
                    Focus::Chat
                } else {
                    Focus::Input
                };
            }
            app.selection = None;
        }
        KeyCode::Up => app.scroll_active(-1),
        KeyCode::Down => app.scroll_active(1),
        KeyCode::PageUp => app.scroll_active(-5),
        KeyCode::PageDown => app.scroll_active(5),
        KeyCode::Enter if app.focus == Focus::Input => {
            if app.task_input.trim().is_empty() {
                return Ok(false);
            }
            let input = app.task_input.trim().to_string();
            if handle_command(app, &input) {
                app.task_input.clear();
                return Ok(false);
            }
            if app.is_generating {
                app.status = "Busy (cancel with Ctrl+C)".to_string();
                return Ok(false);
            }
            app.cancel_flag.store(false, Ordering::SeqCst);
            app.is_generating = true;
            app.status = "Running...".to_string();
            terminal.draw(|f| draw_ui(f, app))?;
            if let Err(err) = start_run(app, cmd_tx) {
                app.set_error(&err);
            } else {
                app.status = "Running...".to_string();
            }
            app.task_input.clear();
            app.cancel_flag.store(false, Ordering::SeqCst);
        }
        KeyCode::Backspace if app.focus == Focus::Input => {
            app.task_input.pop();
        }
        KeyCode::Char('u')
            if key.modifiers.contains(KeyModifiers::CONTROL) && app.focus == Focus::Input =>
        {
            app.task_input.clear();
        }
        KeyCode::Char(c)
            if app.focus == Focus::Input
                && !key.modifiers.contains(KeyModifiers::CONTROL) =>
        {
            app.task_input.push(c);
        }
        _ => {}
    }

    Ok(false)
}

fn handle_mouse(app: &mut App, me: MouseEvent) -> bool {
    app.last_mouse = Some(Position {
        x: me.column,
        y: me.row,
    });
    let focus = focus_from_point(app, me.column, me.row);
    match me.kind {
        MouseEventKind::Down(crossterm::event::MouseButton::Left) => {
            if let Some(f) = focus {
                app.focus = f;
                if matches!(f, Focus::Chat | Focus::Overlay) {
                    let pane = if f == Focus::Chat {
                        Pane::Chat
                    } else {
                        Pane::Overlay
                    };
                    if let Some((line, col)) =
                        selection_pos_for_pane(app, pane, me.column, me.row)
                    {
                        app.selection = Some(Selection {
                            target: pane,
                            start_line: line,
                            start_col: col,
                            end_line: line,
                            end_col: col,
                            active: true,
                        });
                    }
                } else {
                    app.selection = None;
                }
                return true;
            }
        }
        MouseEventKind::Drag(crossterm::event::MouseButton::Left) => {
            let target = match app.selection.as_ref() {
                Some(sel) if sel.active => sel.target,
                _ => return false,
            };
            if let Some((line, col)) = selection_pos_for_pane(app, target, me.column, me.row) {
                if let Some(sel) = app.selection.as_mut() {
                    sel.end_line = line;
                    sel.end_col = col;
                    return true;
                }
            }
        }
        MouseEventKind::Up(crossterm::event::MouseButton::Left) => {
            if let Some(sel) = app.selection.as_mut() {
                sel.active = false;
                app.copy_selection();
                return true;
            }
        }
        MouseEventKind::ScrollUp => {
            if let Some(f) = focus {
                app.focus = f;
                app.scroll_active(-3);
                extend_selection_on_scroll(app);
                return true;
            }
        }
        MouseEventKind::ScrollDown => {
            if let Some(f) = focus {
                app.focus = f;
                app.scroll_active(3);
                extend_selection_on_scroll(app);
                return true;
            }
        }
        _ => {}
    }
    false
}

fn focus_from_point(app: &App, x: u16, y: u16) -> Option<Focus> {
    let p = Position { x, y };
    if app.chat_rect.contains(p) {
        return Some(Focus::Chat);
    }
    if app.overlay.is_some() && app.overlay_rect.contains(p) {
        return Some(Focus::Overlay);
    }
    if app.input_rect.contains(p) {
        return Some(Focus::Input);
    }
    None
}

fn extend_selection_on_scroll(app: &mut App) {
    let target = match app.selection.as_ref() {
        Some(sel) if sel.active => sel.target,
        _ => return,
    };
    let pos = match app.last_mouse {
        Some(p) => p,
        None => return,
    };
    if let Some((line, col)) = selection_pos_for_pane(app, target, pos.x, pos.y) {
        if let Some(sel) = app.selection.as_mut() {
            sel.end_line = line;
            sel.end_col = col;
        }
    }
}

fn inner_rect(rect: Rect) -> Rect {
    Rect {
        x: rect.x.saturating_add(1),
        y: rect.y.saturating_add(1),
        width: rect.width.saturating_sub(2),
        height: rect.height.saturating_sub(2),
    }
}

fn selection_pos_for_pane(
    app: &App,
    pane: Pane,
    x: u16,
    y: u16,
) -> Option<(usize, usize)> {
    let (rect, text, width, scroll) = match pane {
        Pane::Chat => (
            app.chat_rect,
            app.answer.as_str(),
            app.chat_width,
            app.scroll_chat,
        ),
        Pane::Overlay => {
            let text = overlay_text(app)?;
            (app.overlay_rect, text, app.overlay_width, app.scroll_overlay)
        }
    };
    let inner = inner_rect(rect);
    if x < inner.x || y < inner.y || x >= inner.x + inner.width || y >= inner.y + inner.height {
        return None;
    }
    let lines = wrap_lines(text, width);
    if lines.is_empty() {
        return Some((0, 0));
    }
    let local_line = (y - inner.y) as usize + scroll as usize;
    let line_idx = local_line.min(lines.len() - 1);
    let line_len = lines[line_idx].chars().count();
    let local_col = (x - inner.x) as usize;
    let col_idx = local_col.min(line_len);
    Some((line_idx, col_idx))
}

fn start_run(app: &mut App, cmd_tx: &Sender<WorkerCommand>) -> Result<()> {
    let task = app.task_input.trim().to_string();
    let mut context = None;

    app.plan.clear();
    app.scroll_chat = 0;
    app.auto_follow_chat = true;
    app.scroll_overlay = 0;
    app.auto_follow_overlay = true;
    app.tools_log.clear();
    app.pending_tool = None;

    if let Some(store_path) = app.args.store_path.as_deref() {
        if app.args.vars.is_empty() {
            return Err(anyhow!("--store requires --vars <V:...,...>"));
        }

        let store = ContextStore::open(store_path)?;
        let mut var_parts = Vec::new();
        for v in &app.args.vars {
            if let Some(binding) = store.get_var_binding_latest_lossy(v)? {
                let summary_tokens = estimate_tokens(&binding.summary);
                if summary_tokens > 0 {
                    var_parts.push(format!(
                        "{}({}c,~{}t)",
                        v,
                        binding.chunk_ids.len(),
                        summary_tokens
                    ));
                } else {
                    var_parts.push(format!("{}({}c)", v, binding.chunk_ids.len()));
                }
            } else {
                var_parts.push(format!("{}(missing)", v));
            }
        }
        app.var_stats = var_parts.join(", ");
        let candidates = resolve_candidate_chunks(&store, &app.args.vars)?;
        let retriever = OverlapRetriever {
            top_k: app.args.top_k.unwrap_or(8),
        };
        let retrieved = retriever.retrieve(&store, &task, &candidates)?;
        let mut cfg = ContextBuildConfig {
            max_snippets: app.args.max_snippets.unwrap_or(6),
            snippet_chars: app.args.snippet_chars.unwrap_or(800),
        };
        let mut ctx = build_context(&store, &app.args.vars, &retrieved, &cfg)?;
        if app.model_ctx > 0 {
            let budget = ((app.model_ctx as f32) * 0.8) as usize;
            let mut tokens = estimate_tokens(&ctx);
            let mut last_tokens = tokens;
            while tokens > budget {
                if cfg.max_snippets > 1 {
                    cfg.max_snippets = cfg.max_snippets.saturating_sub(1);
                } else if cfg.snippet_chars > 200 {
                    cfg.snippet_chars = cfg.snippet_chars.saturating_sub(200).max(200);
                } else {
                    break;
                }
                ctx = build_context(&store, &app.args.vars, &retrieved, &cfg)?;
                tokens = estimate_tokens(&ctx);
                if tokens >= last_tokens {
                    break;
                }
                last_tokens = tokens;
            }
            if tokens > budget {
                app.status = format!("Context over budget: {} > {}", tokens, budget);
            }
        }
        context = Some(ctx);
    } else {
        app.var_stats = "none".to_string();
    }

    if let Some(ctx) = context.as_ref() {
        app.context = ctx.clone();
        if matches!(app.overlay, Some(OverlayKind::Context)) {
            app.scroll_overlay = 0;
            app.auto_follow_overlay = true;
        }
        app.context_tokens_est = estimate_tokens(ctx);
    } else {
        app.context = "No context loaded.".to_string();
        if matches!(app.overlay, Some(OverlayKind::Context)) {
            app.scroll_overlay = 0;
            app.auto_follow_overlay = true;
        }
        app.context_tokens_est = 0;
    }

    app.current_context_delta = app.context_tokens_est as i64 - app.last_context_tokens_est as i64;
    app.current_context_pct = if app.last_context_tokens_est == 0 {
        0.0
    } else {
        (app.current_context_delta as f32 / app.last_context_tokens_est as f32) * 100.0
    };
    app.last_context_tokens_est = app.context_tokens_est;

    let entry_idx = app.history.len();
    app.history.push(ChatEntry {
        user: task.clone(),
        answer: String::new(),
        tools: String::new(),
        stats: String::new(),
    });
    app.current_turn = Some(entry_idx);
    app.update_history_render();
    app.scroll_chat = max_scroll(&app.answer, app.chat_width, app.chat_height);

    cmd_tx.send(WorkerCommand::Run {
        task,
        context,
        plan_tokens: app.agent_cfg.max_plan_tokens,
        answer_tokens: app.agent_cfg.max_answer_tokens,
        tool_names: app.tool_names.clone(),
        tool_schema_text: app.tool_schema_text.clone(),
    })?;

    Ok(())
}

fn handle_worker_event(app: &mut App, event: WorkerEvent) -> Result<()> {
    match event {
        WorkerEvent::Stream {
            target,
            event,
            chunk,
        } => {
            if matches!(event, StreamEvent::Chunk) {
                app.spinner_idx = app.spinner_idx.wrapping_add(1);
            }
            match (target, event) {
                (StreamTarget::Plan, StreamEvent::Start) => {
                    app.plan.clear();
                    if matches!(app.overlay, Some(OverlayKind::Plan)) {
                        app.scroll_overlay = 0;
                        app.auto_follow_overlay = true;
                    }
                }
                (StreamTarget::Answer, StreamEvent::Start) => {
                    app.auto_follow_chat = true;
                }
                (StreamTarget::Tool, StreamEvent::Start) => {
                    if matches!(app.overlay, Some(OverlayKind::Tools)) {
                        app.scroll_overlay = 0;
                        app.auto_follow_overlay = true;
                    }
                }
                (StreamTarget::Plan, StreamEvent::Chunk) => {
                    let at_bottom = app.scroll_overlay
                        >= max_scroll(&app.plan, app.overlay_width, app.overlay_height);
                    app.plan.push_str(&chunk);
                    if matches!(app.overlay, Some(OverlayKind::Plan))
                        && (app.auto_follow_overlay || at_bottom)
                    {
                        app.scroll_overlay =
                            max_scroll(&app.plan, app.overlay_width, app.overlay_height);
                        app.auto_follow_overlay = true;
                    }
                }
                (StreamTarget::Tool, StreamEvent::Chunk) => {
                    let at_bottom =
                        app.scroll_overlay
                            >= max_scroll(&app.tools_log, app.overlay_width, app.overlay_height);
                    app.tools_log.push_str(&chunk);
                    if matches!(app.overlay, Some(OverlayKind::Tools))
                        && (app.auto_follow_overlay || at_bottom)
                    {
                        app.scroll_overlay =
                            max_scroll(&app.tools_log, app.overlay_width, app.overlay_height);
                        app.auto_follow_overlay = true;
                    }
                }
                (StreamTarget::Answer, StreamEvent::Chunk) => {
                    let idx = match app.current_turn {
                        Some(i) => i,
                        None => return Ok(()),
                    };
                    let at_bottom = app.scroll_chat
                        >= max_scroll(&app.answer, app.chat_width, app.chat_height);
                    if let Some(entry) = app.history.get_mut(idx) {
                        entry.answer.push_str(&chunk);
                    }
                    app.update_history_render();
                    if app.auto_follow_chat || at_bottom {
                        app.scroll_chat =
                            max_scroll(&app.answer, app.chat_width, app.chat_height);
                        app.auto_follow_chat = true;
                    }
                }
                _ => {}
            }
        }
        WorkerEvent::ToolApprovalRequired { id, call, decision } => {
            app.pending_tool = Some(PendingTool { id, call, decision });
            app.approval_prompt_active = false;
            app.approval_prompt.clear();
            app.focus = Focus::Approval;
            app.status =
                "Tool approval required (a/Enter approve, n/Backspace decline, e deny w/prompt)"
                    .to_string();
            if let Some(pending) = app.pending_tool.as_ref() {
                let idx = match app.current_turn {
                    Some(i) => i,
                    None => return Ok(()),
                };
                let mut buf = String::new();
                buf.push_str("\nPENDING TOOL:\n");
                buf.push_str(&format!("name: {}\n", pending.call.name));
                buf.push_str(&format!("args: {}\n", pending.call.args));
                buf.push_str(&format!("risk: {:?}\n", pending.decision.risk));
                if let Some(reason) = pending.decision.reason.as_ref() {
                    buf.push_str(&format!("reason: {}\n", reason));
                }
                if let Some(entry) = app.history.get_mut(idx) {
                    entry.tools.push_str(&buf);
                }
                app.tools_log.push_str("\nPENDING TOOL:\n");
                app.tools_log.push_str(&format!("name: {}\n", pending.call.name));
                app.tools_log.push_str(&format!("args: {}\n", pending.call.args));
                app.tools_log
                    .push_str(&format!("risk: {:?}\n", pending.decision.risk));
                if let Some(reason) = pending.decision.reason.as_ref() {
                    app.tools_log.push_str(&format!("reason: {}\n", reason));
                }
                app.update_history_render();
                let chat_at_bottom =
                    app.scroll_chat >= max_scroll(&app.answer, app.chat_width, app.chat_height);
                if app.auto_follow_chat || chat_at_bottom {
                    app.scroll_chat = max_scroll(&app.answer, app.chat_width, app.chat_height);
                    app.auto_follow_chat = true;
                }
                if matches!(app.overlay, Some(OverlayKind::Tools)) {
                    app.scroll_overlay = max_scroll(
                        &app.tools_log,
                        app.overlay_width,
                        app.overlay_height,
                    );
                    app.auto_follow_overlay = true;
                }
            }
        }
        WorkerEvent::ToolLog { line } => {
            app.spinner_idx = app.spinner_idx.wrapping_add(1);
            let idx = match app.current_turn {
                Some(i) => i,
                None => return Ok(()),
            };
            let at_bottom = app.scroll_overlay
                >= max_scroll(&app.tools_log, app.overlay_width, app.overlay_height);
            if let Some(entry) = app.history.get_mut(idx) {
                entry.tools.push_str(&line);
            }
            app.tools_log.push_str(&line);
            app.update_history_render();
            if matches!(app.overlay, Some(OverlayKind::Tools)) && (app.auto_follow_overlay || at_bottom) {
                app.scroll_overlay =
                    max_scroll(&app.tools_log, app.overlay_width, app.overlay_height);
                app.auto_follow_overlay = true;
            }
            let chat_at_bottom =
                app.scroll_chat >= max_scroll(&app.answer, app.chat_width, app.chat_height);
            if app.auto_follow_chat || chat_at_bottom {
                app.scroll_chat = max_scroll(&app.answer, app.chat_width, app.chat_height);
                app.auto_follow_chat = true;
            }
        }
        WorkerEvent::Done {
            plan,
            answer,
            cancelled,
            elapsed_ms,
        } => {
            app.plan = plan.trim().to_string();
            if matches!(app.overlay, Some(OverlayKind::Plan)) {
                app.scroll_overlay =
                    max_scroll(&app.plan, app.overlay_width, app.overlay_height);
                app.auto_follow_overlay = true;
            }
            if let Some(idx) = app.current_turn.take() {
                if let Some(entry) = app.history.get_mut(idx) {
                    entry.answer = answer.trim().to_string();
                    let elapsed = (elapsed_ms as f32 / 1000.0).max(0.001);
                    let answer_tokens = estimate_tokens(&entry.answer);
                    let tok_per_sec = answer_tokens as f32 / elapsed;
                    entry.stats = format!(
                        "stats: {:.1} tok/s | ctx {} tok ({:+} tok, {:+.1}%)",
                        tok_per_sec,
                        app.context_tokens_est,
                        app.current_context_delta,
                        app.current_context_pct
                    );
                    if app.model_ctx > 0 {
                        entry.stats.push_str(&format!(" | model ctx {}", app.model_ctx));
                    }
                    if cancelled {
                        entry.stats.push_str(" | cancelled");
                    }
                }
            }
            app.update_history_render();
            if app.auto_follow_chat {
                app.scroll_chat = max_scroll(&app.answer, app.chat_width, app.chat_height);
            }
            app.is_generating = false;
            app.spinner_idx = 0;
            if cancelled {
                app.status = "Cancelled".to_string();
            } else {
                app.status = "Done".to_string();
            }
            app.pending_tool = None;
            if app.focus == Focus::Approval {
                app.focus = Focus::Input;
            }
            if let Some(prompt) = app.pending_followup_prompt.take() {
                app.task_input = prompt;
                app.focus = Focus::Input;
            }
        }
        WorkerEvent::Error(msg) => {
            app.is_generating = false;
            app.spinner_idx = 0;
            app.status = format!("Error: {}", msg);
            app.pending_tool = None;
            if app.focus == Focus::Approval {
                app.focus = Focus::Input;
            }
            if let Some(prompt) = app.pending_followup_prompt.take() {
                app.task_input = prompt;
                app.focus = Focus::Input;
            }
        }
    }
    Ok(())
}

fn worker_loop(
    cmd_rx: Receiver<WorkerCommand>,
    evt_tx: Sender<WorkerEvent>,
    llm_cfg: LlmConfig,
    base_agent_cfg: CodingAgentConfig,
    settings: WorkerSettings,
) {
    let cancel_flag = llm_cfg.cancel_flag.clone();
    let mut llm = match LlamaCppLlm::load(llm_cfg) {
        Ok(v) => v,
        Err(err) => {
            let _ = evt_tx.send(WorkerEvent::Error(err.to_string()));
            return;
        }
    };
    let host = match settings.skills_dir.as_ref() {
        Some(dir) => match SkillHost::load(dir) {
            Ok(h) => Some(h),
            Err(err) => {
                let _ = evt_tx.send(WorkerEvent::Error(err.to_string()));
                return;
            }
        },
        None => None,
    };
    let mut tool_id: u64 = 1;

    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            WorkerCommand::Run {
                task,
                context,
                plan_tokens,
                answer_tokens,
                tool_names,
                tool_schema_text,
            } => {
                let mut read_files = std::collections::HashSet::<PathBuf>::new();
                let mut agent_cfg = base_agent_cfg.clone();
                agent_cfg.max_plan_tokens = plan_tokens.max(1);
                agent_cfg.max_answer_tokens = answer_tokens.max(1);
                let mut agent = CodingAgent::new(&mut llm, agent_cfg);
                let start = Instant::now();
                let mut on_stream = |target: StreamTarget, event: StreamEvent, chunk: &str| {
                    let _ = evt_tx.send(WorkerEvent::Stream {
                        target,
                        event,
                        chunk: chunk.to_string(),
                    });
                };
                let mut exec_tool = |call_req: &varctx_proto::agent::ToolCallRequest| -> Result<String> {
                    let id = tool_id;
                    tool_id = tool_id.saturating_add(1);
                    let call = ToolCall {
                        id,
                        name: call_req.name.clone(),
                        args: call_req.args.clone(),
                    };
                    let decision = settings.policy.classify(&call);
                    if !decision.allowed {
                        let msg = decision
                            .reason
                            .clone()
                            .unwrap_or_else(|| "tool denied".to_string());
                        let _ = evt_tx.send(WorkerEvent::ToolLog {
                            line: format!("\nTOOL_DENIED: {} ({})\n", call.name, msg),
                        });
                        return Err(anyhow!(msg));
                    }
                    if decision.needs_approval {
                        let _ = evt_tx.send(WorkerEvent::ToolApprovalRequired {
                            id,
                            call: call.clone(),
                            decision: decision.clone(),
                        });
                        let approved = wait_for_tool_decision(&cmd_rx, id)?;
                        if !approved {
                            let _ = evt_tx.send(WorkerEvent::ToolLog {
                                line: format!("\nTOOL_REJECTED: {}\n", call.name),
                            });
                            return Err(anyhow!("tool rejected by user"));
                        }
                    }
                    let _ = evt_tx.send(WorkerEvent::ToolLog {
                        line: format!("\nTOOL_RUN: {} {}\n", call.name, call.args),
                    });
                    if call.name == "code.exec" {
                        let out = run_code_exec(&call.args, &settings.policy)?;
                        let _ = evt_tx.send(WorkerEvent::ToolLog {
                            line: format!("\nTOOL_RESULT: {}\n", out.trim_end()),
                        });
                        return Ok(out.trim().to_string());
                    }
                    if call.name == "shell.exec" {
                        let out = run_shell_exec(&call.args, &settings.policy.workspace_root)?;
                        let _ = evt_tx.send(WorkerEvent::ToolLog {
                            line: format!("\nTOOL_RESULT: {}\n", out.trim_end()),
                        });
                        return Ok(out.trim().to_string());
                    }
                    if call.name == "fs.list" {
                        let out = list_dir(&call.args, &settings.policy.workspace_root)?;
                        let _ = evt_tx.send(WorkerEvent::ToolLog {
                            line: format!("\nTOOL_RESULT: {}\n", out.trim_end()),
                        });
                        return Ok(out.trim().to_string());
                    }
                    if call.name == "fs.read" {
                        let path = resolve_path_arg(&call.args, &settings.policy.workspace_root)?;
                        let out = read_file_at(&path, &call.args)?;
                        read_files.insert(path);
                        let _ = evt_tx.send(WorkerEvent::ToolLog {
                            line: format!("\nTOOL_RESULT: {}\n", out.trim_end()),
                        });
                        return Ok(out.trim().to_string());
                    }
                    if call.name == "fs.edit" {
                        let path = resolve_path_arg(&call.args, &settings.policy.workspace_root)?;
                        if !read_files.contains(&path) {
                            return Err(anyhow!(
                                "must read file before edit: {}",
                                path.display()
                            ));
                        }
                        let out = edit_file_at(&path, &call.args)?;
                        let _ = evt_tx.send(WorkerEvent::ToolLog {
                            line: format!("\nTOOL_RESULT: {}\n", out.trim_end()),
                        });
                        return Ok(out.trim().to_string());
                    }
                    let host = host.as_ref().ok_or_else(|| anyhow!("no skills loaded"))?;
                    let result = host.run_tool(
                        &call,
                        settings.tool_timeout,
                        &settings.policy.workspace_root,
                    )?;
                    let mut out = String::new();
                    if !result.stdout.trim().is_empty() {
                        out.push_str(&result.stdout);
                    }
                    if !result.stderr.trim().is_empty() {
                        out.push_str("\nSTDERR:\n");
                        out.push_str(&result.stderr);
                    }
                    if let Some(err) = result.error.as_ref() {
                        out.push_str("\nERROR:\n");
                        out.push_str(err);
                    }
                    let _ = evt_tx.send(WorkerEvent::ToolLog {
                        line: format!("\nTOOL_RESULT: {}\n", out.trim_end()),
                    });
                    Ok(out.trim().to_string())
                };
                match agent.run_with_tools_streaming(
                    &task,
                    context.as_deref(),
                    &tool_names,
                    Some(tool_schema_text.as_str()),
                    &mut exec_tool,
                    &mut on_stream,
                ) {
                    Ok(result) => {
                        let cancelled = cancel_flag
                            .as_ref()
                            .map(|f| f.load(Ordering::SeqCst))
                            .unwrap_or(false);
                        let _ = evt_tx.send(WorkerEvent::Done {
                            plan: result.plan,
                            answer: result.answer,
                            cancelled,
                            elapsed_ms: start.elapsed().as_millis(),
                        });
                    }
                    Err(err) => {
                        let _ = evt_tx.send(WorkerEvent::Error(err.to_string()));
                    }
                }
            }
            WorkerCommand::ToolDecision { .. } => {
                // Tool approvals are consumed by wait_for_tool_decision.
            }
            WorkerCommand::Shutdown => {
                break;
            }
        }
    }
}

fn wait_for_tool_decision(cmd_rx: &Receiver<WorkerCommand>, id: u64) -> Result<bool> {
    loop {
        match cmd_rx.recv()? {
            WorkerCommand::ToolDecision { id: resp_id, allow } => {
                if resp_id == id {
                    return Ok(allow);
                }
            }
            WorkerCommand::Shutdown => {
                return Err(anyhow!("shutdown"));
            }
            WorkerCommand::Run { .. } => {
                // Ignore nested runs while waiting for approval.
            }
        }
    }
}

fn draw_ui(f: &mut Frame, app: &mut App) {
    let size = f.area();
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Length(2), Constraint::Min(5), Constraint::Length(3)])
        .split(size);

    let header = Paragraph::new(header_text(app))
        .block(Block::default().borders(Borders::BOTTOM));
    f.render_widget(header, outer[0]);

    let body = outer[1];
    let (chat_rect, overlay_rect) = if app.overlay.is_some() {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(body);
        (cols[0], cols[1])
    } else {
        (body, Rect::default())
    };

    app.chat_rect = chat_rect;
    let chat_block = block_with_focus("Chat", app.focus == Focus::Chat);
    let chat_inner = chat_block.inner(app.chat_rect);
    app.chat_width = chat_inner.width.max(1);
    app.chat_height = chat_inner.height.max(1);
    app.scroll_chat = clamp_scroll(app.scroll_chat, &app.answer, app.chat_width, app.chat_height);
    let chat = Paragraph::new(render_lines(
        &app.answer,
        app.chat_width,
        app.selection.as_ref().filter(|s| s.target == Pane::Chat),
    ))
    .block(chat_block)
    .scroll((app.scroll_chat, 0));
    f.render_widget(chat, app.chat_rect);

    if let Some(kind) = app.overlay {
        app.overlay_rect = overlay_rect;
        let overlay_block = block_with_focus(overlay_title(kind), app.focus == Focus::Overlay);
        let overlay_inner = overlay_block.inner(app.overlay_rect);
        app.overlay_width = overlay_inner.width.max(1);
        app.overlay_height = overlay_inner.height.max(1);
        let overlay_text = match kind {
            OverlayKind::Context => app.context.as_str(),
            OverlayKind::Plan => app.plan.as_str(),
            OverlayKind::Tools => app.tools_log.as_str(),
            OverlayKind::Help => HELP_TEXT,
        };
        app.scroll_overlay = clamp_scroll(
            app.scroll_overlay,
            overlay_text,
            app.overlay_width,
            app.overlay_height,
        );
        let overlay = Paragraph::new(render_lines(
            overlay_text,
            app.overlay_width,
            app.selection.as_ref().filter(|s| s.target == Pane::Overlay),
        ))
        .block(overlay_block)
        .scroll((app.scroll_overlay, 0));
        f.render_widget(overlay, app.overlay_rect);
    } else {
        app.overlay_rect = Rect::default();
        app.overlay_width = 1;
        app.overlay_height = 1;
        app.scroll_overlay = 0;
    }

    app.input_rect = outer[2];
    let input_block = block_with_focus("Input", app.focus == Focus::Input);
    let input = Paragraph::new(app.task_input.clone())
        .block(input_block)
        .wrap(Wrap { trim: false });
    f.render_widget(input, outer[2]);

    if app.pending_tool.is_some() {
        draw_approval_modal(f, app);
    }
}

fn block_with_focus(title: &str, focused: bool) -> Block<'_> {
    let style = if focused {
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(style)
}

fn draw_approval_modal(f: &mut Frame, app: &App) {
    let area = centered_rect(70, 50, f.area());
    f.render_widget(Clear, area);
    let title = if app.approval_prompt_active {
        "Deny With Prompt"
    } else {
        "Tool Approval"
    };
    let block = block_with_focus(title, app.focus == Focus::Approval);
    let inner = block.inner(area);
    let text = approval_modal_text(app);
    let paragraph = Paragraph::new(render_lines(&text, inner.width.max(1), None))
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(paragraph, area);
}

fn approval_modal_text(app: &App) -> String {
    let Some(pending) = app.pending_tool.as_ref() else {
        return String::new();
    };
    let mut out = String::new();
    out.push_str("Tool request:\n");
    out.push_str(&format!("name: {}\n", pending.call.name));
    out.push_str(&format!("args: {}\n", pending.call.args));
    out.push_str(&format!("risk: {:?}\n", pending.decision.risk));
    if let Some(reason) = pending.decision.reason.as_ref() {
        out.push_str(&format!("reason: {}\n", reason));
    }
    if app.approval_prompt_active {
        out.push_str("\nDeny with prompt:\n> ");
        out.push_str(&app.approval_prompt);
        out.push_str("\n\nEnter = deny + prompt, Esc = cancel");
    } else {
        out.push_str("\nApprove: a or Enter\n");
        out.push_str("Decline: n or Backspace\n");
        out.push_str("Deny with prompt: e");
    }
    out
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let percent_x = percent_x.min(100).max(10);
    let percent_y = percent_y.min(100).max(10);
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1]);
    horizontal[1]
}

fn header_text(app: &App) -> String {
    let store = app
        .args
        .store_path
        .as_deref()
        .unwrap_or("none")
        .to_string();
    let vars = if app.args.vars.is_empty() {
        "none".to_string()
    } else {
        app.var_stats.clone()
    };
    let spinner = if app.is_generating {
        const FRAMES: [&str; 4] = ["|", "/", "-", "\\"];
        FRAMES[app.spinner_idx % FRAMES.len()]
    } else {
        ""
    };
    let status = if app.is_generating && !spinner.is_empty() {
        format!("{} {}", app.status, spinner)
    } else {
        app.status.clone()
    };
    let ctx_delta = if app.current_context_delta == 0 {
        String::new()
    } else {
        format!(
            " ({:+} tok, {:+.1}%)",
            app.current_context_delta, app.current_context_pct
        )
    };
    if app.model_ctx > 0 {
        format!(
            "EquiCode TUI | Store: {} | Vars: {} | Mode: {:?} | Ctx: {}/{} tok{} | Status: {}",
            store,
            vars,
            app.exec_mode,
            app.context_tokens_est,
            app.model_ctx,
            ctx_delta,
            status
        )
    } else {
        format!(
            "EquiCode TUI | Store: {} | Vars: {} | Mode: {:?} | Ctx: {} tok{} | Status: {}",
            store, vars, app.exec_mode, app.context_tokens_est, ctx_delta, status
        )
    }
}

fn clamp_scroll(current: u16, text: &str, width: u16, height: u16) -> u16 {
    current.min(max_scroll(text, width, height))
}

fn max_scroll(text: &str, width: u16, height: u16) -> u16 {
    let lines = wrapped_line_count(text, width).max(1);
    let visible = height.max(1) as usize;
    lines.saturating_sub(visible) as u16
}

fn wrapped_line_count(text: &str, width: u16) -> usize {
    wrap_lines(text, width).len()
}

fn wrap_lines(text: &str, width: u16) -> Vec<String> {
    let w = width.max(1) as usize;
    let mut out = Vec::new();
    if text.is_empty() {
        return out;
    }
    for line in text.split('\n') {
        if line.is_empty() {
            out.push(String::new());
            continue;
        }
        let chars: Vec<char> = line.chars().collect();
        let mut start = 0;
        while start < chars.len() {
            let end = (start + w).min(chars.len());
            let slice: String = chars[start..end].iter().collect();
            out.push(slice);
            start = end;
        }
    }
    out
}

fn render_lines(text: &str, width: u16, selection: Option<&Selection>) -> Vec<Line<'static>> {
    let lines = wrap_lines(text, width);
    if lines.is_empty() {
        return vec![Line::from("")];
    }
    let Some(sel) = selection else {
        return lines.into_iter().map(Line::from).collect();
    };
    if sel.is_empty() {
        return lines.into_iter().map(Line::from).collect();
    }
    let (start_line, start_col, end_line, end_col) = sel.normalized();
    let highlight = Style::default().bg(Color::Blue).fg(Color::White);
    lines
        .into_iter()
        .enumerate()
        .map(|(idx, line)| {
            let line_len = line.chars().count();
            if idx < start_line || idx > end_line {
                return Line::from(line);
            }
            let sel_start = if idx == start_line { start_col } else { 0 };
            let sel_end = if idx == end_line { end_col } else { line_len };
            if sel_start >= sel_end || line_len == 0 {
                return Line::from(line);
            }
            let chars: Vec<char> = line.chars().collect();
            let pre: String = chars.iter().take(sel_start).collect();
            let mid: String = chars.iter().skip(sel_start).take(sel_end - sel_start).collect();
            let post: String = chars.iter().skip(sel_end).collect();
            let mut spans = Vec::new();
            if !pre.is_empty() {
                spans.push(Span::raw(pre));
            }
            if !mid.is_empty() {
                spans.push(Span::styled(mid, highlight));
            }
            if !post.is_empty() {
                spans.push(Span::raw(post));
            }
            Line::from(spans)
        })
        .collect()
}

fn extract_selection_text(text: &str, width: u16, selection: &Selection) -> String {
    let lines = wrap_lines(text, width);
    if lines.is_empty() {
        return String::new();
    }
    let (start_line, start_col, end_line, end_col) = selection.normalized();
    let mut out = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if idx < start_line || idx > end_line {
            continue;
        }
        let line_len = line.chars().count();
        let sel_start = if idx == start_line { start_col } else { 0 };
        let sel_end = if idx == end_line { end_col } else { line_len };
        if sel_start >= sel_end || line_len == 0 {
            out.push(String::new());
            continue;
        }
        let chars: Vec<char> = line.chars().collect();
        let slice: String = chars
            .iter()
            .skip(sel_start.min(line_len))
            .take(sel_end.saturating_sub(sel_start))
            .collect();
        out.push(slice);
    }
    out.join("\n")
}

fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    if chars == 0 {
        0
    } else {
        (chars + 3) / 4
    }
}

fn init_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    terminal
        .backend_mut()
        .execute(DisableMouseCapture)?
        .execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;
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

fn parse_exec_mode(mode: Option<String>) -> ExecMode {
    match mode.as_deref() {
        Some("confirm") => ExecMode::Confirm,
        Some("paranoid") => ExecMode::Paranoid,
        _ => ExecMode::Yolo,
    }
}

fn resolve_skills_dir(override_dir: Option<&str>) -> Option<PathBuf> {
    let dir = override_dir?;
    let p = PathBuf::from(dir);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}
