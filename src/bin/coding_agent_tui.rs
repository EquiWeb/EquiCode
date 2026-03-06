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
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap};
use std::io;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use varctx_proto::agent::context::{build_context, resolve_candidate_chunks, ContextBuildConfig};
use varctx_proto::agent::{CodingAgent, CodingAgentConfig, ContextBreakdown, StreamEvent, StreamTarget};
use varctx_proto::config::{Backend, Config};
use varctx_proto::llm::openrouter::{OpenRouterLlm, OpenRouterModel};
use varctx_proto::llm::{ConvMessage, LlamaCppLlm, LlmConfig, Llm, Prompt};
use varctx_proto::retrieval::OverlapRetriever;
use varctx_proto::store::ContextStore;
use varctx_proto::tools::{
    builtin_fs::{edit_file_at, list_dir, read_file_at, resolve_path_arg},
    builtin_shell::run_shell_exec,
    monty::run_code_exec,
    ExecMode, PolicyDecision, SafetyPolicy, SkillHost, ToolCall, ToolSpec,
};

// ─── Backend enum for TUI ────────────────────────────────────────────────────

enum AnyLlm {
    Local(LlamaCppLlm),
    OpenRouter(OpenRouterLlm),
}

impl Llm for AnyLlm {
    fn generate(&mut self, prompt: &Prompt, max_tokens: usize) -> Result<String> {
        match self {
            Self::Local(l) => l.generate(prompt, max_tokens),
            Self::OpenRouter(l) => l.generate(prompt, max_tokens),
        }
    }
    fn generate_stream(
        &mut self,
        prompt: &Prompt,
        max_tokens: usize,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String> {
        match self {
            Self::Local(l) => l.generate_stream(prompt, max_tokens, on_token),
            Self::OpenRouter(l) => l.generate_stream(prompt, max_tokens, on_token),
        }
    }
    fn generate_messages_stream(
        &mut self,
        messages: &[ConvMessage],
        max_tokens: usize,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String> {
        match self {
            Self::Local(l) => l.generate_messages_stream(messages, max_tokens, on_token),
            Self::OpenRouter(l) => l.generate_messages_stream(messages, max_tokens, on_token),
        }
    }
}

fn try_init_llm(config: &Config, cancel_flag: Option<Arc<AtomicBool>>) -> Option<AnyLlm> {
    match config.backend {
        Backend::Local => {
            let path = config.local_model_path.as_ref()?;
            let mut llm_cfg = LlmConfig::new(path);
            llm_cfg.n_ctx = config.n_ctx;
            llm_cfg.silence_logs = true;
            llm_cfg.cancel_flag = cancel_flag;
            LlamaCppLlm::load(llm_cfg).ok().map(AnyLlm::Local)
        }
        Backend::OpenRouter => {
            let api_key = config.effective_openrouter_key()?;
            let model = config.openrouter_model.clone().filter(|m| !m.is_empty())?;
            Some(AnyLlm::OpenRouter(OpenRouterLlm::new(api_key, model)))
        }
    }
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let config = Config::load();
    let args = TuiArgs::from_env();

    let mut agent_cfg = CodingAgentConfig::default();
    agent_cfg.max_plan_tokens = config.plan_tokens;
    agent_cfg.max_answer_tokens = config.answer_tokens;
    if let Some(tokens) = args.plan_tokens {
        agent_cfg.max_plan_tokens = tokens.max(1);
    }
    if let Some(tokens) = args.answer_tokens {
        agent_cfg.max_answer_tokens = tokens.max(1);
    }

    let cancel_flag = Arc::new(AtomicBool::new(false));

    let skills_dir = config
        .skills_dir
        .as_deref()
        .or(args.skills_dir.as_deref())
        .and_then(|d| {
            let p = PathBuf::from(d);
            if p.exists() { Some(p) } else { None }
        });

    let mut tool_names: Vec<String> = Vec::new();
    let mut tool_specs: Vec<ToolSpec> = Vec::new();
    if let Some(dir) = skills_dir.as_ref() {
        if let Ok(host) = SkillHost::load(dir) {
            tool_specs = host.tool_specs();
            tool_names = tool_specs.iter().map(|t| t.name.clone()).collect();
        }
    }
    for def in varctx_proto::agent::BUILTIN_TOOLS {
        if !tool_names.iter().any(|t| t == def.name) {
            tool_names.push(def.name.to_string());
        }
    }

    let exec_mode_str = args
        .exec_mode
        .clone()
        .unwrap_or_else(|| config.exec_mode.clone());
    let policy = SafetyPolicy {
        mode: parse_exec_mode(Some(exec_mode_str)),
        workspace_root: std::env::current_dir()?,
        allow_network: args.allow_network,
    };

    let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCommand>();
    let (evt_tx, evt_rx) = mpsc::channel::<WorkerEvent>();

    let worker_config = config.clone();
    let worker_agent_cfg = agent_cfg.clone();
    let worker_settings = WorkerSettings {
        skills_dir: skills_dir.clone(),
        policy,
        tool_timeout: Duration::from_secs(30),
        cancel_flag: Some(cancel_flag.clone()),
    };

    let worker_handle =
        thread::spawn(move || worker_loop(cmd_rx, evt_tx, worker_config, worker_agent_cfg, worker_settings));

    let model_ctx = config.n_ctx.unwrap_or(8192);
    let mut terminal = init_terminal()?;
    let mut app = App::new(config, args, agent_cfg, model_ctx, cancel_flag);
    app.tool_names = tool_names;
    app.tool_schema_text = format_tool_schemas(&tool_specs);

    if !app.config.is_configured() {
        app.overlay = Some(OverlayKind::Settings);
        app.focus = Focus::Overlay;
        load_settings_buffer(&mut app);
        app.status = "Not configured — set backend and credentials below".to_string();
    }

    let res = run_app(&mut terminal, &mut app, cmd_tx.clone(), evt_rx);
    restore_terminal(&mut terminal)?;
    let _ = cmd_tx.send(WorkerCommand::Shutdown);
    let _ = worker_handle.join();
    res
}

// ─── Args ────────────────────────────────────────────────────────────────────

struct TuiArgs {
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
    fn from_env() -> Self {
        let mut args = std::env::args().skip(1);
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
                "--store" => store_path = args.next(),
                "--vars" => {
                    if let Some(v) = args.next() {
                        vars.extend(v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()));
                    }
                }
                "--top-k" => { if let Some(v) = args.next() { top_k = v.parse().ok(); } }
                "--max-snippets" => { if let Some(v) = args.next() { max_snippets = v.parse().ok(); } }
                "--snippet-chars" => { if let Some(v) = args.next() { snippet_chars = v.parse().ok(); } }
                "--plan-tokens" => { if let Some(v) = args.next() { plan_tokens = v.parse().ok(); } }
                "--answer-tokens" => { if let Some(v) = args.next() { answer_tokens = v.parse().ok(); } }
                "--skills-dir" => skills_dir = args.next(),
                "--mode" => exec_mode = args.next(),
                "--allow-network" => allow_network = true,
                "--task" => preset_task = args.next(),
                _ => {}
            }
        }

        Self {
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
        }
    }
}

// ─── Types ───────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
enum Focus {
    Input,
    Chat,
    Overlay,
    Approval,
    Question,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum OverlayKind {
    // Panel overlays (right-side split)
    Context,
    Plan,
    Tools,
    // Modal overlays (centered floating)
    ContextViz,
    Settings,
    ModelPicker,
    SkillsManager,
    Experiments,
}

fn is_panel_overlay(kind: OverlayKind) -> bool {
    matches!(kind, OverlayKind::Context | OverlayKind::Plan | OverlayKind::Tools)
}

fn is_modal_overlay(kind: OverlayKind) -> bool {
    !is_panel_overlay(kind)
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
        user_turn_only: bool,
        disable_plan: bool,
        enforce_todos: bool,
        max_todo_interventions: Option<usize>,
    },
    ToolDecision {
        id: u64,
        allow: bool,
    },
    FetchModels,
    InstallSkill { url: String },
    Reconfigure(Config),
    ClearHistory,
    QuestionAnswer { id: u64, answer: String },
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
        context_breakdown: ContextBreakdown,
    },
    Error(String),
    LlmReady,
    ModelsReady(Vec<OpenRouterModel>),
    SkillInstalled { name: String },
    SkillInstallError(String),
    TodosUpdated(Vec<TodoItem>),
    UserQuestion { id: u64, question: String, options: Vec<String> },
}

struct WorkerSettings {
    skills_dir: Option<PathBuf>,
    policy: SafetyPolicy,
    tool_timeout: Duration,
    cancel_flag: Option<Arc<AtomicBool>>,
}

// ─── App ─────────────────────────────────────────────────────────────────────

struct App {
    config: Config,
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
    // Multi-turn history (kept in worker; tracked here for display/clear)
    conv_turn_count: usize,
    // Context breakdown for ContextViz
    context_breakdown: ContextBreakdown,
    // Settings modal state
    settings_focus: usize,
    settings_buffer: String,
    // Experiments modal state
    experiments_cursor: usize,
    experiments_number_buffer: String, // editable limit for enforce_todos
    // Model picker state
    openrouter_models: Option<Vec<OpenRouterModel>>,
    model_filter: String,
    model_selected: usize,
    // Skills manager state
    skills_url: String,
    skills_list: Vec<String>,
    skills_status: String,
    // Todo tracking (mirrored from worker)
    todos: Vec<TodoItem>,
    // Agent ask.user question
    pending_question: Option<PendingQuestion>,
    question_cursor: usize,
    question_custom: String,
    question_custom_active: bool,
    question_deadline: Option<Instant>,
}

impl App {
    fn new(
        config: Config,
        args: TuiArgs,
        agent_cfg: CodingAgentConfig,
        model_ctx: u32,
        cancel_flag: Arc<AtomicBool>,
    ) -> Self {
        let task_input = args.preset_task.clone().unwrap_or_default();
        let exec_mode_str = args.exec_mode.clone().unwrap_or_else(|| config.exec_mode.clone());
        let exec_mode = parse_exec_mode(Some(exec_mode_str));
        let experiments_number_buffer = config.experiments.max_todo_interventions
            .map(|n| n.to_string())
            .unwrap_or_default();
        let mut app = Self {
            config,
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
            conv_turn_count: 0,
            context_breakdown: ContextBreakdown::default(),
            settings_focus: 0,
            settings_buffer: String::new(),
            experiments_cursor: 0,
            experiments_number_buffer,
            openrouter_models: None,
            model_filter: String::new(),
            model_selected: 0,
            skills_url: String::new(),
            skills_list: Vec::new(),
            skills_status: String::new(),
            todos: Vec::new(),
            pending_question: None,
            question_cursor: 0,
            question_custom: String::new(),
            question_custom_active: false,
            question_deadline: None,
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
        // Marker bytes decoded by render_lines for colored styling:
        //   \x01U  → "You"       (yellow bold)
        //   \x01A  → "Assistant" (cyan bold)
        //   \x01T  → "Tools"     (magenta bold)
        //   \x01S  → dim stats
        //   \x01─  → turn divider (dark gray rule)
        let mut out = String::new();
        for (i, entry) in self.history.iter().enumerate() {
            if i > 0 {
                out.push_str("\n\x01─\n\n");
            }
            out.push_str("\x01U\n");
            out.push_str(&entry.user);
            out.push_str("\n\n\x01A\n");
            out.push_str(&entry.answer);
            if !entry.tools.trim().is_empty() {
                out.push_str("\n\n\x01T\n");
                out.push_str(entry.tools.trim_end());
            }
            if !entry.stats.trim().is_empty() {
                out.push_str("\n\n\x01S");
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
                Ok(_) => self.status = format!("Copied {} chars", text.chars().count()),
                Err(err) => self.status = format!("Clipboard error: {}", err),
            },
            None => self.status = "Clipboard unavailable".to_string(),
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
                let Some(overlay_text) = overlay_panel_text(self) else {
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
                    self.scroll_chat, delta, &self.answer, self.chat_width, self.chat_height,
                );
                self.auto_follow_chat =
                    self.scroll_chat >= max_scroll(&self.answer, self.chat_width, self.chat_height);
            }
            Focus::Overlay => {
                let text = match self.overlay {
                    Some(OverlayKind::Context) => self.context.as_str(),
                    Some(OverlayKind::Plan) => self.plan.as_str(),
                    Some(OverlayKind::Tools) => self.tools_log.as_str(),
                    _ => return, // modal overlays handle their own nav
                };
                let next = scroll_offset(self.scroll_overlay, delta, text, self.overlay_width, self.overlay_height);
                self.scroll_overlay = next;
                self.auto_follow_overlay =
                    self.scroll_overlay >= max_scroll(text, self.overlay_width, self.overlay_height);
            }
            Focus::Input | Focus::Approval | Focus::Question => {}
        }
    }
}

// ─── Structs ─────────────────────────────────────────────────────────────────

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
struct PendingQuestion {
    id: u64,
    question: String,
    options: Vec<String>,
}

#[derive(Debug, Clone)]
struct TodoItem {
    id: u64,
    task: String,
    completed: bool,
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
            (self.start_line, self.start_col, self.end_line, self.end_col)
        } else {
            (self.end_line, self.end_col, self.start_line, self.start_col)
        }
    }
}

// ─── Focus helpers ───────────────────────────────────────────────────────────

impl Focus {
    fn next(self) -> Self {
        match self {
            Self::Input => Self::Chat,
            Self::Chat => Self::Overlay,
            Self::Overlay => Self::Input,
            Self::Approval | Self::Question => Self::Input,
        }
    }
    fn prev(self) -> Self {
        match self {
            Self::Input => Self::Overlay,
            Self::Chat => Self::Input,
            Self::Overlay => Self::Chat,
            Self::Approval | Self::Question => Self::Input,
        }
    }
}

// ─── Scroll helpers ──────────────────────────────────────────────────────────

fn scroll_offset(current: u16, delta: i16, text: &str, width: u16, height: u16) -> u16 {
    let lines = wrapped_line_count(text, width).max(1);
    let visible = height.max(1) as i16;
    let max_scroll = (lines as i16).saturating_sub(visible).max(0);
    let mut next = current as i16 + delta;
    if next < 0 { next = 0; }
    if next > max_scroll { next = max_scroll; }
    next as u16
}

// ─── Event loop ──────────────────────────────────────────────────────────────

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
        // Yolo auto-answer: fire when deadline passes
        if let (Some(pq), Some(deadline)) = (app.pending_question.as_ref(), app.question_deadline) {
            if Instant::now() >= deadline {
                let answer = pq.options.first().cloned().unwrap_or_default();
                let id = pq.id;
                app.pending_question = None;
                app.question_deadline = None;
                app.focus = Focus::Chat;
                app.status = format!("Auto-selected: {}", answer);
                let _ = cmd_tx.send(WorkerCommand::QuestionAnswer { id, answer });
                needs_draw = true;
            }
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

// ─── Help & quick start text ─────────────────────────────────────────────────

const HELP_TEXT: &str = "\
Commands:
/context   Toggle context overlay
/plan      Toggle plan overlay
/tools     Toggle tools overlay
/help      Print this help to chat
/close     Close overlay
/settings  Open settings
/models    Open model picker (OpenRouter)
/skills    Open skills manager
/ctx       Context window visualization
/exp       Experiments (user-turn-only etc.)
/clear     Clear chat + history
/todos     Show TODO list in chat

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
- /settings to configure backend (OpenRouter or local llama.cpp)
- /context, /plan, /tools to open overlays.
- /models to browse OpenRouter models.
- /exp to toggle experiments (user-turn-only mode).
- Tab switches focus, Esc closes overlays.

Built-in tools:
- fs.list, fs.read, fs.edit (read before edit)
- shell.exec (requires approval)
- code.exec (Monty, requires approval)
";

// ─── Overlay helpers ─────────────────────────────────────────────────────────

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
        OverlayKind::ContextViz => "Context Visualization",
        OverlayKind::Settings => "Settings",
        OverlayKind::ModelPicker => "Model Picker",
        OverlayKind::SkillsManager => "Skills Manager",
        OverlayKind::Experiments => "Experiments",
    }
}

fn overlay_panel_text(app: &App) -> Option<&str> {
    match app.overlay {
        Some(OverlayKind::Context) => Some(&app.context),
        Some(OverlayKind::Plan) => Some(&app.plan),
        Some(OverlayKind::Tools) => Some(&app.tools_log),
        _ => None,
    }
}

fn toggle_panel_overlay(app: &mut App, kind: OverlayKind) {
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

fn open_modal_overlay(app: &mut App, kind: OverlayKind, cmd_tx: &Sender<WorkerCommand>) {
    if app.overlay == Some(kind) {
        app.overlay = None;
        app.focus = Focus::Chat;
        app.status = format!("{} closed", overlay_title(kind));
        return;
    }
    app.overlay = Some(kind);
    app.focus = Focus::Overlay;
    app.selection = None;
    app.status = format!("{} open", overlay_title(kind));
    match kind {
        OverlayKind::Settings => {
            load_settings_buffer(app);
        }
        OverlayKind::ModelPicker => {
            if app.openrouter_models.is_none() {
                let _ = cmd_tx.send(WorkerCommand::FetchModels);
                app.status = "Fetching models from OpenRouter...".to_string();
            }
            app.model_filter.clear();
            app.model_selected = 0;
        }
        OverlayKind::SkillsManager => {
            app.skills_list = list_installed_skills(&app.config);
            app.skills_status.clear();
        }
        OverlayKind::Experiments => {
            app.experiments_number_buffer = app.config.experiments.max_todo_interventions
                .map(|n| n.to_string())
                .unwrap_or_default();
        }
        _ => {}
    }
}

fn list_installed_skills(config: &Config) -> Vec<String> {
    let Some(dir) = config.skills_dir.as_ref() else { return Vec::new(); };
    let path = std::path::Path::new(dir);
    let Ok(entries) = std::fs::read_dir(path) else { return Vec::new(); };
    let mut skills: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    skills.sort();
    skills
}

// ─── Settings helpers ────────────────────────────────────────────────────────

const SETTINGS_FIELD_COUNT: usize = 9;

fn settings_field_name(idx: usize) -> &'static str {
    match idx {
        0 => "Backend",
        1 => "OpenRouter API Key",
        2 => "OpenRouter Model",
        3 => "Local Model Path",
        4 => "Skills Dir",
        5 => "Exec Mode",
        6 => "Plan Tokens",
        7 => "Answer Tokens",
        8 => "Max Hist Turns",
        _ => "",
    }
}

fn settings_field_is_enum(idx: usize) -> bool {
    matches!(idx, 0 | 5)
}

fn load_settings_buffer(app: &mut App) {
    app.settings_buffer = match app.settings_focus {
        1 => app.config.openrouter_api_key.clone().unwrap_or_default(),
        2 => app.config.openrouter_model.clone().unwrap_or_default(),
        3 => app.config.local_model_path.clone().unwrap_or_default(),
        4 => app.config.skills_dir.clone().unwrap_or_default(),
        6 => app.config.plan_tokens.to_string(),
        7 => app.config.answer_tokens.to_string(),
        8 => app.config.max_history_turns.map(|n| n.to_string()).unwrap_or_default(),
        _ => String::new(),
    };
}

fn commit_settings_buffer(app: &mut App) {
    match app.settings_focus {
        1 => {
            app.config.openrouter_api_key = if app.settings_buffer.is_empty() {
                None
            } else {
                Some(app.settings_buffer.clone())
            }
        }
        2 => {
            app.config.openrouter_model = if app.settings_buffer.is_empty() {
                None
            } else {
                Some(app.settings_buffer.clone())
            }
        }
        3 => {
            app.config.local_model_path = if app.settings_buffer.is_empty() {
                None
            } else {
                Some(app.settings_buffer.clone())
            }
        }
        4 => {
            app.config.skills_dir = if app.settings_buffer.is_empty() {
                None
            } else {
                Some(app.settings_buffer.clone())
            }
        }
        6 => {
            if let Ok(n) = app.settings_buffer.parse::<usize>() {
                app.config.plan_tokens = n;
            }
        }
        7 => {
            if let Ok(n) = app.settings_buffer.parse::<usize>() {
                app.config.answer_tokens = n;
            }
        }
        8 => {
            if app.settings_buffer.is_empty() {
                app.config.max_history_turns = None;
            } else if let Ok(n) = app.settings_buffer.parse::<usize>() {
                app.config.max_history_turns = Some(n.max(1));
            }
        }
        _ => {}
    }
}

fn settings_save(app: &mut App, cmd_tx: &Sender<WorkerCommand>) {
    commit_settings_buffer(app);
    match app.config.save() {
        Ok(_) => {
            let _ = cmd_tx.send(WorkerCommand::Reconfigure(app.config.clone()));
            app.exec_mode = parse_exec_mode(Some(app.config.exec_mode.clone()));
            app.agent_cfg.max_plan_tokens = app.config.plan_tokens;
            app.agent_cfg.max_answer_tokens = app.config.answer_tokens;
            app.overlay = None;
            app.focus = Focus::Chat;
            app.status = "Settings saved — reconfiguring backend...".to_string();
        }
        Err(err) => {
            app.status = format!("Save error: {}", err);
        }
    }
}

// ─── Slash command handling ───────────────────────────────────────────────────

fn handle_command(app: &mut App, input: &str, cmd_tx: &Sender<WorkerCommand>) -> bool {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return false;
    }
    let cmd = trimmed.trim_start_matches('/').split_whitespace().next().unwrap_or("");
    match cmd {
        "context" => toggle_panel_overlay(app, OverlayKind::Context),
        "plan" => toggle_panel_overlay(app, OverlayKind::Plan),
        "tools" => toggle_panel_overlay(app, OverlayKind::Tools),
        "help" | "?" => {
            app.history.push(ChatEntry {
                user: "/help".to_string(),
                answer: HELP_TEXT.to_string(),
                tools: String::new(),
                stats: String::new(),
            });
            app.update_history_render();
            app.scroll_chat = max_scroll(&app.answer, app.chat_width, app.chat_height);
            app.status = "Help shown in chat".to_string();
        }
        "settings" | "config" => open_modal_overlay(app, OverlayKind::Settings, cmd_tx),
        "models" | "model" => open_modal_overlay(app, OverlayKind::ModelPicker, cmd_tx),
        "skills" | "skill" => open_modal_overlay(app, OverlayKind::SkillsManager, cmd_tx),
        "ctx" | "context-viz" => open_modal_overlay(app, OverlayKind::ContextViz, cmd_tx),
        "exp" | "experiments" => open_modal_overlay(app, OverlayKind::Experiments, cmd_tx),
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
            app.conv_turn_count = 0;
            app.context_breakdown = ContextBreakdown::default();
            app.status = "Chat cleared".to_string();
            let _ = cmd_tx.send(WorkerCommand::ClearHistory);
        }
        "todos" | "todo" => {
            let mut text = String::from("TODOs this session:\n");
            if app.todos.is_empty() {
                text.push_str("  (none)\n");
            } else {
                for t in &app.todos {
                    let mark = if t.completed { "✓" } else { "○" };
                    text.push_str(&format!("  {} [{}] {}\n", mark, t.id, t.task));
                }
                let pending = app.todos.iter().filter(|t| !t.completed).count();
                text.push_str(&format!("\n{} pending, {} completed",
                    pending, app.todos.len() - pending));
            }
            app.history.push(ChatEntry {
                user: "/todos".to_string(),
                answer: text,
                tools: String::new(),
                stats: String::new(),
            });
            app.update_history_render();
            app.scroll_chat = max_scroll(&app.answer, app.chat_width, app.chat_height);
            app.status = "TODOs shown in chat".to_string();
        }
        _ => {
            app.status = format!("Unknown command: /{}", cmd);
        }
    }
    true
}

// ─── Key handling ─────────────────────────────────────────────────────────────

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

    // Tool approval handling
    if let Some(pending) = app.pending_tool.clone() {
        if app.approval_prompt_active {
            match key.code {
                KeyCode::Enter => {
                    if app.approval_prompt.trim().is_empty() {
                        app.status = "Enter a prompt or Esc to cancel".to_string();
                        return Ok(false);
                    }
                    cmd_tx.send(WorkerCommand::ToolDecision { id: pending.id, allow: false })?;
                    app.pending_followup_prompt = Some(app.approval_prompt.trim().to_string());
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
                KeyCode::Backspace => { app.approval_prompt.pop(); return Ok(false); }
                KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    app.approval_prompt.push(c);
                    return Ok(false);
                }
                _ => return Ok(false),
            }
        } else {
            match key.code {
                KeyCode::Enter | KeyCode::Char('a') => {
                    cmd_tx.send(WorkerCommand::ToolDecision { id: pending.id, allow: true })?;
                    app.status = "Tool approved".to_string();
                    app.pending_tool = None;
                    app.focus = Focus::Chat;
                    return Ok(false);
                }
                KeyCode::Char('n') | KeyCode::Backspace => {
                    cmd_tx.send(WorkerCommand::ToolDecision { id: pending.id, allow: false })?;
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

    // Agent question modal (intercepts all keys when active)
    if app.pending_question.is_some() {
        return handle_question_key(app, key, cmd_tx);
    }

    // Modal overlay key handling (intercepts all keys for modal overlays)
    if let Some(overlay) = app.overlay {
        if is_modal_overlay(overlay) {
            return handle_modal_key(app, overlay, key, cmd_tx);
        }
    }

    // Normal key handling
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
                app.focus = if app.focus == Focus::Input { Focus::Chat } else { Focus::Input };
            }
            app.selection = None;
        }
        KeyCode::BackTab => {
            if app.overlay.is_some() {
                app.focus = app.focus.prev();
            } else {
                app.focus = if app.focus == Focus::Input { Focus::Chat } else { Focus::Input };
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
            if handle_command(app, &input, cmd_tx) {
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
        KeyCode::Backspace if app.focus == Focus::Input => { app.task_input.pop(); }
        KeyCode::Char('u')
            if key.modifiers.contains(KeyModifiers::CONTROL) && app.focus == Focus::Input =>
        {
            app.task_input.clear();
        }
        KeyCode::Char(c)
            if app.focus == Focus::Input && !key.modifiers.contains(KeyModifiers::CONTROL) =>
        {
            app.task_input.push(c);
        }
        _ => {}
    }
    Ok(false)
}

// ─── Modal key handlers ───────────────────────────────────────────────────────

fn handle_modal_key(
    app: &mut App,
    overlay: OverlayKind,
    key: KeyEvent,
    cmd_tx: &Sender<WorkerCommand>,
) -> Result<bool> {
    match overlay {
        OverlayKind::Settings => handle_settings_key(app, key, cmd_tx),
        OverlayKind::Experiments => handle_experiments_key(app, key, cmd_tx),
        OverlayKind::ContextViz => {
            if key.code == KeyCode::Esc || key.code == KeyCode::Char('q') {
                app.overlay = None;
                app.focus = Focus::Chat;
            }
            Ok(false)
        }
        OverlayKind::ModelPicker => handle_model_picker_key(app, key, cmd_tx),
        OverlayKind::SkillsManager => handle_skills_manager_key(app, key, cmd_tx),
        _ => Ok(false),
    }
}

fn handle_settings_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &Sender<WorkerCommand>,
) -> Result<bool> {
    match key.code {
        KeyCode::Esc => {
            app.overlay = None;
            app.focus = Focus::Chat;
            app.status = "Settings closed (unsaved)".to_string();
        }
        KeyCode::Char('s') if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            settings_save(app, cmd_tx);
        }
        KeyCode::Up => {
            if app.settings_focus > 0 {
                commit_settings_buffer(app);
                app.settings_focus -= 1;
                load_settings_buffer(app);
            }
        }
        KeyCode::Down | KeyCode::Tab => {
            if app.settings_focus + 1 < SETTINGS_FIELD_COUNT {
                commit_settings_buffer(app);
                app.settings_focus += 1;
                load_settings_buffer(app);
            }
        }
        KeyCode::Left => {
            // Cycle enum fields backwards
            match app.settings_focus {
                0 => {
                    app.config.backend = match app.config.backend {
                        Backend::Local => Backend::OpenRouter,
                        Backend::OpenRouter => Backend::Local,
                    };
                }
                5 => {
                    app.config.exec_mode = match app.config.exec_mode.as_str() {
                        "yolo" => "paranoid".to_string(),
                        "confirm" => "yolo".to_string(),
                        "paranoid" => "confirm".to_string(),
                        _ => "yolo".to_string(),
                    };
                }
                _ => {}
            }
        }
        KeyCode::Right => {
            // Cycle enum fields forwards
            match app.settings_focus {
                0 => {
                    app.config.backend = match app.config.backend {
                        Backend::Local => Backend::OpenRouter,
                        Backend::OpenRouter => Backend::Local,
                    };
                }
                5 => {
                    app.config.exec_mode = match app.config.exec_mode.as_str() {
                        "yolo" => "confirm".to_string(),
                        "confirm" => "paranoid".to_string(),
                        "paranoid" => "yolo".to_string(),
                        _ => "yolo".to_string(),
                    };
                }
                _ => {}
            }
        }
        KeyCode::Enter => {
            if app.settings_focus == 2 {
                // Model field: open model picker
                commit_settings_buffer(app);
                open_modal_overlay(app, OverlayKind::ModelPicker, cmd_tx);
            } else {
                settings_save(app, cmd_tx);
            }
        }
        KeyCode::Backspace => {
            if !settings_field_is_enum(app.settings_focus) {
                app.settings_buffer.pop();
            }
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            if !settings_field_is_enum(app.settings_focus) {
                app.settings_buffer.push(c);
            }
        }
        _ => {}
    }
    Ok(false)
}

fn handle_experiments_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &Sender<WorkerCommand>,
) -> Result<bool> {
    const EXP_COUNT: usize = 3;
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => {
            app.overlay = None;
            app.focus = Focus::Chat;
        }
        KeyCode::Up => {
            if app.experiments_cursor > 0 {
                app.experiments_cursor -= 1;
            }
        }
        KeyCode::Down => {
            if app.experiments_cursor + 1 < EXP_COUNT {
                app.experiments_cursor += 1;
            }
        }
        KeyCode::Char(' ') => {
            match app.experiments_cursor {
                0 => app.config.experiments.user_turn_only = !app.config.experiments.user_turn_only,
                1 => app.config.experiments.disable_plan_phase = !app.config.experiments.disable_plan_phase,
                2 => app.config.experiments.enforce_todos = !app.config.experiments.enforce_todos,
                _ => {}
            }
        }
        // When cursor is on enforce_todos (item 2), let the user edit the limit inline
        KeyCode::Backspace if app.experiments_cursor == 2 => {
            app.experiments_number_buffer.pop();
        }
        KeyCode::Char(c) if app.experiments_cursor == 2 && c.is_ascii_digit()
            && !key.modifiers.contains(KeyModifiers::CONTROL) =>
        {
            app.experiments_number_buffer.push(c);
        }
        KeyCode::Char('s') => {
            // Parse and save the intervention limit
            app.config.experiments.max_todo_interventions =
                if app.experiments_number_buffer.trim().is_empty() {
                    None
                } else {
                    app.experiments_number_buffer.trim().parse::<usize>().ok().map(|n| n.max(1))
                };
            let _ = app.config.save();
            let _ = cmd_tx.send(WorkerCommand::Reconfigure(app.config.clone()));
            app.overlay = None;
            app.focus = Focus::Chat;
            app.status = "Experiments saved".to_string();
        }
        _ => {}
    }
    Ok(false)
}

fn handle_question_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &Sender<WorkerCommand>,
) -> Result<bool> {
    let Some(pq) = app.pending_question.clone() else { return Ok(false); };
    let n_options = pq.options.len().min(4);

    fn submit(app: &mut App, cmd_tx: &Sender<WorkerCommand>, id: u64, answer: String) -> Result<bool> {
        app.pending_question = None;
        app.question_deadline = None;
        app.question_custom.clear();
        app.question_custom_active = false;
        app.focus = Focus::Chat;
        app.status = format!("Answered: {}", answer);
        cmd_tx.send(WorkerCommand::QuestionAnswer { id, answer })?;
        Ok(false)
    }

    if app.question_custom_active {
        match key.code {
            KeyCode::Enter => {
                let answer = app.question_custom.trim().to_string();
                if answer.is_empty() {
                    app.question_custom_active = false;
                    return Ok(false);
                }
                return submit(app, cmd_tx, pq.id, answer);
            }
            KeyCode::Esc => {
                app.question_custom_active = false;
                app.question_custom.clear();
            }
            KeyCode::Backspace => { app.question_custom.pop(); }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.question_custom.push(c);
            }
            _ => {}
        }
        return Ok(false);
    }

    match key.code {
        KeyCode::Up => {
            if app.question_cursor > 0 { app.question_cursor -= 1; }
        }
        KeyCode::Down => {
            if app.question_cursor + 1 < n_options { app.question_cursor += 1; }
        }
        KeyCode::Char(c @ '1'..='4') => {
            let idx = (c as usize) - ('1' as usize);
            if idx < n_options {
                let answer = pq.options[idx].clone();
                return submit(app, cmd_tx, pq.id, answer);
            }
        }
        KeyCode::Enter => {
            if let Some(answer) = pq.options.get(app.question_cursor).cloned() {
                return submit(app, cmd_tx, pq.id, answer);
            }
        }
        KeyCode::Char('c') | KeyCode::Char('C') => {
            app.question_custom_active = true;
            app.question_custom.clear();
        }
        _ => {}
    }
    Ok(false)
}

fn handle_model_picker_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &Sender<WorkerCommand>,
) -> Result<bool> {
    match key.code {
        KeyCode::Esc => {
            app.overlay = None;
            app.focus = Focus::Chat;
        }
        KeyCode::Up => {
            if app.model_selected > 0 {
                app.model_selected -= 1;
            }
        }
        KeyCode::Down => {
            let filtered_count = filtered_models(app).len();
            if app.model_selected + 1 < filtered_count {
                app.model_selected += 1;
            }
        }
        KeyCode::Enter => {
            let filtered = filtered_models(app);
            if let Some(model) = filtered.get(app.model_selected) {
                let id = model.id.clone();
                let ctx_len = model.context_length;
                app.config.openrouter_model = Some(id.clone());
                if let Some(ctx) = ctx_len {
                    app.model_ctx = ctx.min(u32::MAX as u64) as u32;
                }
                let _ = app.config.save();
                // Update settings buffer if settings was open before
                app.settings_buffer = id;
                app.overlay = None;
                app.focus = Focus::Chat;
                app.status = format!("Model selected: {}", app.config.openrouter_model.as_deref().unwrap_or(""));
            }
        }
        KeyCode::Backspace => {
            app.model_filter.pop();
            app.model_selected = 0;
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.model_filter.push(c);
            app.model_selected = 0;
        }
        _ => {}
    }
    let _ = cmd_tx; // suppress unused warning
    Ok(false)
}

fn handle_skills_manager_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &Sender<WorkerCommand>,
) -> Result<bool> {
    match key.code {
        KeyCode::Esc => {
            app.overlay = None;
            app.focus = Focus::Chat;
        }
        KeyCode::Enter => {
            let url = app.skills_url.trim().to_string();
            if !url.is_empty() {
                cmd_tx.send(WorkerCommand::InstallSkill { url: url.clone() })?;
                app.skills_status = format!("Installing {}...", url);
                app.skills_url.clear();
            }
        }
        KeyCode::Backspace => { app.skills_url.pop(); }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.skills_url.push(c);
        }
        _ => {}
    }
    Ok(false)
}

fn filtered_models(app: &App) -> Vec<&OpenRouterModel> {
    let filter = app.model_filter.to_lowercase();
    match &app.openrouter_models {
        None => Vec::new(),
        Some(models) => models
            .iter()
            .filter(|m| {
                filter.is_empty()
                    || m.id.to_lowercase().contains(&filter)
                    || m.name.to_lowercase().contains(&filter)
            })
            .collect(),
    }
}

// ─── Mouse handling ───────────────────────────────────────────────────────────

fn handle_mouse(app: &mut App, me: MouseEvent) -> bool {
    app.last_mouse = Some(Position { x: me.column, y: me.row });
    let focus = focus_from_point(app, me.column, me.row);
    match me.kind {
        MouseEventKind::Down(crossterm::event::MouseButton::Left) => {
            if let Some(f) = focus {
                app.focus = f;
                if matches!(f, Focus::Chat | Focus::Overlay) {
                    let pane = if f == Focus::Chat { Pane::Chat } else { Pane::Overlay };
                    if let Some((line, col)) = selection_pos_for_pane(app, pane, me.column, me.row) {
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
    if app.chat_rect.contains(p) { return Some(Focus::Chat); }
    if app.overlay.map(is_panel_overlay).unwrap_or(false) && app.overlay_rect.contains(p) {
        return Some(Focus::Overlay);
    }
    if app.input_rect.contains(p) { return Some(Focus::Input); }
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

fn selection_pos_for_pane(app: &App, pane: Pane, x: u16, y: u16) -> Option<(usize, usize)> {
    let (rect, text, width, scroll) = match pane {
        Pane::Chat => (app.chat_rect, app.answer.as_str(), app.chat_width, app.scroll_chat),
        Pane::Overlay => {
            let text = overlay_panel_text(app)?;
            (app.overlay_rect, text, app.overlay_width, app.scroll_overlay)
        }
    };
    let inner = inner_rect(rect);
    if x < inner.x || y < inner.y || x >= inner.x + inner.width || y >= inner.y + inner.height {
        return None;
    }
    let lines = wrap_lines(text, width);
    if lines.is_empty() { return Some((0, 0)); }
    let local_line = (y - inner.y) as usize + scroll as usize;
    let line_idx = local_line.min(lines.len() - 1);
    let line_len = lines[line_idx].chars().count();
    let local_col = (x - inner.x) as usize;
    let col_idx = local_col.min(line_len);
    Some((line_idx, col_idx))
}

// ─── Run ─────────────────────────────────────────────────────────────────────

fn start_run(app: &mut App, cmd_tx: &Sender<WorkerCommand>) -> Result<()> {
    if !app.config.is_configured() {
        open_modal_overlay(app, OverlayKind::Settings, cmd_tx);
        return Err(anyhow!("Not configured — open /settings to configure backend"));
    }

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
                    var_parts.push(format!("{}({}c,~{}t)", v, binding.chunk_ids.len(), summary_tokens));
                } else {
                    var_parts.push(format!("{}({}c)", v, binding.chunk_ids.len()));
                }
            } else {
                var_parts.push(format!("{}(missing)", v));
            }
        }
        app.var_stats = var_parts.join(", ");
        let candidates = resolve_candidate_chunks(&store, &app.args.vars)?;
        let retriever = OverlapRetriever { top_k: app.args.top_k.unwrap_or(8) };
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
                if tokens >= last_tokens { break; }
                last_tokens = tokens;
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
        user_turn_only: app.config.experiments.user_turn_only,
        disable_plan: app.config.experiments.disable_plan_phase,
        enforce_todos: app.config.experiments.enforce_todos,
        max_todo_interventions: app.config.experiments.max_todo_interventions,
    })?;

    Ok(())
}

// ─── Worker event handling ────────────────────────────────────────────────────

fn handle_worker_event(app: &mut App, event: WorkerEvent) -> Result<()> {
    match event {
        WorkerEvent::Stream { target, event, chunk } => {
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
                    if matches!(app.overlay, Some(OverlayKind::Plan)) && (app.auto_follow_overlay || at_bottom) {
                        app.scroll_overlay = max_scroll(&app.plan, app.overlay_width, app.overlay_height);
                        app.auto_follow_overlay = true;
                    }
                }
                (StreamTarget::Tool, StreamEvent::Chunk) => {
                    let at_bottom = app.scroll_overlay
                        >= max_scroll(&app.tools_log, app.overlay_width, app.overlay_height);
                    app.tools_log.push_str(&chunk);
                    if matches!(app.overlay, Some(OverlayKind::Tools)) && (app.auto_follow_overlay || at_bottom) {
                        app.scroll_overlay = max_scroll(&app.tools_log, app.overlay_width, app.overlay_height);
                        app.auto_follow_overlay = true;
                    }
                }
                (StreamTarget::Answer, StreamEvent::Chunk) => {
                    let idx = match app.current_turn {
                        Some(i) => i,
                        None => return Ok(()),
                    };
                    let at_bottom = app.scroll_chat >= max_scroll(&app.answer, app.chat_width, app.chat_height);
                    if let Some(entry) = app.history.get_mut(idx) {
                        entry.answer.push_str(&chunk);
                    }
                    app.update_history_render();
                    if app.auto_follow_chat || at_bottom {
                        app.scroll_chat = max_scroll(&app.answer, app.chat_width, app.chat_height);
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
            app.status = "Tool approval required (a/Enter approve, n/Backspace decline, e deny w/prompt)".to_string();
            if let Some(pending) = app.pending_tool.as_ref() {
                let idx = match app.current_turn { Some(i) => i, None => return Ok(()) };
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
                app.tools_log.push_str(&format!("risk: {:?}\n", pending.decision.risk));
                if let Some(reason) = pending.decision.reason.as_ref() {
                    app.tools_log.push_str(&format!("reason: {}\n", reason));
                }
                app.update_history_render();
                if app.auto_follow_chat {
                    app.scroll_chat = max_scroll(&app.answer, app.chat_width, app.chat_height);
                }
                if matches!(app.overlay, Some(OverlayKind::Tools)) {
                    app.scroll_overlay = max_scroll(&app.tools_log, app.overlay_width, app.overlay_height);
                    app.auto_follow_overlay = true;
                }
            }
        }
        WorkerEvent::ToolLog { line } => {
            app.spinner_idx = app.spinner_idx.wrapping_add(1);
            let idx = match app.current_turn { Some(i) => i, None => return Ok(()) };
            let at_bottom = app.scroll_overlay >= max_scroll(&app.tools_log, app.overlay_width, app.overlay_height);
            if let Some(entry) = app.history.get_mut(idx) {
                entry.tools.push_str(&line);
            }
            app.tools_log.push_str(&line);
            app.update_history_render();
            if matches!(app.overlay, Some(OverlayKind::Tools)) && (app.auto_follow_overlay || at_bottom) {
                app.scroll_overlay = max_scroll(&app.tools_log, app.overlay_width, app.overlay_height);
                app.auto_follow_overlay = true;
            }
            if app.auto_follow_chat {
                app.scroll_chat = max_scroll(&app.answer, app.chat_width, app.chat_height);
            }
        }
        WorkerEvent::Done { plan, answer, cancelled, elapsed_ms, context_breakdown } => {
            app.plan = plan.trim().to_string();
            app.context_breakdown = context_breakdown;
            if matches!(app.overlay, Some(OverlayKind::Plan)) {
                app.scroll_overlay = max_scroll(&app.plan, app.overlay_width, app.overlay_height);
                app.auto_follow_overlay = true;
            }
            if let Some(idx) = app.current_turn.take() {
                if let Some(entry) = app.history.get_mut(idx) {
                    entry.answer = answer.trim().to_string();
                    let elapsed = (elapsed_ms as f32 / 1000.0).max(0.001);
                    let answer_tokens = estimate_tokens(&entry.answer);
                    let tok_per_sec = answer_tokens as f32 / elapsed;
                    let turns = app.conv_turn_count + 1;
                    entry.stats = format!(
                        "stats: {:.1} tok/s | ctx {} tok ({:+} tok, {:+.1}%) | turns {}",
                        tok_per_sec, app.context_tokens_est, app.current_context_delta,
                        app.current_context_pct, turns
                    );
                    if app.model_ctx > 0 {
                        entry.stats.push_str(&format!(" | model ctx {}", app.model_ctx));
                    }
                    if cancelled {
                        entry.stats.push_str(" | cancelled");
                    }
                }
            }
            app.conv_turn_count += 1;
            app.update_history_render();
            if app.auto_follow_chat {
                app.scroll_chat = max_scroll(&app.answer, app.chat_width, app.chat_height);
            }
            app.is_generating = false;
            app.spinner_idx = 0;
            app.status = if cancelled { "Cancelled".to_string() } else { "Done".to_string() };
            app.pending_tool = None;
            if app.focus == Focus::Approval { app.focus = Focus::Input; }
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
            if app.focus == Focus::Approval { app.focus = Focus::Input; }
            if let Some(prompt) = app.pending_followup_prompt.take() {
                app.task_input = prompt;
                app.focus = Focus::Input;
            }
        }
        WorkerEvent::LlmReady => {
            app.status = "Backend ready".to_string();
        }
        WorkerEvent::ModelsReady(models) => {
            app.model_selected = 0;
            app.openrouter_models = Some(models);
            app.status = format!(
                "Loaded {} models",
                app.openrouter_models.as_ref().map(|m| m.len()).unwrap_or(0)
            );
        }
        WorkerEvent::SkillInstalled { name } => {
            app.skills_list = list_installed_skills(&app.config);
            app.skills_status = format!("Installed: {}", name);
            app.status = format!("Skill installed: {}", name);
        }
        WorkerEvent::SkillInstallError(msg) => {
            app.skills_status = format!("Error: {}", msg);
            app.status = format!("Skill install error: {}", msg);
        }
        WorkerEvent::TodosUpdated(items) => {
            app.todos = items;
        }
        WorkerEvent::UserQuestion { id, question, options } => {
            let is_yolo = app.exec_mode == ExecMode::Yolo;
            app.pending_question = Some(PendingQuestion { id, question, options });
            app.question_cursor = 0;
            app.question_custom.clear();
            app.question_custom_active = false;
            app.question_deadline = if is_yolo {
                Some(Instant::now() + Duration::from_secs(20))
            } else {
                None
            };
            app.focus = Focus::Question;
            app.status = if is_yolo {
                "Agent question — answer within 20s or first option auto-selected".to_string()
            } else {
                "Agent question — select an option or type a custom answer".to_string()
            };
        }
    }
    Ok(())
}

// ─── Worker loop ──────────────────────────────────────────────────────────────

fn worker_loop(
    cmd_rx: Receiver<WorkerCommand>,
    evt_tx: Sender<WorkerEvent>,
    config: Config,
    base_agent_cfg: CodingAgentConfig,
    settings: WorkerSettings,
) {
    let cancel_flag = settings.cancel_flag.clone();
    let mut current_config = config;
    let mut settings = settings;
    let mut llm: Option<AnyLlm> = {
        let llm = try_init_llm(&current_config, cancel_flag.clone());
        if llm.is_some() {
            let _ = evt_tx.send(WorkerEvent::LlmReady);
        }
        llm
    };
    let mut host: Option<SkillHost> = settings.skills_dir.as_ref().and_then(|d| SkillHost::load(d).ok());
    let mut tool_id: u64 = 1;
    let mut conv_history: Vec<ConvMessage> = Vec::new();
    let todo_items = std::cell::RefCell::new(Vec::<TodoItem>::new());
    let mut next_todo_id: u64 = 1;

    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            WorkerCommand::Run {
                task,
                context,
                plan_tokens,
                answer_tokens,
                tool_names,
                tool_schema_text,
                user_turn_only,
                disable_plan,
                enforce_todos,
                max_todo_interventions,
            } => {
                let Some(llm) = llm.as_mut() else {
                    let _ = evt_tx.send(WorkerEvent::Error(
                        "No LLM configured. Use /settings to configure.".to_string(),
                    ));
                    continue;
                };

                let mut read_files = std::collections::HashSet::<PathBuf>::new();
                let mut agent_cfg = base_agent_cfg.clone();
                agent_cfg.max_plan_tokens = plan_tokens.max(1);
                agent_cfg.max_answer_tokens = answer_tokens.max(1);
                if disable_plan {
                    agent_cfg.max_plan_tokens = 1;
                }

                let mut agent = CodingAgent::new(llm, agent_cfg);
                let start = Instant::now();
                let task_clone = task.clone();
                let history_snapshot = conv_history.clone();

                let mut on_stream = |target: StreamTarget, event: StreamEvent, chunk: &str| {
                    let _ = evt_tx.send(WorkerEvent::Stream {
                        target,
                        event,
                        chunk: chunk.to_string(),
                    });
                };

                let evt_tx2 = evt_tx.clone();
                let policy_ref = &settings.policy;
                let timeout = settings.tool_timeout;
                let mut exec_tool = |call_req: &varctx_proto::agent::ToolCallRequest| -> Result<String> {
                    let id = tool_id;
                    tool_id = tool_id.saturating_add(1);
                    let call = ToolCall {
                        id,
                        name: call_req.name.clone(),
                        args: call_req.args.clone(),
                    };
                    let decision = policy_ref.classify(&call);
                    if !decision.allowed {
                        let msg = decision.reason.clone().unwrap_or_else(|| "tool denied".to_string());
                        let _ = evt_tx2.send(WorkerEvent::ToolLog {
                            line: format!("\nTOOL_DENIED: {} ({})\n", call.name, msg),
                        });
                        return Err(anyhow!(msg));
                    }
                    if decision.needs_approval {
                        let _ = evt_tx2.send(WorkerEvent::ToolApprovalRequired {
                            id,
                            call: call.clone(),
                            decision: decision.clone(),
                        });
                        let approved = wait_for_tool_decision(&cmd_rx, id)?;
                        if !approved {
                            let _ = evt_tx2.send(WorkerEvent::ToolLog {
                                line: format!("\nTOOL_REJECTED: {}\n", call.name),
                            });
                            return Err(anyhow!("tool rejected by user"));
                        }
                    }
                    let _ = evt_tx2.send(WorkerEvent::ToolLog {
                        line: format!("\nTOOL_RUN: {} {}\n", call.name, call.args),
                    });
                    if call.name == "ask.user" {
                        let question = call.args.get("question")
                            .and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
                        let options: Vec<String> = call.args.get("options")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect())
                            .unwrap_or_default();
                        if question.is_empty() {
                            return Err(anyhow!("ask.user requires a non-empty question"));
                        }
                        let _ = evt_tx2.send(WorkerEvent::UserQuestion {
                            id,
                            question: question.clone(),
                            options,
                        });
                        let answer = wait_for_question_answer(&cmd_rx, id)?;
                        let _ = evt_tx2.send(WorkerEvent::ToolLog {
                            line: format!("\nUSER_ANSWERED: {}\n", answer),
                        });
                        return Ok(answer);
                    }
                    if call.name == "todo.create" {
                        let task_text = call.args.get("task").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
                        if task_text.is_empty() {
                            return Err(anyhow!("todo.create requires {{\"task\": \"description\"}}"));
                        }
                        let id = next_todo_id;
                        next_todo_id += 1;
                        todo_items.borrow_mut().push(TodoItem { id, task: task_text.clone(), completed: false });
                        let snap = todo_items.borrow().clone();
                        let _ = evt_tx2.send(WorkerEvent::TodosUpdated(snap));
                        let result = format!("Created TODO #{}: {}", id, task_text);
                        let _ = evt_tx2.send(WorkerEvent::ToolLog { line: format!("\nTOOL_RESULT: {}\n", result) });
                        return Ok(result);
                    }
                    if call.name == "todo.complete" {
                        let id = call.args.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
                        let mut todos = todo_items.borrow_mut();
                        if let Some(t) = todos.iter_mut().find(|t| t.id == id) {
                            t.completed = true;
                            let task_text = t.task.clone();
                            drop(todos);
                            let snap = todo_items.borrow().clone();
                            let _ = evt_tx2.send(WorkerEvent::TodosUpdated(snap));
                            let result = format!("Completed TODO #{}: {}", id, task_text);
                            let _ = evt_tx2.send(WorkerEvent::ToolLog { line: format!("\nTOOL_RESULT: {}\n", result) });
                            return Ok(result);
                        } else {
                            return Err(anyhow!("TODO #{} not found", id));
                        }
                    }
                    if call.name == "code.exec" {
                        let out = run_code_exec(&call.args, policy_ref)?;
                        let _ = evt_tx2.send(WorkerEvent::ToolLog { line: format!("\nTOOL_RESULT: {}\n", out.trim_end()) });
                        return Ok(out.trim().to_string());
                    }
                    if call.name == "shell.exec" {
                        let out = run_shell_exec(&call.args, &policy_ref.workspace_root)?;
                        let _ = evt_tx2.send(WorkerEvent::ToolLog { line: format!("\nTOOL_RESULT: {}\n", out.trim_end()) });
                        return Ok(out.trim().to_string());
                    }
                    if call.name == "fs.list" {
                        let out = list_dir(&call.args, &policy_ref.workspace_root)?;
                        let _ = evt_tx2.send(WorkerEvent::ToolLog { line: format!("\nTOOL_RESULT: {}\n", out.trim_end()) });
                        return Ok(out.trim().to_string());
                    }
                    if call.name == "fs.read" {
                        let path = resolve_path_arg(&call.args, &policy_ref.workspace_root)?;
                        let out = read_file_at(&path, &call.args)?;
                        read_files.insert(path);
                        let _ = evt_tx2.send(WorkerEvent::ToolLog { line: format!("\nTOOL_RESULT: {}\n", out.trim_end()) });
                        return Ok(out.trim().to_string());
                    }
                    if call.name == "fs.edit" {
                        let path = resolve_path_arg(&call.args, &policy_ref.workspace_root)?;
                        if !read_files.contains(&path) {
                            return Err(anyhow!("must read file before edit: {}", path.display()));
                        }
                        let out = edit_file_at(&path, &call.args)?;
                        let _ = evt_tx2.send(WorkerEvent::ToolLog { line: format!("\nTOOL_RESULT: {}\n", out.trim_end()) });
                        return Ok(out.trim().to_string());
                    }
                    let h = host.as_ref().ok_or_else(|| anyhow!("no skills loaded"))?;
                    let result = h.run_tool(&call, timeout, &policy_ref.workspace_root)?;
                    let mut out = String::new();
                    if !result.stdout.trim().is_empty() { out.push_str(&result.stdout); }
                    if !result.stderr.trim().is_empty() {
                        out.push_str("\nSTDERR:\n");
                        out.push_str(&result.stderr);
                    }
                    if let Some(err) = result.error.as_ref() {
                        out.push_str("\nERROR:\n");
                        out.push_str(err);
                    }
                    let _ = evt_tx2.send(WorkerEvent::ToolLog { line: format!("\nTOOL_RESULT: {}\n", out.trim_end()) });
                    Ok(out.trim().to_string())
                };

                let mut todo_intervention_count: usize = 0;
                let mut on_final_check = || -> Option<String> {
                    if !enforce_todos { return None; }
                    let pending: Vec<TodoItem> = todo_items.borrow().iter().filter(|t| !t.completed).cloned().collect();
                    if pending.is_empty() { return None; }
                    if let Some(max) = max_todo_interventions {
                        if todo_intervention_count >= max { return None; }
                    }
                    todo_intervention_count += 1;
                    let mut msg = format!(
                        "You have {} pending TODO item(s) that must be completed or marked done before you finish:\n",
                        pending.len()
                    );
                    for t in &pending {
                        msg.push_str(&format!("  - [id={}] {}\n", t.id, t.task));
                    }
                    msg.push_str("\nEither complete the work described above, or call todo.complete {\"id\": <id>} to mark it done, then output FINAL.");
                    Some(msg)
                };

                let result = agent.run_with_history_streaming(
                    &task_clone,
                    context.as_deref(),
                    &history_snapshot,
                    user_turn_only,
                    &tool_names,
                    Some(tool_schema_text.as_str()),
                    &mut exec_tool,
                    &mut on_final_check,
                    &mut on_stream,
                );

                match result {
                    Ok(result) => {
                        let cancelled = cancel_flag
                            .as_ref()
                            .map(|f| f.load(Ordering::SeqCst))
                            .unwrap_or(false);
                        // Accumulate conversation history
                        conv_history.push(ConvMessage {
                            role: "user".to_string(),
                            content: task_clone,
                        });
                        conv_history.push(ConvMessage {
                            role: "assistant".to_string(),
                            content: result.answer.clone(),
                        });
                        // Enforce sliding window: drop oldest turns first
                        if let Some(max_turns) = current_config.max_history_turns {
                            let max_msgs = max_turns * 2;
                            if conv_history.len() > max_msgs {
                                conv_history.drain(0..conv_history.len() - max_msgs);
                            }
                        }
                        let _ = evt_tx.send(WorkerEvent::Done {
                            plan: result.plan,
                            answer: result.answer,
                            cancelled,
                            elapsed_ms: start.elapsed().as_millis(),
                            context_breakdown: result.context_breakdown,
                        });
                    }
                    Err(err) => {
                        let _ = evt_tx.send(WorkerEvent::Error(err.to_string()));
                    }
                }
            }
            WorkerCommand::ToolDecision { .. } => {
                // Consumed by wait_for_tool_decision
            }
            WorkerCommand::FetchModels => {
                let api_key = current_config.effective_openrouter_key().unwrap_or_default();
                match OpenRouterLlm::fetch_models(&api_key) {
                    Ok(models) => { let _ = evt_tx.send(WorkerEvent::ModelsReady(models)); }
                    Err(err) => { let _ = evt_tx.send(WorkerEvent::Error(format!("fetch models: {}", err))); }
                }
            }
            WorkerCommand::InstallSkill { url } => {
                match install_skill(&url, settings.skills_dir.as_ref()) {
                    Ok(name) => { let _ = evt_tx.send(WorkerEvent::SkillInstalled { name }); }
                    Err(err) => { let _ = evt_tx.send(WorkerEvent::SkillInstallError(err.to_string())); }
                }
            }
            WorkerCommand::Reconfigure(new_config) => {
                drop(llm.take());
                settings.policy.mode = parse_exec_mode(Some(new_config.exec_mode.clone()));
                if let Some(dir) = new_config.skills_dir.as_deref() {
                    let p = PathBuf::from(dir);
                    if p.exists() {
                        host = SkillHost::load(&p).ok();
                        settings.skills_dir = Some(p);
                    }
                }
                current_config = new_config;
                llm = try_init_llm(&current_config, cancel_flag.clone());
                if llm.is_some() {
                    let _ = evt_tx.send(WorkerEvent::LlmReady);
                } else if current_config.is_configured() {
                    let _ = evt_tx.send(WorkerEvent::Error("Failed to initialize backend".to_string()));
                }
            }
            WorkerCommand::ClearHistory => {
                conv_history.clear();
                todo_items.borrow_mut().clear();
                next_todo_id = 1;
                let _ = evt_tx.send(WorkerEvent::TodosUpdated(vec![]));
            }
            WorkerCommand::Shutdown => break,
            WorkerCommand::QuestionAnswer { .. } => {} // handled inside wait_for_question_answer
        }
    }
}

fn install_skill(url: &str, skills_dir: Option<&PathBuf>) -> Result<String> {
    let dir = skills_dir.ok_or_else(|| anyhow!("No skills directory configured"))?;
    let name = url
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("skill")
        .trim_end_matches(".git")
        .to_string();
    let dest = dir.join(&name);
    let status = std::process::Command::new("git")
        .args(["clone", url, dest.to_str().unwrap_or(".")])
        .status();
    match status {
        Ok(s) if s.success() => Ok(name),
        _ => Err(anyhow!("git clone failed for {}", url)),
    }
}

fn wait_for_tool_decision(cmd_rx: &Receiver<WorkerCommand>, id: u64) -> Result<bool> {
    loop {
        match cmd_rx.recv()? {
            WorkerCommand::ToolDecision { id: resp_id, allow } => {
                if resp_id == id { return Ok(allow); }
            }
            WorkerCommand::Shutdown => return Err(anyhow!("shutdown")),
            _ => {}
        }
    }
}

fn wait_for_question_answer(cmd_rx: &Receiver<WorkerCommand>, id: u64) -> Result<String> {
    loop {
        match cmd_rx.recv()? {
            WorkerCommand::QuestionAnswer { id: resp_id, answer } => {
                if resp_id == id { return Ok(answer); }
            }
            WorkerCommand::Shutdown => return Err(anyhow!("shutdown")),
            _ => {}
        }
    }
}

// ─── Drawing ─────────────────────────────────────────────────────────────────

fn draw_ui(f: &mut Frame, app: &mut App) {
    let size = f.area();
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Length(2), Constraint::Min(5), Constraint::Length(3)])
        .split(size);

    let header = Paragraph::new(header_text(app)).block(Block::default().borders(Borders::BOTTOM));
    f.render_widget(header, outer[0]);

    let body = outer[1];
    let has_panel = app.overlay.map(is_panel_overlay).unwrap_or(false);

    let (chat_rect, overlay_rect) = if has_panel {
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

    if has_panel {
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
                _ => "",
            };
            app.scroll_overlay =
                clamp_scroll(app.scroll_overlay, overlay_text, app.overlay_width, app.overlay_height);
            let overlay = Paragraph::new(render_lines(
                overlay_text,
                app.overlay_width,
                app.selection.as_ref().filter(|s| s.target == Pane::Overlay),
            ))
            .block(overlay_block)
            .scroll((app.scroll_overlay, 0));
            f.render_widget(overlay, app.overlay_rect);
        }
    } else {
        app.overlay_rect = Rect::default();
        app.overlay_width = 1;
        app.overlay_height = 1;
        app.scroll_overlay = 0;
    }

    app.input_rect = outer[2];
    let input_block = block_with_focus("Input", app.focus == Focus::Input);
    let input_content: Line = if app.task_input.is_empty() && !app.is_generating {
        Line::from(Span::styled("Type a task…  (/help for commands)", Style::default().fg(Color::DarkGray)))
    } else {
        Line::from(app.task_input.clone())
    };
    let input = Paragraph::new(input_content).block(input_block).wrap(Wrap { trim: false });
    f.render_widget(input, outer[2]);

    // Modal overlays rendered on top
    match app.overlay {
        Some(OverlayKind::Settings) => draw_settings_modal(f, app),
        Some(OverlayKind::Experiments) => draw_experiments_modal(f, app),
        Some(OverlayKind::ContextViz) => draw_context_viz_modal(f, app),
        Some(OverlayKind::ModelPicker) => draw_model_picker_modal(f, app),
        Some(OverlayKind::SkillsManager) => draw_skills_manager_modal(f, app),
        _ => {}
    }

    // Tool approval modal (highest priority)
    if app.pending_tool.is_some() {
        draw_approval_modal(f, app);
    }

    // Agent question modal (higher priority than approval — can't happen simultaneously)
    if app.pending_question.is_some() {
        draw_question_modal(f, app);
    }
}

// ─── Modal drawing functions ──────────────────────────────────────────────────

fn draw_settings_modal(f: &mut Frame, app: &App) {
    let area = centered_rect(70, 80, f.area());
    f.render_widget(Clear, area);
    let block = block_modal("Settings  (↑↓ navigate · ←→ cycle · type · s=save · Esc=cancel)");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let fields: Vec<(String, Style)> = (0..SETTINGS_FIELD_COUNT)
        .map(|i| {
            let focused = i == app.settings_focus;
            let prefix = if focused { "> " } else { "  " };
            let name = settings_field_name(i);
            let value: String = match i {
                0 => format!("[{:?}]", app.config.backend),
                1 => {
                    if focused {
                        format!("{}_", app.settings_buffer)
                    } else {
                        let config_key = app.config.openrouter_api_key.clone().unwrap_or_default();
                        if !config_key.is_empty() {
                            let visible: String = config_key.chars().take(4).collect();
                            let stars = "*".repeat(config_key.len().saturating_sub(4).min(20));
                            format!("{}{}  ({} chars)", visible, stars, config_key.len())
                        } else if let Ok(env_key) = std::env::var("OPENROUTER_API_KEY") {
                            if !env_key.is_empty() {
                                format!("(from $OPENROUTER_API_KEY, {} chars)", env_key.len())
                            } else {
                                String::new()
                            }
                        } else {
                            String::new()
                        }
                    }
                }
                2 => {
                    if focused {
                        format!("{}_", app.settings_buffer)
                    } else {
                        app.config.openrouter_model.clone().unwrap_or_default()
                    }
                }
                3 => {
                    if focused {
                        format!("{}_", app.settings_buffer)
                    } else {
                        app.config.local_model_path.clone().unwrap_or_default()
                    }
                }
                4 => {
                    if focused {
                        format!("{}_", app.settings_buffer)
                    } else {
                        app.config.skills_dir.clone().unwrap_or_default()
                    }
                }
                5 => format!("[{}]", app.config.exec_mode),
                6 => {
                    if focused {
                        format!("{}_", app.settings_buffer)
                    } else {
                        app.config.plan_tokens.to_string()
                    }
                }
                7 => {
                    if focused {
                        format!("{}_", app.settings_buffer)
                    } else {
                        app.config.answer_tokens.to_string()
                    }
                }
                8 => {
                    if focused {
                        format!("{}_", app.settings_buffer)
                    } else {
                        match app.config.max_history_turns {
                            Some(n) => n.to_string(),
                            None => "(unlimited)".to_string(),
                        }
                    }
                }
                _ => String::new(),
            };
            let row = format!("{}{:<18} {}", prefix, name, value);
            let style = if focused {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default()
            };
            (row, style)
        })
        .collect::<Vec<_>>();

    let lines: Vec<Line> = fields
        .into_iter()
        .map(|(text, style)| Line::from(Span::styled(text, style)))
        .collect();
    let para = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(para, inner);
}

fn draw_experiments_modal(f: &mut Frame, app: &App) {
    let area = centered_rect(60, 50, f.area());
    f.render_widget(Clear, area);
    let block = block_modal("Experiments  (↑↓ navigate · Space toggle · s=save · Esc=close)");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let experiments = [
        (
            app.config.experiments.user_turn_only,
            "User-turn-only mode",
            "Omit assistant responses from history (~10x token savings).\nPer \"Do LLMs Benefit From Their Own Words?\"",
        ),
        (
            app.config.experiments.disable_plan_phase,
            "Disable plan phase",
            "Skip planning step, go straight to answer generation.",
        ),
        (
            app.config.experiments.enforce_todos,
            "Enforce TODO completion",
            "If the agent tries to exit while pending TODOs exist, inject a\nSystem reminder forcing it to complete or mark them done first.",
        ),
    ];

    let focused_style = Style::default().fg(Color::Cyan);
    let dim_style = Style::default().fg(Color::DarkGray);
    let mut lines: Vec<Line> = Vec::new();

    for (i, (enabled, name, desc)) in experiments.iter().enumerate() {
        let focused = i == app.experiments_cursor;
        let cursor_ch = if focused { ">" } else { " " };
        let checkbox = if *enabled { "[x]" } else { "[ ]" };
        let row_style = if focused { focused_style } else { Style::default() };

        lines.push(Line::from(vec![
            Span::styled(format!("{} {} ", cursor_ch, checkbox), row_style),
            Span::styled(name.to_string(), row_style.add_modifier(Modifier::BOLD)),
        ]));
        for line in desc.lines() {
            lines.push(Line::from(Span::styled(format!("      {}", line), dim_style)));
        }

        // Sub-field: intervention limit, shown only under enforce_todos (item 2)
        if i == 2 {
            let limit_value = if focused {
                format!("{}{}",
                    app.experiments_number_buffer,
                    if focused { "_" } else { "" })
            } else {
                app.experiments_number_buffer.clone()
            };
            let limit_display = if limit_value.trim_end_matches('_').is_empty() {
                format!("      Max interventions: [{}]  unlimited", limit_value)
            } else {
                format!("      Max interventions: [{}]", limit_value)
            };
            let sub_style = if focused { focused_style } else { dim_style };
            lines.push(Line::from(Span::styled(limit_display, sub_style)));
            lines.push(Line::from(Span::styled(
                "      (type a number · blank = unlimited · default = 2)",
                dim_style,
            )));
        }

        lines.push(Line::from(""));
    }

    lines.push(Line::from(Span::styled(
        "Space = Toggle  ·  type digits = set limit  ·  s = Save  ·  Esc = Close",
        dim_style,
    )));

    let para = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(para, inner);
}

fn draw_context_viz_modal(f: &mut Frame, app: &App) {
    let area = centered_rect(60, 55, f.area());
    f.render_widget(Clear, area);
    let block = block_modal("Context Window Visualization  (Esc to close)");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let bd = &app.context_breakdown;
    let total_used = bd.system_tokens
        + bd.history_user_tokens
        + bd.history_assistant_tokens
        + bd.task_tokens
        + bd.context_chunks_tokens
        + bd.tool_log_tokens;
    let total_ctx = if app.model_ctx > 0 { app.model_ctx as usize } else { total_used.max(1) };
    let bar_width: usize = 24;

    let mut text = String::new();

    let pct = (total_used as f64 / total_ctx as f64 * 100.0).min(100.0);
    let spinner = if app.is_generating {
        const FRAMES: [&str; 4] = ["|", "/", "-", "\\"];
        FRAMES[app.spinner_idx % FRAMES.len()]
    } else {
        " "
    };
    text.push_str(&format!(
        "Total  [{}] {} {}/{} tok ({:.0}%)\n\n",
        render_bar(total_used, total_ctx, bar_width),
        spinner,
        total_used,
        total_ctx,
        pct
    ));

    let rows = [
        ("System ", bd.system_tokens),
        ("Hist(U)", bd.history_user_tokens),
        ("Hist(A)", bd.history_assistant_tokens),
        ("Task   ", bd.task_tokens),
        ("Chunks ", bd.context_chunks_tokens),
        ("Tools  ", bd.tool_log_tokens),
    ];
    for (label, tokens) in &rows {
        let row_pct = if total_ctx > 0 { *tokens as f64 / total_ctx as f64 * 100.0 } else { 0.0 };
        text.push_str(&format!(
            "{} [{}] {:>5}t {:>3.0}%\n",
            label,
            render_bar(*tokens, total_ctx, bar_width),
            tokens,
            row_pct
        ));
    }
    text.push_str(&format!("\nConversation turns: {}", app.conv_turn_count));
    let utu = if app.config.experiments.user_turn_only { "ON" } else { "OFF" };
    text.push_str(&format!("\nUser-turn-only: {}", utu));
    text.push_str("\n\nEsc = Close");

    let para = Paragraph::new(text).wrap(Wrap { trim: false });
    f.render_widget(para, inner);
}

fn draw_model_picker_modal(f: &mut Frame, app: &App) {
    let area = centered_rect(72, 75, f.area());
    f.render_widget(Clear, area);
    let block = block_modal("Model Picker — OpenRouter  (type to filter · ↑↓ navigate · Enter select · Esc close)");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut text = format!("Filter: {}_\n{}\n", app.model_filter, "─".repeat(inner.width.saturating_sub(2) as usize));

    let filtered = filtered_models(app);
    if filtered.is_empty() {
        if app.openrouter_models.is_none() {
            text.push_str("Loading models...\n");
        } else {
            text.push_str("No models match filter.\n");
        }
    } else {
        let visible_start = app.model_selected.saturating_sub(10);
        for (i, model) in filtered.iter().enumerate().skip(visible_start).take(20) {
            let cursor = if i == app.model_selected { "> " } else { "  " };
            let ctx_info = model
                .context_length
                .map(|c| format!(" [{:.0}k ctx]", c as f64 / 1000.0))
                .unwrap_or_default();
            text.push_str(&format!("{}{}{}\n", cursor, model.id, ctx_info));
        }
        if filtered.len() > 20 {
            text.push_str(&format!("  ... {} more\n", filtered.len() - 20));
        }
    }

    let para = Paragraph::new(text).wrap(Wrap { trim: false });
    f.render_widget(para, inner);
}

fn draw_skills_manager_modal(f: &mut Frame, app: &App) {
    let area = centered_rect(65, 60, f.area());
    f.render_widget(Clear, area);
    let block = block_modal("Skills Manager  (type URL · Enter install · Esc close)");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let skills_dir_str = app
        .config
        .skills_dir
        .as_deref()
        .unwrap_or("(not configured — set Skills Dir in /settings)");

    let mut text = format!("Dir: {}\n\nInstall from GitHub URL:\n> {}_\n\n", skills_dir_str, app.skills_url);

    if !app.skills_status.is_empty() {
        text.push_str(&format!("Status: {}\n\n", app.skills_status));
    }

    text.push_str("Installed skills:\n");
    if app.skills_list.is_empty() {
        text.push_str("  (none)\n");
    } else {
        for skill in &app.skills_list {
            text.push_str(&format!("  • {}\n", skill));
        }
    }

    let para = Paragraph::new(text).wrap(Wrap { trim: false });
    f.render_widget(para, inner);
}

fn render_bar(value: usize, total: usize, width: usize) -> String {
    if total == 0 || width == 0 {
        return "░".repeat(width);
    }
    let filled = ((value as f64 / total as f64) * width as f64).round() as usize;
    let filled = filled.min(width);
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

// ─── Existing drawing helpers ─────────────────────────────────────────────────

fn block_with_focus(title: &str, focused: bool) -> Block<'_> {
    let style = if focused {
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    Block::default().borders(Borders::ALL).title(title).border_style(style)
}

fn block_modal(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(title)
        .border_style(Style::default().fg(Color::Cyan))
}

fn draw_approval_modal(f: &mut Frame, app: &App) {
    let Some(pending) = app.pending_tool.as_ref() else { return; };

    let area = centered_rect(66, 60, f.area());
    f.render_widget(Clear, area);

    let (title, border_color) = if app.approval_prompt_active {
        ("✏  Deny With Prompt", Color::Yellow)
    } else {
        ("⚠  Tool Approval Required", Color::Yellow)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(title)
        .border_style(Style::default().fg(border_color).add_modifier(Modifier::BOLD));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let w = inner.width.max(1) as usize;
    let rule: String = "─".repeat(w.saturating_sub(2));

    let mut lines: Vec<Line> = Vec::new();

    if app.approval_prompt_active {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Explain why and the agent will adjust its approach:",
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(""));
        let prompt_display = format!("> {}_", app.approval_prompt);
        lines.push(Line::from(Span::styled(prompt_display, Style::default().fg(Color::White))));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(&rule, Style::default().fg(Color::DarkGray))));
        lines.push(Line::from(vec![
            Span::styled("  Enter", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(" send denial  ·  "),
            Span::styled("Esc", Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)),
            Span::raw(" go back"),
        ]));
    } else {
        // Tool info section
        lines.push(Line::from(""));
        let label_style = Style::default().fg(Color::DarkGray);
        let value_style = Style::default().fg(Color::White);

        lines.push(Line::from(vec![
            Span::styled("  Tool   ", label_style),
            Span::styled(pending.call.name.clone(), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]));

        // Args: wrap long args across lines
        let args_str = pending.call.args.to_string().replace('\n', " ");
        let max_arg_len = w.saturating_sub(12);
        if args_str.len() <= max_arg_len {
            lines.push(Line::from(vec![
                Span::styled("  Args   ", label_style),
                Span::styled(args_str.clone(), value_style),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled("  Args   ", label_style),
                Span::styled(args_str[..max_arg_len].to_string(), value_style),
            ]));
            let rest = &args_str[max_arg_len..];
            for chunk in rest.chars().collect::<Vec<char>>().chunks(max_arg_len) {
                let s: String = chunk.iter().collect();
                lines.push(Line::from(vec![
                    Span::raw("           "),
                    Span::styled(s, value_style),
                ]));
            }
        }

        let risk_color = match pending.decision.risk {
            varctx_proto::tools::Risk::Low    => Color::Green,
            varctx_proto::tools::Risk::Medium => Color::Yellow,
            varctx_proto::tools::Risk::High   => Color::Red,
        };
        let risk_label = match pending.decision.risk {
            varctx_proto::tools::Risk::Low    => "Low",
            varctx_proto::tools::Risk::Medium => "Medium",
            varctx_proto::tools::Risk::High   => "High",
        };
        lines.push(Line::from(vec![
            Span::styled("  Risk   ", label_style),
            Span::styled(risk_label, Style::default().fg(risk_color).add_modifier(Modifier::BOLD)),
        ]));

        if let Some(reason) = pending.decision.reason.as_ref() {
            lines.push(Line::from(vec![
                Span::styled("  Reason ", label_style),
                Span::styled(reason.clone(), value_style),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(&rule, Style::default().fg(Color::DarkGray))));
        lines.push(Line::from(""));

        // Action buttons
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("[ A ]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled(" Approve       ", Style::default().fg(Color::Green)),
            Span::styled("[ N ]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::styled(" Decline       ", Style::default().fg(Color::Red)),
            Span::styled("[ E ]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(" Deny with prompt", Style::default().fg(Color::Yellow)),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Also: Enter = approve  ·  Backspace = decline",
            Style::default().fg(Color::DarkGray),
        )));
    }

    let para = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(para, inner);
}

fn draw_question_modal(f: &mut Frame, app: &App) {
    let Some(pq) = app.pending_question.as_ref() else { return; };
    let n_options = pq.options.len().min(4);

    let area = centered_rect(68, 65, f.area());
    f.render_widget(Clear, area);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title("❓ Agent Question")
        .border_style(Style::default().fg(Color::LightBlue).add_modifier(Modifier::BOLD));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let w = inner.width.max(1) as usize;
    let rule: String = "─".repeat(w.saturating_sub(2));
    let dim = Style::default().fg(Color::DarkGray);
    let cyan = Style::default().fg(Color::Cyan);
    let selected = Style::default().fg(Color::Green).add_modifier(Modifier::BOLD);

    let mut lines: Vec<Line> = vec![Line::from("")];

    // Question text
    for line in pq.question.lines() {
        lines.push(Line::from(Span::styled(
            format!("  {}", line),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(&rule, dim)));
    lines.push(Line::from(""));

    // Options
    for (i, opt) in pq.options.iter().take(4).enumerate() {
        let is_cursor = i == app.question_cursor;
        let cursor_ch = if is_cursor { "▶" } else { " " };
        let num = format!("[{}]", i + 1);
        let rec_tag = if i == 0 { " (recommended)" } else { "" };
        let style = if is_cursor { selected } else { Style::default() };
        lines.push(Line::from(vec![
            Span::styled(format!("  {} {} ", cursor_ch, num), style),
            Span::styled(format!("{}{}", opt, rec_tag), style),
        ]));
    }

    // Countdown for yolo mode
    if let Some(deadline) = app.question_deadline {
        let secs_left = deadline.saturating_duration_since(Instant::now()).as_secs();
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  ⏱  Auto-selecting recommended answer in {}s…", secs_left),
            Style::default().fg(Color::Yellow),
        )));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(&rule, dim)));

    // Custom input
    if app.question_custom_active {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("  Custom: ", cyan),
            Span::styled(format!("{}_", app.question_custom), Style::default().fg(Color::White)),
        ]));
        lines.push(Line::from(Span::styled("  Enter = submit  ·  Esc = cancel", dim)));
    } else {
        lines.push(Line::from(vec![
            Span::styled("  ↑↓ / 1–", dim),
            Span::styled(n_options.to_string(), dim),
            Span::styled(" navigate  ·  Enter select  ·  ", dim),
            Span::styled("C", cyan.add_modifier(Modifier::BOLD)),
            Span::styled(" custom answer", dim),
        ]));
    }

    let para = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(para, inner);
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
    let backend_str = match app.config.backend {
        Backend::Local => {
            let model = app.config.local_model_path.as_deref().unwrap_or("none");
            let name = std::path::Path::new(model)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(model);
            format!("Local:{}", name)
        }
        Backend::OpenRouter => {
            let model = app.config.openrouter_model.as_deref().unwrap_or("none");
            format!("OR:{}", model)
        }
    };
    let store = app.args.store_path.as_deref().unwrap_or("none");
    let vars = if app.args.vars.is_empty() { "none".to_string() } else { app.var_stats.clone() };
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
        format!(" ({:+} tok, {:+.1}%)", app.current_context_delta, app.current_context_pct)
    };
    let turns_str = if app.conv_turn_count > 0 {
        format!(" | turns {}", app.conv_turn_count)
    } else {
        String::new()
    };
    let pending_todos = app.todos.iter().filter(|t| !t.completed).count();
    let todos_str = if pending_todos > 0 {
        format!(" | {} TODO{}", pending_todos, if pending_todos == 1 { "" } else { "s" })
    } else {
        String::new()
    };
    if app.model_ctx > 0 {
        format!(
            "EquiCode | Backend: {} | Store: {} | Vars: {} | Mode: {:?}{}{} | Ctx: {}/{}{} | {}",
            backend_str, store, vars, app.exec_mode, turns_str, todos_str,
            app.context_tokens_est, app.model_ctx, ctx_delta, status
        )
    } else {
        format!(
            "EquiCode | Backend: {} | Store: {} | Vars: {} | Mode: {:?}{}{} | Ctx: {}{} | {}",
            backend_str, store, vars, app.exec_mode, turns_str, todos_str,
            app.context_tokens_est, ctx_delta, status
        )
    }
}

// ─── Text utilities ───────────────────────────────────────────────────────────

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
    if text.is_empty() { return out; }
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

/// Decode a `\x01X` marker line into a styled `Line`, or return None if not a marker.
fn decode_marker_line(line: &str, width: u16) -> Option<Line<'static>> {
    let mut chars = line.chars();
    if chars.next() != Some('\x01') { return None; }
    let tag = chars.next()?;
    let rest: String = chars.collect();
    Some(match tag {
        'U' => Line::from(Span::styled("You", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))),
        'A' => Line::from(Span::styled("Assistant", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        'T' => Line::from(Span::styled("Tools", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))),
        'S' => Line::from(Span::styled(rest, Style::default().fg(Color::DarkGray))),
        '─' => Line::from(Span::styled("─".repeat(width.max(4) as usize), Style::default().fg(Color::DarkGray))),
        _ => Line::from(line.to_string()),
    })
}

fn render_lines(text: &str, width: u16, selection: Option<&Selection>) -> Vec<Line<'static>> {
    let lines = wrap_lines(text, width);
    if lines.is_empty() { return vec![Line::from("")]; }
    let Some(sel) = selection else {
        return lines.into_iter().map(|l| {
            decode_marker_line(&l, width).unwrap_or_else(|| Line::from(l))
        }).collect();
    };
    if sel.is_empty() {
        return lines.into_iter().map(|l| {
            decode_marker_line(&l, width).unwrap_or_else(|| Line::from(l))
        }).collect();
    }
    let (start_line, start_col, end_line, end_col) = sel.normalized();
    let highlight = Style::default().bg(Color::Blue).fg(Color::White);
    lines
        .into_iter()
        .enumerate()
        .map(|(idx, line)| {
            // Marker lines are never part of selection
            if let Some(styled) = decode_marker_line(&line, width) { return styled; }
            let line_len = line.chars().count();
            if idx < start_line || idx > end_line { return Line::from(line); }
            let sel_start = if idx == start_line { start_col } else { 0 };
            let sel_end = if idx == end_line { end_col } else { line_len };
            if sel_start >= sel_end || line_len == 0 { return Line::from(line); }
            let chars: Vec<char> = line.chars().collect();
            let pre: String = chars.iter().take(sel_start).collect();
            let mid: String = chars.iter().skip(sel_start).take(sel_end - sel_start).collect();
            let post: String = chars.iter().skip(sel_end).collect();
            let mut spans = Vec::new();
            if !pre.is_empty() { spans.push(Span::raw(pre)); }
            if !mid.is_empty() { spans.push(Span::styled(mid, highlight)); }
            if !post.is_empty() { spans.push(Span::raw(post)); }
            Line::from(spans)
        })
        .collect()
}

fn extract_selection_text(text: &str, width: u16, selection: &Selection) -> String {
    let lines = wrap_lines(text, width);
    if lines.is_empty() { return String::new(); }
    let (start_line, start_col, end_line, end_col) = selection.normalized();
    let mut out = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if idx < start_line || idx > end_line { continue; }
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
    if chars == 0 { 0 } else { (chars + 3) / 4 }
}

// ─── Terminal helpers ─────────────────────────────────────────────────────────

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
    terminal.backend_mut().execute(DisableMouseCapture)?.execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn parse_exec_mode(mode: Option<String>) -> ExecMode {
    match mode.as_deref() {
        Some("confirm") => ExecMode::Confirm,
        Some("paranoid") => ExecMode::Paranoid,
        _ => ExecMode::Yolo,
    }
}
