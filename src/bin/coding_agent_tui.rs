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
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::io;
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

    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (evt_tx, evt_rx) = mpsc::channel();
    let worker_cfg = llm_cfg;
    let worker_agent_cfg = agent_cfg.clone();
    let worker_handle = thread::spawn(move || worker_loop(cmd_rx, evt_tx, worker_cfg, worker_agent_cfg));

    let mut terminal = init_terminal()?;
    let mut app = App::new(args, agent_cfg, model_ctx, cancel_flag);
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
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Focus {
    Input,
    Context,
    Plan,
    Answer,
}

enum WorkerCommand {
    Run {
        task: String,
        context: Option<String>,
        plan_tokens: usize,
        answer_tokens: usize,
    },
    Shutdown,
}

enum WorkerEvent {
    Stream {
        target: StreamTarget,
        event: StreamEvent,
        chunk: String,
    },
    Done {
        plan: String,
        answer: String,
        cancelled: bool,
        elapsed_ms: u128,
    },
    Error(String),
}

impl Focus {
    fn next(self) -> Self {
        match self {
            Self::Input => Self::Context,
            Self::Context => Self::Plan,
            Self::Plan => Self::Answer,
            Self::Answer => Self::Input,
        }
    }

    fn prev(self) -> Self {
        match self {
            Self::Input => Self::Answer,
            Self::Context => Self::Input,
            Self::Plan => Self::Context,
            Self::Answer => Self::Plan,
        }
    }
}

struct App {
    args: TuiArgs,
    agent_cfg: CodingAgentConfig,
    focus: Focus,
    status: String,
    task_input: String,
    context: String,
    plan: String,
    answer: String,
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
    scroll_context: u16,
    scroll_plan: u16,
    scroll_answer: u16,
    context_width: u16,
    plan_width: u16,
    answer_width: u16,
    context_rect: Rect,
    plan_rect: Rect,
    answer_rect: Rect,
    input_rect: Rect,
    auto_follow_plan: bool,
    auto_follow_answer: bool,
    clipboard: Option<Clipboard>,
    selection_mode: bool,
    selection: Option<Selection>,
    last_mouse: Option<Position>,
}

impl App {
    fn new(
        args: TuiArgs,
        agent_cfg: CodingAgentConfig,
        model_ctx: u32,
        cancel_flag: Arc<AtomicBool>,
    ) -> Self {
        let task_input = args.preset_task.clone().unwrap_or_default();
        Self {
            args,
            agent_cfg,
            focus: Focus::Input,
            status: "Ready".to_string(),
            task_input,
            context: "No context loaded.".to_string(),
            plan: String::new(),
            answer: String::new(),
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
            scroll_context: 0,
            scroll_plan: 0,
            scroll_answer: 0,
            context_width: 80,
            plan_width: 80,
            answer_width: 80,
            context_rect: Rect::default(),
            plan_rect: Rect::default(),
            answer_rect: Rect::default(),
            input_rect: Rect::default(),
            auto_follow_plan: true,
            auto_follow_answer: true,
            clipboard: Clipboard::new().ok(),
            selection_mode: false,
            selection: None,
            last_mouse: None,
        }
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
            if !entry.stats.trim().is_empty() {
                out.push_str("\n\n");
                out.push_str(entry.stats.trim());
            }
        }
        self.answer = out;
    }

    fn copy_focused(&mut self) {
        let text = self
            .selection_text(self.focus)
            .unwrap_or_else(|| match self.focus {
            Focus::Context => self.context.clone(),
            Focus::Plan => self.plan.clone(),
            Focus::Answer => self.answer.clone(),
            Focus::Input => self.task_input.clone(),
            });
        if text.is_empty() {
            self.status = "Nothing to copy".to_string();
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

    fn selection_text(&self, focus: Focus) -> Option<String> {
        let sel = self.selection.as_ref()?;
        if sel.target != focus || sel.is_empty() {
            return None;
        }
        let (text, width) = match focus {
            Focus::Context => (&self.context, self.context_width),
            Focus::Plan => (&self.plan, self.plan_width),
            Focus::Answer => (&self.answer, self.answer_width),
            Focus::Input => (&self.task_input, self.context_width),
        };
        Some(extract_selection_text(text, width, sel))
    }

    fn scroll_active(&mut self, delta: i16) {
        match self.focus {
            Focus::Context => {
                self.scroll_context =
                    scroll_offset(self.scroll_context, delta, &self.context, self.context_width)
            }
            Focus::Plan => {
                self.scroll_plan =
                    scroll_offset(self.scroll_plan, delta, &self.plan, self.plan_width);
                self.auto_follow_plan =
                    self.scroll_plan >= max_scroll(&self.plan, self.plan_width);
            }
            Focus::Answer => {
                self.scroll_answer =
                    scroll_offset(self.scroll_answer, delta, &self.answer, self.answer_width);
                self.auto_follow_answer =
                    self.scroll_answer >= max_scroll(&self.answer, self.answer_width);
            }
            Focus::Input => {}
        }
    }
}

fn toggle_selection_mode(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    if app.selection_mode {
        terminal.backend_mut().execute(EnableMouseCapture)?;
        app.selection_mode = false;
        app.status = if app.is_generating {
            "Running...".to_string()
        } else {
            "Ready".to_string()
        };
    } else {
        terminal.backend_mut().execute(DisableMouseCapture)?;
        app.selection_mode = true;
        app.status = "Selection mode: drag to select, press s to resume".to_string();
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ChatEntry {
    user: String,
    answer: String,
    stats: String,
}

#[derive(Debug, Clone)]
struct Selection {
    target: Focus,
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

fn scroll_offset(current: u16, delta: i16, text: &str, width: u16) -> u16 {
    let lines = wrapped_line_count(text, width).max(1);
    let max_scroll = lines.saturating_sub(1) as i16;
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

    if app.selection_mode {
        match key.code {
            KeyCode::Char('s')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    || app.focus != Focus::Input =>
            {
                toggle_selection_mode(terminal, app)?;
            }
            KeyCode::Esc => {
                toggle_selection_mode(terminal, app)?;
            }
            KeyCode::Char('q') => return Ok(true),
            _ => {}
        }
        return Ok(false);
    }

    match key.code {
        KeyCode::Esc => {
            if app.selection.is_some() {
                app.selection = None;
                app.status = "Selection cleared".to_string();
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
            app.focus = app.focus.next();
            app.selection = None;
        }
        KeyCode::BackTab => {
            app.focus = app.focus.prev();
            app.selection = None;
        }
        KeyCode::Up => app.scroll_active(-1),
        KeyCode::Down => app.scroll_active(1),
        KeyCode::PageUp => app.scroll_active(-5),
        KeyCode::PageDown => app.scroll_active(5),
        KeyCode::Char('s')
            if key.modifiers.contains(KeyModifiers::CONTROL) || app.focus != Focus::Input =>
        {
            toggle_selection_mode(terminal, app)?;
        }
        KeyCode::Char('y')
            if app.focus != Focus::Input
                || key.modifiers.contains(KeyModifiers::CONTROL) =>
        {
            app.copy_focused();
        }
        KeyCode::Enter if app.focus == Focus::Input => {
            if app.task_input.trim().is_empty() {
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
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) && app.focus == Focus::Input => {
            app.task_input.clear();
        }
        KeyCode::Char(c) if app.focus == Focus::Input && !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.task_input.push(c);
        }
        _ => {}
    }

    Ok(false)
}

fn handle_mouse(app: &mut App, me: MouseEvent) -> bool {
    if app.selection_mode {
        return false;
    }
    app.last_mouse = Some(Position {
        x: me.column,
        y: me.row,
    });
    let focus = focus_from_point(app, me.column, me.row);
    match me.kind {
        MouseEventKind::Down(crossterm::event::MouseButton::Left) => {
            if let Some(f) = focus {
                app.focus = f;
                if matches!(f, Focus::Context | Focus::Plan | Focus::Answer) {
                    if let Some((line, col)) =
                        selection_pos_for_focus(app, f, me.column, me.row)
                    {
                        app.selection = Some(Selection {
                            target: f,
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
            if let Some((line, col)) = selection_pos_for_focus(app, target, me.column, me.row) {
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
                app.copy_focused();
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
    if app.context_rect.contains(p) {
        return Some(Focus::Context);
    }
    if app.plan_rect.contains(p) {
        return Some(Focus::Plan);
    }
    if app.answer_rect.contains(p) {
        return Some(Focus::Answer);
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
    if let Some((line, col)) = selection_pos_for_focus(app, target, pos.x, pos.y) {
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

fn selection_pos_for_focus(
    app: &App,
    focus: Focus,
    x: u16,
    y: u16,
) -> Option<(usize, usize)> {
    let (rect, text, width, scroll) = match focus {
        Focus::Context => (
            app.context_rect,
            &app.context,
            app.context_width,
            app.scroll_context,
        ),
        Focus::Plan => (app.plan_rect, &app.plan, app.plan_width, app.scroll_plan),
        Focus::Answer => (
            app.answer_rect,
            &app.answer,
            app.answer_width,
            app.scroll_answer,
        ),
        Focus::Input => return None,
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
    app.scroll_plan = 0;
    app.scroll_answer = 0;
    app.auto_follow_plan = true;
    app.auto_follow_answer = true;

    if let Some(store_path) = app.args.store_path.as_deref() {
        if app.args.vars.is_empty() {
            return Err(anyhow!("--store requires --vars <V:...,...>"));
        }

        let store = ContextStore::open(store_path)?;
        let mut var_parts = Vec::new();
        for v in &app.args.vars {
            if let Some(binding) = store.get_var_binding_latest_lossy(v)? {
                var_parts.push(format!("{}({})", v, binding.chunk_ids.len()));
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
        let cfg = ContextBuildConfig {
            max_snippets: app.args.max_snippets.unwrap_or(6),
            snippet_chars: app.args.snippet_chars.unwrap_or(800),
        };
        let ctx = build_context(&store, &app.args.vars, &retrieved, &cfg)?;
        context = Some(ctx);
    } else {
        app.var_stats = "none".to_string();
    }

    if let Some(ctx) = context.as_ref() {
        app.context = ctx.clone();
        app.scroll_context = 0;
        app.context_tokens_est = estimate_tokens(ctx);
    } else {
        app.context = "No context loaded.".to_string();
        app.scroll_context = 0;
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
        stats: String::new(),
    });
    app.current_turn = Some(entry_idx);
    app.update_history_render();
    app.scroll_answer = max_scroll(&app.answer, app.answer_width);

    cmd_tx.send(WorkerCommand::Run {
        task,
        context,
        plan_tokens: app.agent_cfg.max_plan_tokens,
        answer_tokens: app.agent_cfg.max_answer_tokens,
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
            match (target, event) {
                (StreamTarget::Plan, StreamEvent::Start) => {
                    app.plan.clear();
                    app.scroll_plan = 0;
                    app.auto_follow_plan = true;
                }
                (StreamTarget::Answer, StreamEvent::Start) => {
                    app.scroll_answer = 0;
                    app.auto_follow_answer = true;
                }
                (StreamTarget::Plan, StreamEvent::Chunk) => {
                    let at_bottom = app.scroll_plan >= max_scroll(&app.plan, app.plan_width);
                    app.plan.push_str(&chunk);
                    if app.auto_follow_plan || at_bottom {
                        app.scroll_plan = max_scroll(&app.plan, app.plan_width);
                        app.auto_follow_plan = true;
                    }
                }
                (StreamTarget::Answer, StreamEvent::Chunk) => {
                    let idx = match app.current_turn {
                        Some(i) => i,
                        None => return Ok(()),
                    };
                    let at_bottom = app.scroll_answer >= max_scroll(&app.answer, app.answer_width);
                    if let Some(entry) = app.history.get_mut(idx) {
                        entry.answer.push_str(&chunk);
                    }
                    app.update_history_render();
                    if app.auto_follow_answer || at_bottom {
                        app.scroll_answer = max_scroll(&app.answer, app.answer_width);
                        app.auto_follow_answer = true;
                    }
                }
                _ => {}
            }
        }
        WorkerEvent::Done {
            plan,
            answer,
            cancelled,
            elapsed_ms,
        } => {
            app.plan = plan.trim().to_string();
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
            app.is_generating = false;
            if cancelled {
                app.status = "Cancelled".to_string();
            } else {
                app.status = "Done".to_string();
            }
        }
        WorkerEvent::Error(msg) => {
            app.is_generating = false;
            app.status = format!("Error: {}", msg);
        }
    }
    Ok(())
}

fn worker_loop(
    cmd_rx: Receiver<WorkerCommand>,
    evt_tx: Sender<WorkerEvent>,
    llm_cfg: LlmConfig,
    base_agent_cfg: CodingAgentConfig,
) {
    let cancel_flag = llm_cfg.cancel_flag.clone();
    let mut llm = match LlamaCppLlm::load(llm_cfg) {
        Ok(v) => v,
        Err(err) => {
            let _ = evt_tx.send(WorkerEvent::Error(err.to_string()));
            return;
        }
    };

    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            WorkerCommand::Run {
                task,
                context,
                plan_tokens,
                answer_tokens,
            } => {
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
                match agent.run_streaming(&task, context.as_deref(), &mut on_stream) {
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
            WorkerCommand::Shutdown => {
                break;
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

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(outer[1]);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(5)])
        .split(body[1]);

    app.context_rect = body[0];
    let context_block = block_with_focus("Context", app.focus == Focus::Context);
    let context_inner = context_block.inner(app.context_rect);
    app.context_width = context_inner.width.max(1);
    app.scroll_context = clamp_scroll(app.scroll_context, &app.context, app.context_width);
    let context = Paragraph::new(render_lines(
        &app.context,
        app.context_width,
        app.selection.as_ref().filter(|s| s.target == Focus::Context),
    ))
        .block(context_block)
        .scroll((app.scroll_context, 0));
    f.render_widget(context, body[0]);

    app.plan_rect = right[0];
    let plan_block = block_with_focus("Plan", app.focus == Focus::Plan);
    let plan_inner = plan_block.inner(app.plan_rect);
    app.plan_width = plan_inner.width.max(1);
    app.scroll_plan = clamp_scroll(app.scroll_plan, &app.plan, app.plan_width);
    let plan = Paragraph::new(render_lines(
        &app.plan,
        app.plan_width,
        app.selection.as_ref().filter(|s| s.target == Focus::Plan),
    ))
        .block(plan_block)
        .scroll((app.scroll_plan, 0));
    f.render_widget(plan, right[0]);

    app.answer_rect = right[1];
    let answer_block = block_with_focus("Answer", app.focus == Focus::Answer);
    let answer_inner = answer_block.inner(app.answer_rect);
    app.answer_width = answer_inner.width.max(1);
    app.scroll_answer = clamp_scroll(app.scroll_answer, &app.answer, app.answer_width);
    let answer = Paragraph::new(render_lines(
        &app.answer,
        app.answer_width,
        app.selection.as_ref().filter(|s| s.target == Focus::Answer),
    ))
        .block(answer_block)
        .scroll((app.scroll_answer, 0));
    f.render_widget(answer, right[1]);

    app.input_rect = outer[2];
    let input_block = block_with_focus("Task Input", app.focus == Focus::Input);
    let input = Paragraph::new(app.task_input.clone())
        .block(input_block)
        .wrap(Wrap { trim: false });
    f.render_widget(input, outer[2]);
}

fn block_with_focus(title: &str, focused: bool) -> Block<'_> {
    let style = if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    };
    Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(style)
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
    if app.model_ctx > 0 {
        format!(
            "Varctx TUI | Store: {} | Vars: {} | Ctx: {}/{} tok | Status: {}",
            store, vars, app.context_tokens_est, app.model_ctx, app.status
        )
    } else {
        format!(
            "Varctx TUI | Store: {} | Vars: {} | Ctx: {} tok | Status: {}",
            store, vars, app.context_tokens_est, app.status
        )
    }
}

fn clamp_scroll(current: u16, text: &str, width: u16) -> u16 {
    let lines = wrapped_line_count(text, width).max(1);
    let max_scroll = lines.saturating_sub(1) as u16;
    current.min(max_scroll)
}

fn max_scroll(text: &str, width: u16) -> u16 {
    let lines = wrapped_line_count(text, width).max(1);
    lines.saturating_sub(1) as u16
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
