use anyhow::{anyhow, Result};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::{execute, ExecutableCommand};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::io;
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

    let mut llm_cfg = LlmConfig::from_env(model_path);
    llm_cfg.silence_logs = true;
    let mut llm = LlamaCppLlm::load(llm_cfg)?;

    let mut agent_cfg = CodingAgentConfig::default();
    if let Some(tokens) = args.plan_tokens {
        agent_cfg.max_plan_tokens = tokens.max(1);
    }
    if let Some(tokens) = args.answer_tokens {
        agent_cfg.max_answer_tokens = tokens.max(1);
    }

    let mut terminal = init_terminal()?;
    let mut app = App::new(args, agent_cfg);
    let res = run_app(&mut terminal, &mut llm, &mut app);
    restore_terminal(&mut terminal)?;
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
    scroll_context: u16,
    scroll_plan: u16,
    scroll_answer: u16,
}

impl App {
    fn new(args: TuiArgs, agent_cfg: CodingAgentConfig) -> Self {
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
            scroll_context: 0,
            scroll_plan: 0,
            scroll_answer: 0,
        }
    }

    fn set_error(&mut self, err: &anyhow::Error) {
        self.status = format!("Error: {}", err);
    }

    fn scroll_active(&mut self, delta: i16) {
        match self.focus {
            Focus::Context => self.scroll_context = scroll_offset(self.scroll_context, delta, &self.context),
            Focus::Plan => self.scroll_plan = scroll_offset(self.scroll_plan, delta, &self.plan),
            Focus::Answer => self.scroll_answer = scroll_offset(self.scroll_answer, delta, &self.answer),
            Focus::Input => {}
        }
    }
}

fn scroll_offset(current: u16, delta: i16, text: &str) -> u16 {
    let lines = text.lines().count().max(1);
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
    llm: &mut LlamaCppLlm,
    app: &mut App,
) -> Result<()> {
    loop {
        terminal.draw(|f| draw_ui(f, app))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if handle_key(terminal, llm, app, key)? {
                    return Ok(());
                }
            }
        }
    }
}

fn handle_key(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    llm: &mut LlamaCppLlm,
    app: &mut App,
    key: KeyEvent,
) -> Result<bool> {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Ok(true);
    }

    match key.code {
        KeyCode::Esc => {
            if app.focus == Focus::Input && !app.task_input.is_empty() {
                app.task_input.clear();
            } else {
                return Ok(true);
            }
        }
        KeyCode::Char('q') if app.focus != Focus::Input => return Ok(true),
        KeyCode::Tab => app.focus = app.focus.next(),
        KeyCode::BackTab => app.focus = app.focus.prev(),
        KeyCode::Up => app.scroll_active(-1),
        KeyCode::Down => app.scroll_active(1),
        KeyCode::PageUp => app.scroll_active(-5),
        KeyCode::PageDown => app.scroll_active(5),
        KeyCode::Enter if app.focus == Focus::Input => {
            if app.task_input.trim().is_empty() {
                return Ok(false);
            }
            app.status = "Running...".to_string();
            terminal.draw(|f| draw_ui(f, app))?;
            if let Err(err) = run_task_streaming(terminal, llm, app) {
                app.set_error(&err);
            } else {
                app.status = "Done".to_string();
            }
            app.task_input.clear();
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

fn run_task_streaming(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    llm: &mut LlamaCppLlm,
    app: &mut App,
) -> Result<()> {
    let task = app.task_input.trim().to_string();
    let mut context = None;

    app.plan.clear();
    app.answer.clear();
    app.scroll_plan = 0;
    app.scroll_answer = 0;

    if let Some(store_path) = app.args.store_path.as_deref() {
        if app.args.vars.is_empty() {
            return Err(anyhow!("--store requires --vars <V:...,...>"));
        }

        let store = ContextStore::open(store_path)?;
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
    }

    if let Some(ctx) = context.as_ref() {
        app.context = ctx.clone();
        app.scroll_context = 0;
    } else {
        app.context = "No context loaded.".to_string();
        app.scroll_context = 0;
    }

    let mut agent = CodingAgent::new(llm, app.agent_cfg.clone());
    let mut last_draw = Instant::now();
    let mut on_stream = |target: StreamTarget, event: StreamEvent, chunk: &str| {
        match (target, event) {
            (StreamTarget::Plan, StreamEvent::Start) => {
                app.plan.clear();
                app.scroll_plan = 0;
            }
            (StreamTarget::Answer, StreamEvent::Start) => {
                app.answer.clear();
                app.scroll_answer = 0;
            }
            (StreamTarget::Plan, StreamEvent::Chunk) => {
                app.plan.push_str(chunk);
            }
            (StreamTarget::Answer, StreamEvent::Chunk) => {
                app.answer.push_str(chunk);
            }
            _ => {}
        }

        let should_draw = last_draw.elapsed() >= Duration::from_millis(33) || chunk.contains('\n');
        if should_draw {
            let _ = terminal.draw(|f| draw_ui(f, app));
            last_draw = Instant::now();
        }
    };

    let result = agent.run_streaming(&task, context.as_deref(), &mut on_stream)?;

    app.plan = result.plan.trim().to_string();
    app.answer = result.answer.trim().to_string();
    let _ = terminal.draw(|f| draw_ui(f, app));
    app.scroll_plan = 0;
    app.scroll_answer = 0;
    Ok(())
}

fn draw_ui(f: &mut Frame, app: &App) {
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

    let context_block = block_with_focus("Context", app.focus == Focus::Context);
    let context = Paragraph::new(app.context.clone())
        .block(context_block)
        .wrap(Wrap { trim: false })
        .scroll((app.scroll_context, 0));
    f.render_widget(context, body[0]);

    let plan_block = block_with_focus("Plan", app.focus == Focus::Plan);
    let plan = Paragraph::new(app.plan.clone())
        .block(plan_block)
        .wrap(Wrap { trim: false })
        .scroll((app.scroll_plan, 0));
    f.render_widget(plan, right[0]);

    let answer_block = block_with_focus("Answer", app.focus == Focus::Answer);
    let answer = Paragraph::new(app.answer.clone())
        .block(answer_block)
        .wrap(Wrap { trim: false })
        .scroll((app.scroll_answer, 0));
    f.render_widget(answer, right[1]);

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
        app.args.vars.join(",")
    };
    format!(
        "Varctx TUI | Store: {} | Vars: {} | Status: {}",
        store, vars, app.status
    )
}

fn init_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;
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
