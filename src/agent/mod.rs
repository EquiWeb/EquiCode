use anyhow::Result;

use crate::llm::{Llm, Prompt};
use crate::pipeline::assertions::{run_with_retry, Attempt, Constraint};

pub mod context;

#[derive(Debug, Clone)]
pub struct CodingAgentConfig {
    pub plan_system: String,
    pub answer_system: String,
    pub max_plan_tokens: usize,
    pub max_answer_tokens: usize,
    pub plan_constraints: Vec<Constraint>,
    pub answer_constraints: Vec<Constraint>,
}

impl Default for CodingAgentConfig {
    fn default() -> Self {
        Self {
            plan_system: "You are a planning module. Produce a short, actionable plan.".to_string(),
            answer_system: "You are a coding agent. Use the plan and context to respond."
                .to_string(),
            max_plan_tokens: 256,
            max_answer_tokens: 512,
            plan_constraints: Vec::new(),
            answer_constraints: Vec::new(),
        }
    }
}

pub struct CodingAgent<'a> {
    llm: &'a mut dyn Llm,
    config: CodingAgentConfig,
}

#[derive(Debug, Clone)]
pub struct AgentResult {
    pub plan: String,
    pub answer: String,
    pub plan_attempts: Vec<Attempt>,
    pub answer_attempts: Vec<Attempt>,
}

#[derive(Debug, Clone, Copy)]
pub enum StreamTarget {
    Plan,
    Answer,
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
        })
    }
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
