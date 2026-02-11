use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::pipeline::assertions::{run_with_retry, Attempt, Constraint};

#[derive(Debug, Default)]
pub struct PipelineState {
    outputs: HashMap<String, String>,
}

impl PipelineState {
    pub fn get(&self, name: &str) -> Option<&str> {
        self.outputs.get(name).map(|v| v.as_str())
    }

    pub fn insert(&mut self, name: impl Into<String>, value: String) {
        self.outputs.insert(name.into(), value);
    }
}

pub struct Step<'a> {
    pub name: String,
    pub constraints: Vec<Constraint>,
    pub backtrack_to: Option<String>,
    pub run: Box<dyn FnMut(&PipelineState, &[Attempt]) -> Result<String> + 'a>,
}

impl<'a> Step<'a> {
    pub fn new(
        name: impl Into<String>,
        run: impl FnMut(&PipelineState, &[Attempt]) -> Result<String> + 'a,
    ) -> Self {
        Self {
            name: name.into(),
            constraints: Vec::new(),
            backtrack_to: None,
            run: Box::new(run),
        }
    }

    pub fn with_constraints(mut self, constraints: Vec<Constraint>) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn with_backtrack_to(mut self, step_name: impl Into<String>) -> Self {
        self.backtrack_to = Some(step_name.into());
        self
    }
}

pub struct Pipeline<'a> {
    steps: Vec<Step<'a>>,
    max_iters: usize,
}

impl<'a> Pipeline<'a> {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            max_iters: 64,
        }
    }

    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters.max(1);
        self
    }

    pub fn add_step(&mut self, step: Step<'a>) {
        self.steps.push(step);
    }

    pub fn run(&mut self) -> Result<PipelineState> {
        let mut state = PipelineState::default();
        let mut iter_count = 0usize;

        let mut name_to_idx = HashMap::new();
        for (idx, step) in self.steps.iter().enumerate() {
            name_to_idx.insert(step.name.clone(), idx);
        }

        let mut idx = 0usize;
        while idx < self.steps.len() {
            iter_count += 1;
            if iter_count > self.max_iters {
                return Err(anyhow!(
                    "pipeline exceeded max iterations ({})",
                    self.max_iters
                ));
            }

            let step = &mut self.steps[idx];
            let result = run_with_retry(&step.constraints, |attempts| (step.run)(&state, attempts));

            match result {
                Ok((out, _attempts)) => {
                    state.insert(step.name.clone(), out);
                    idx += 1;
                }
                Err(err) => {
                    if let Some(target_name) = step.backtrack_to.as_deref() {
                        let Some(&target_idx) = name_to_idx.get(target_name) else {
                            return Err(anyhow!("backtrack target not found: {}", target_name));
                        };

                        for s in self.steps.iter().skip(target_idx) {
                            state.outputs.remove(&s.name);
                        }
                        idx = target_idx;
                    } else {
                        return Err(err);
                    }
                }
            }
        }

        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::assertions::{Constraint, ConstraintKind};

    #[test]
    fn pipeline_runs_steps_and_collects_outputs() -> Result<()> {
        let mut pipeline = Pipeline::new().with_max_iters(4);

        pipeline.add_step(Step::new("step1", |_state, _attempts| {
            Ok("one".to_string())
        }));

        pipeline.add_step(Step::new("step2", |state, _attempts| {
            let prev = state.get("step1").unwrap_or("");
            Ok(format!("two:{prev}"))
        }));

        let state = pipeline.run()?;
        assert_eq!(state.get("step1"), Some("one"));
        assert_eq!(state.get("step2"), Some("two:one"));
        Ok(())
    }

    #[test]
    fn pipeline_backtracks_on_error() -> Result<()> {
        let mut pipeline = Pipeline::new().with_max_iters(6);

        let mut counter = 0;
        pipeline.add_step(Step::new("step1", move |_state, _attempts| {
            counter += 1;
            if counter == 1 {
                Ok("bad".to_string())
            } else {
                Ok("good".to_string())
            }
        }));

        let constraints = vec![Constraint {
            kind: ConstraintKind::Assert,
            message: "must contain ok".to_string(),
            check: |s| s.contains("ok"),
            max_retries: 0,
        }];

        pipeline.add_step(
            Step::new("step2", |state, _attempts| {
                let v = state.get("step1").unwrap_or("");
                if v == "good" {
                    Ok("ok".to_string())
                } else {
                    Ok("nope".to_string())
                }
            })
            .with_constraints(constraints)
            .with_backtrack_to("step1"),
        );

        let state = pipeline.run()?;
        assert_eq!(state.get("step1"), Some("good"));
        assert_eq!(state.get("step2"), Some("ok"));
        Ok(())
    }
}
