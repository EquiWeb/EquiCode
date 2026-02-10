use anyhow::{anyhow, Result};

#[derive(Debug, Clone, Copy)]
pub enum ConstraintKind {
    Assert,
    Suggest,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub kind: ConstraintKind,
    pub message: String,
    pub check: fn(&str) -> bool, // for prototype; later: Box<dyn Fn(...)>
    pub max_retries: usize,
}

#[derive(Debug, Clone)]
pub struct Attempt {
    pub output: String,
    pub error_msg: String,
}

pub fn run_with_retry<F>(
    constraints: &[Constraint],
    mut run_once: F,
) -> Result<(String, Vec<Attempt>)>
where
    F: FnMut(&[Attempt]) -> Result<String>,
{
    let mut attempts: Vec<Attempt> = vec![];

    // unify retry budget: max across constraints (simple)
    let max_r = constraints
        .iter()
        .map(|c| c.max_retries)
        .max()
        .unwrap_or(0);

    for _try in 0..=max_r {
        let out = run_once(&attempts)?;

        let mut hard_failed: Option<String> = None;
        let mut soft_failed: Option<String> = None;

        for c in constraints {
            let ok = (c.check)(&out);
            if !ok {
                match c.kind {
                    ConstraintKind::Assert => hard_failed = Some(c.message.clone()),
                    ConstraintKind::Suggest => {
                        if soft_failed.is_none() {
                            soft_failed = Some(c.message.clone());
                        }
                    }
                }
            }
        }

        if hard_failed.is_none() && soft_failed.is_none() {
            return Ok((out, attempts));
        }

        // Decide whether to retry.
        if let Some(msg) = hard_failed {
            attempts.push(Attempt {
                output: out,
                error_msg: msg,
            });
            continue; // retry until exhausted
        }

        // Suggest-only failure:
        if let Some(msg) = soft_failed {
            attempts.push(Attempt {
                output: out,
                error_msg: msg,
            });
            // retry if budget remains, else return last output
            if attempts.len() <= max_r {
                continue;
            } else {
                let last = attempts.last().unwrap().output.clone();
                return Ok((last, attempts));
            }
        }
    }

    Err(anyhow!(
        "assertion failed after max retries"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assert_retries_then_errors() {
        let constraints = vec![Constraint {
            kind: ConstraintKind::Assert,
            message: "must contain ok".to_string(),
            check: |s| s.contains("ok"),
            max_retries: 2,
        }];

        let mut calls = 0;
        let res = run_with_retry(&constraints, |_attempts| {
            calls += 1;
            Ok("nope".to_string())
        });

        assert!(res.is_err());
        assert_eq!(calls, 3);
    }

    #[test]
    fn suggest_returns_last_after_budget() -> Result<()> {
        let constraints = vec![Constraint {
            kind: ConstraintKind::Suggest,
            message: "must contain ok".to_string(),
            check: |s| s.contains("ok"),
            max_retries: 1,
        }];

        let mut calls = 0;
        let (out, attempts) = run_with_retry(&constraints, |_attempts| {
            calls += 1;
            Ok(format!("try {calls}"))
        })?;

        assert_eq!(out, "try 2");
        assert_eq!(attempts.len(), 2);
        Ok(())
    }
}
