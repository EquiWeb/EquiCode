use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use monty::{
    CollectStringPrint, ExcType, ExternalResult, MontyException, MontyObject, MontyRun, NoLimitTracker,
    OsFunction, RunProgress,
};

use super::{SafetyPolicy, ToolCall};

#[derive(Debug, Deserialize)]
struct CodeExecArgs {
    code: String,
    #[serde(default)]
    inputs: Map<String, Value>,
    #[serde(default)]
    input_names: Vec<String>,
    script_name: Option<String>,
}

pub fn run_code_exec(args: &Value, policy: &SafetyPolicy) -> Result<String> {
    let parsed: CodeExecArgs = serde_json::from_value(args.clone())
        .map_err(|err| anyhow!("invalid code.exec args: {err}"))?;

    if parsed.code.trim().is_empty() {
        return Err(anyhow!("code.exec requires non-empty code"));
    }

    let input_names = if parsed.input_names.is_empty() {
        let mut names: Vec<String> = parsed.inputs.keys().cloned().collect();
        names.sort();
        names
    } else {
        parsed.input_names.clone()
    };

    let mut input_values = Vec::with_capacity(input_names.len());
    for name in &input_names {
        let value = parsed.inputs.get(name).cloned().unwrap_or(Value::Null);
        input_values.push(json_to_monty(&value));
    }

    let script_name = parsed
        .script_name
        .unwrap_or_else(|| "code.exec".to_string());

    let external_functions = vec![
        "read".to_string(),
        "write".to_string(),
        "list".to_string(),
        "grep".to_string(),
        "exists".to_string(),
    ];
    let runner = MontyRun::new(parsed.code, &script_name, input_names, external_functions)
        .map_err(|err| anyhow!("monty init failed: {err}"))?;

    let mut printer = CollectStringPrint::new();
    let mut progress = runner
        .start(input_values, NoLimitTracker, &mut printer)
        .map_err(|err| anyhow!("monty start failed: {err}"))?;

    loop {
        match progress {
            RunProgress::Complete(value) => {
                let mut output = String::new();
                let print_out = printer.into_output();
                if !print_out.trim().is_empty() {
                    output.push_str("PRINT:\n");
                    output.push_str(print_out.trim_end());
                    output.push('\n');
                }
                output.push_str("RESULT:\n");
                output.push_str(&format!("{value}"));
                return Ok(output.trim().to_string());
            }
            RunProgress::FunctionCall {
                function_name,
                args,
                kwargs,
                state,
                ..
            } => {
                let ext_result = handle_external_function(&function_name, &args, &kwargs, policy);
                progress = state
                    .run(ext_result, &mut printer)
                    .map_err(|err| anyhow!("monty run failed: {err}"))?;
            }
            RunProgress::OsCall {
                function,
                args,
                kwargs,
                state,
                ..
            } => {
                let ext_result = handle_os_call(function, &args, &kwargs, policy);
                progress = state
                    .run(ext_result, &mut printer)
                    .map_err(|err| anyhow!("monty run failed: {err}"))?;
            }
            RunProgress::ResolveFutures(_) => {
                return Err(anyhow!("async Monty futures are not supported"));
            }
        }
    }
}

fn json_to_monty(value: &Value) -> MontyObject {
    match value {
        Value::Null => MontyObject::None,
        Value::Bool(v) => MontyObject::Bool(*v),
        Value::Number(v) => {
            if let Some(i) = v.as_i64() {
                MontyObject::Int(i)
            } else if let Some(f) = v.as_f64() {
                MontyObject::Float(f)
            } else {
                MontyObject::String(v.to_string())
            }
        }
        Value::String(v) => MontyObject::String(v.clone()),
        Value::Array(items) => MontyObject::List(items.iter().map(json_to_monty).collect()),
        Value::Object(map) => MontyObject::dict(
            map.iter()
                .map(|(k, v)| (MontyObject::String(k.clone()), json_to_monty(v)))
                .collect::<Vec<_>>(),
        ),
    }
}

fn handle_external_function(
    name: &str,
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    policy: &SafetyPolicy,
) -> ExternalResult {
    match name {
        "read" => read_text(args, kwargs, policy),
        "write" => write_text(args, kwargs, policy),
        "list" => list_dir(args, kwargs, policy),
        "grep" => grep_text(args, kwargs, policy),
        "exists" => exists_path(args, kwargs, policy),
        _ => ExternalResult::Error(MontyException::new(
            ExcType::NotImplementedError,
            Some(format!("unknown external function: {name}")),
        )),
    }
}

fn handle_os_call(
    function: OsFunction,
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    policy: &SafetyPolicy,
) -> ExternalResult {
    match function {
        OsFunction::ReadText => read_text(args, kwargs, policy),
        OsFunction::WriteText => write_text(args, kwargs, policy),
        OsFunction::Iterdir => list_dir(args, kwargs, policy),
        OsFunction::Exists => exists_path(args, kwargs, policy),
        OsFunction::IsFile => is_file(args, kwargs, policy),
        OsFunction::IsDir => is_dir(args, kwargs, policy),
        OsFunction::ReadBytes => read_bytes(args, kwargs, policy),
        OsFunction::WriteBytes => write_bytes(args, kwargs, policy),
        OsFunction::Resolve | OsFunction::Absolute => resolve_path_obj(args, kwargs, policy),
        _ => ExternalResult::Error(MontyException::new(
            ExcType::NotImplementedError,
            Some(format!("os call not supported: {function}")),
        )),
    }
}

fn resolve_path_obj(
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    policy: &SafetyPolicy,
) -> ExternalResult {
    let path = match get_path_arg(args, kwargs, "path") {
        Ok(p) => p,
        Err(err) => return err,
    };
    match resolve_path(policy, &path) {
        Ok(p) => ExternalResult::Return(MontyObject::Path(p.to_string_lossy().to_string())),
        Err(err) => err,
    }
}

fn exists_path(
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    policy: &SafetyPolicy,
) -> ExternalResult {
    let path = match get_path_arg(args, kwargs, "path") {
        Ok(p) => p,
        Err(err) => return err,
    };
    if let Err(err) = check_policy(policy, "filesystem.read", &path) {
        return err;
    }
    match resolve_path(policy, &path) {
        Ok(p) => ExternalResult::Return(MontyObject::Bool(p.exists())),
        Err(err) => err,
    }
}

fn is_file(
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    policy: &SafetyPolicy,
) -> ExternalResult {
    let path = match get_path_arg(args, kwargs, "path") {
        Ok(p) => p,
        Err(err) => return err,
    };
    if let Err(err) = check_policy(policy, "filesystem.read", &path) {
        return err;
    }
    match resolve_path(policy, &path) {
        Ok(p) => ExternalResult::Return(MontyObject::Bool(p.is_file())),
        Err(err) => err,
    }
}

fn is_dir(
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    policy: &SafetyPolicy,
) -> ExternalResult {
    let path = match get_path_arg(args, kwargs, "path") {
        Ok(p) => p,
        Err(err) => return err,
    };
    if let Err(err) = check_policy(policy, "filesystem.read", &path) {
        return err;
    }
    match resolve_path(policy, &path) {
        Ok(p) => ExternalResult::Return(MontyObject::Bool(p.is_dir())),
        Err(err) => err,
    }
}

fn read_text(
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    policy: &SafetyPolicy,
) -> ExternalResult {
    let path = match get_path_arg(args, kwargs, "path") {
        Ok(p) => p,
        Err(err) => return err,
    };
    if let Err(err) = check_policy(policy, "filesystem.read", &path) {
        return err;
    }
    let path = match resolve_path(policy, &path) {
        Ok(p) => p,
        Err(err) => return err,
    };
    match std::fs::read_to_string(&path) {
        Ok(text) => ExternalResult::Return(MontyObject::String(truncate_text(&text, 120_000))),
        Err(err) => ExternalResult::Error(MontyException::new(
            ExcType::FileNotFoundError,
            Some(err.to_string()),
        )),
    }
}

fn read_bytes(
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    policy: &SafetyPolicy,
) -> ExternalResult {
    let path = match get_path_arg(args, kwargs, "path") {
        Ok(p) => p,
        Err(err) => return err,
    };
    if let Err(err) = check_policy(policy, "filesystem.read", &path) {
        return err;
    }
    let path = match resolve_path(policy, &path) {
        Ok(p) => p,
        Err(err) => return err,
    };
    match std::fs::read(&path) {
        Ok(bytes) => ExternalResult::Return(MontyObject::Bytes(bytes)),
        Err(err) => ExternalResult::Error(MontyException::new(
            ExcType::FileNotFoundError,
            Some(err.to_string()),
        )),
    }
}

fn write_text(
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    policy: &SafetyPolicy,
) -> ExternalResult {
    let path = match get_path_arg(args, kwargs, "path") {
        Ok(p) => p,
        Err(err) => return err,
    };
    let content = match get_string_arg(args, kwargs, "content", 1) {
        Ok(c) => c,
        Err(err) => return err,
    };
    if let Err(err) = check_policy(policy, "filesystem.write", &path) {
        return err;
    }
    let path = match resolve_path(policy, &path) {
        Ok(p) => p,
        Err(err) => return err,
    };
    match std::fs::write(&path, content) {
        Ok(()) => ExternalResult::Return(MontyObject::Bool(true)),
        Err(err) => ExternalResult::Error(MontyException::new(
            ExcType::OSError,
            Some(err.to_string()),
        )),
    }
}

fn write_bytes(
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    policy: &SafetyPolicy,
) -> ExternalResult {
    let path = match get_path_arg(args, kwargs, "path") {
        Ok(p) => p,
        Err(err) => return err,
    };
    let content = match get_bytes_arg(args, kwargs, "content", 1) {
        Ok(c) => c,
        Err(err) => return err,
    };
    if let Err(err) = check_policy(policy, "filesystem.write", &path) {
        return err;
    }
    let path = match resolve_path(policy, &path) {
        Ok(p) => p,
        Err(err) => return err,
    };
    match std::fs::write(&path, content) {
        Ok(()) => ExternalResult::Return(MontyObject::Bool(true)),
        Err(err) => ExternalResult::Error(MontyException::new(
            ExcType::OSError,
            Some(err.to_string()),
        )),
    }
}

fn list_dir(
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    policy: &SafetyPolicy,
) -> ExternalResult {
    let path = match get_path_arg(args, kwargs, "path") {
        Ok(p) => p,
        Err(err) => return err,
    };
    if let Err(err) = check_policy(policy, "filesystem.list", &path) {
        return err;
    }
    let path = match resolve_path(policy, &path) {
        Ok(p) => p,
        Err(err) => return err,
    };
    let mut entries = Vec::new();
    let read_dir = match std::fs::read_dir(&path) {
        Ok(rd) => rd,
        Err(err) => {
            return ExternalResult::Error(MontyException::new(
                ExcType::OSError,
                Some(err.to_string()),
            ))
        }
    };
    for entry in read_dir.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        entries.push(MontyObject::String(name));
        if entries.len() >= 200 {
            break;
        }
    }
    ExternalResult::Return(MontyObject::List(entries))
}

fn grep_text(
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    policy: &SafetyPolicy,
) -> ExternalResult {
    let path = match get_path_arg(args, kwargs, "path") {
        Ok(p) => p,
        Err(err) => return err,
    };
    let pattern = match get_string_arg(args, kwargs, "pattern", 1) {
        Ok(p) => p,
        Err(err) => return err,
    };
    if let Err(err) = check_policy(policy, "filesystem.read", &path) {
        return err;
    }
    let path = match resolve_path(policy, &path) {
        Ok(p) => p,
        Err(err) => return err,
    };
    let content = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) => {
            return ExternalResult::Error(MontyException::new(
                ExcType::FileNotFoundError,
                Some(err.to_string()),
            ))
        }
    };
    let mut matches = Vec::new();
    for line in content.lines() {
        if line.contains(&pattern) {
            matches.push(MontyObject::String(line.to_string()));
            if matches.len() >= 200 {
                break;
            }
        }
    }
    ExternalResult::Return(MontyObject::List(matches))
}

fn get_path_arg(
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    name: &str,
) -> Result<String, ExternalResult> {
    if let Some(arg) = args.first() {
        return object_to_string(arg).map_err(ExternalResult::Error);
    }
    for (k, v) in kwargs {
        if object_to_string(k)
            .map(|s| s == name)
            .unwrap_or(false)
        {
            return object_to_string(v).map_err(ExternalResult::Error);
        }
    }
    Err(ExternalResult::Error(MontyException::new(
        ExcType::TypeError,
        Some(format!("missing required argument: {name}")),
    )))
}

fn get_string_arg(
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    name: &str,
    pos_index: usize,
) -> Result<String, ExternalResult> {
    if let Some(arg) = args.get(pos_index) {
        return object_to_string(arg).map_err(ExternalResult::Error);
    }
    for (k, v) in kwargs {
        if object_to_string(k)
            .map(|s| s == name)
            .unwrap_or(false)
        {
            return object_to_string(v).map_err(ExternalResult::Error);
        }
    }
    Err(ExternalResult::Error(MontyException::new(
        ExcType::TypeError,
        Some(format!("missing required argument: {name}")),
    )))
}

fn get_bytes_arg(
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    name: &str,
    pos_index: usize,
) -> Result<Vec<u8>, ExternalResult> {
    if let Some(arg) = args.get(pos_index) {
        return object_to_bytes(arg).map_err(ExternalResult::Error);
    }
    for (k, v) in kwargs {
        if object_to_string(k)
            .map(|s| s == name)
            .unwrap_or(false)
        {
            return object_to_bytes(v).map_err(ExternalResult::Error);
        }
    }
    Err(ExternalResult::Error(MontyException::new(
        ExcType::TypeError,
        Some(format!("missing required argument: {name}")),
    )))
}

fn object_to_string(value: &MontyObject) -> Result<String, MontyException> {
    match value {
        MontyObject::String(s) => Ok(s.clone()),
        MontyObject::Path(p) => Ok(p.clone()),
        _ => Err(MontyException::new(
            ExcType::TypeError,
            Some(format!("expected string, got {}", value.type_name())),
        )),
    }
}

fn object_to_bytes(value: &MontyObject) -> Result<Vec<u8>, MontyException> {
    match value {
        MontyObject::Bytes(b) => Ok(b.clone()),
        MontyObject::String(s) => Ok(s.as_bytes().to_vec()),
        _ => Err(MontyException::new(
            ExcType::TypeError,
            Some(format!("expected bytes, got {}", value.type_name())),
        )),
    }
}

fn check_policy(policy: &SafetyPolicy, tool_name: &str, path: &str) -> Result<(), ExternalResult> {
    let call = ToolCall {
        id: 0,
        name: tool_name.to_string(),
        args: json!({ "path": path }),
    };
    let decision = policy.classify(&call);
    if !decision.allowed {
        return Err(ExternalResult::Error(MontyException::new(
            ExcType::OSError,
            decision.reason.or_else(|| Some("operation denied".to_string())),
        )));
    }
    Ok(())
}

fn resolve_path(policy: &SafetyPolicy, raw: &str) -> Result<std::path::PathBuf, ExternalResult> {
    let path = std::path::PathBuf::from(raw);
    let root = &policy.workspace_root;
    let resolved = if path.is_absolute() {
        path
    } else {
        root.join(path)
    };
    if !resolved.starts_with(root) {
        return Err(ExternalResult::Error(MontyException::new(
            ExcType::OSError,
            Some("path outside workspace".to_string()),
        )));
    }
    Ok(resolved)
}

fn truncate_text(text: &str, max_chars: usize) -> String {
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
