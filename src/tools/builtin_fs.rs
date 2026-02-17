use anyhow::{anyhow, Result};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const DEFAULT_MAX_ENTRIES: usize = 200;

pub fn resolve_path_arg(args: &Value, root: &Path) -> Result<PathBuf> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing required field: path"))?;
    let raw = Path::new(path);
    let joined = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        root.join(raw)
    };
    let normalized = normalize_path(&joined);
    if !normalized.starts_with(root) {
        return Err(anyhow!("path outside workspace"));
    }
    Ok(normalized)
}

pub fn list_dir(args: &Value, root: &Path) -> Result<String> {
    let path = resolve_path_arg(args, root)?;
    let recursive = args.get("recursive").and_then(|v| v.as_bool()).unwrap_or(false);
    let max_entries = args
        .get("max_entries")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_MAX_ENTRIES);

    let mut entries = Vec::new();
    if recursive {
        for entry in WalkDir::new(&path).into_iter().filter_map(Result::ok) {
            if entry.path() == path {
                continue;
            }
            entries.push(format_entry(entry.path(), root));
            if entries.len() >= max_entries {
                break;
            }
        }
    } else {
        for entry in fs::read_dir(&path)? {
            let entry = entry?;
            entries.push(format_entry(&entry.path(), root));
            if entries.len() >= max_entries {
                break;
            }
        }
    }

    entries.sort();
    let mut out = entries.join("\n");
    if entries.len() >= max_entries {
        out.push_str(&format!("\n... truncated at {} entries", max_entries));
    }
    Ok(out)
}

pub fn read_file_at(path: &Path, args: &Value) -> Result<String> {
    let text = fs::read_to_string(path)?;
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    if total == 0 {
        return Ok(format!("FILE: {}\n(empty)\n", path.display()));
    }

    let mut start = 1usize;
    let mut end = total;

    let start_line = args.get("start_line").and_then(|v| v.as_u64()).map(|v| v as usize);
    let end_line = args.get("end_line").and_then(|v| v.as_u64()).map(|v| v as usize);
    let head = args.get("head").and_then(|v| v.as_u64()).map(|v| v as usize);
    let tail = args.get("tail").and_then(|v| v.as_u64()).map(|v| v as usize);

    if start_line.is_some() || end_line.is_some() {
        start = start_line.unwrap_or(1);
        end = end_line.unwrap_or(total);
    } else if let Some(n) = head {
        end = n.min(total);
    } else if let Some(n) = tail {
        if n >= total {
            start = 1;
        } else {
            start = total - n + 1;
        }
    }

    if start == 0 || start > end || end > total {
        return Err(anyhow!(
            "invalid line range: {}-{} (total {})",
            start,
            end,
            total
        ));
    }

    let mut out = String::new();
    out.push_str(&format!(
        "FILE: {}\nLINES: {}-{} / {}\n",
        path.display(),
        start,
        end,
        total
    ));
    for idx in start..=end {
        let line = lines.get(idx - 1).unwrap_or(&"");
        out.push_str(&format!("{:>4}| {}\n", idx, line));
    }
    Ok(out)
}

pub fn edit_file_at(path: &Path, args: &Value) -> Result<String> {
    let mut text = fs::read_to_string(path)?;
    let original_ends_with_newline = text.ends_with('\n');

    let has_range = args.get("start_line").is_some() || args.get("end_line").is_some();
    let has_replace = args.get("old").is_some() || args.get("new").is_some();

    if has_range {
        let start = args
            .get("start_line")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("start_line required for range edit"))? as usize;
        let end = args
            .get("end_line")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("end_line required for range edit"))? as usize;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("content required for range edit"))?;
        if start == 0 || end < start {
            return Err(anyhow!("invalid range: {}-{}", start, end));
        }
        let mut lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        if end > lines.len() {
            return Err(anyhow!(
                "range end {} exceeds file length {}",
                end,
                lines.len()
            ));
        }
        let new_lines: Vec<String> = content.split('\n').map(|l| l.to_string()).collect();
        lines.splice(start - 1..end, new_lines);
        text = lines.join("\n");
        if original_ends_with_newline {
            text.push('\n');
        }
        fs::write(path, text)?;
        return Ok(format!(
            "EDIT_OK: {}\nLINES: {}-{} replaced",
            path.display(),
            start,
            end
        ));
    }

    if has_replace {
        let old = args
            .get("old")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("old required for replace edit"))?;
        let new = args
            .get("new")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("new required for replace edit"))?;
        if old.is_empty() {
            return Err(anyhow!("old cannot be empty"));
        }
        let mut count = 0usize;
        let mut pos = 0usize;
        while let Some(idx) = text[pos..].find(old) {
            count += 1;
            pos += idx + old.len();
            if count > 1 {
                break;
            }
        }
        if count == 0 {
            return Err(anyhow!("old text not found"));
        }
        if count > 1 {
            return Err(anyhow!("old text matched multiple times"));
        }
        text = text.replacen(old, new, 1);
        fs::write(path, text)?;
        return Ok(format!(
            "EDIT_OK: {}\nREPLACED: 1 match",
            path.display()
        ));
    }

    Err(anyhow!(
        "edit requires either start_line/end_line/content or old/new"
    ))
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::RootDir => out.push(comp.as_os_str()),
            std::path::Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            std::path::Component::Normal(part) => out.push(part),
        }
    }
    out
}

fn format_entry(path: &Path, root: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let mut s = rel.display().to_string();
    if path.is_dir() {
        s.push('/');
    }
    s
}
