#!/usr/bin/env python3
import json
import os
import sys
from typing import Any, Dict


def emit(ok: bool, stdout: str = "", stderr: str = "", error: str = None) -> None:
    payload = {
        "ok": ok,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
    }
    sys.stdout.write(json.dumps(payload))


def read_call() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("empty stdin")
    return json.loads(raw)


def workspace_root() -> str:
    root = os.environ.get("VARCTX_WORKSPACE_ROOT", os.getcwd())
    return os.path.abspath(root)


def resolve_path(path: str, root: str) -> str:
    if os.path.isabs(path):
        full = path
    else:
        full = os.path.join(root, path)
    full = os.path.abspath(full)
    root_norm = os.path.abspath(root)
    if full != root_norm and not full.startswith(root_norm + os.sep):
        raise ValueError("path outside workspace root")
    return full


def tool_read(args: Dict[str, Any], root: str) -> str:
    path = resolve_path(args.get("path", ""), root)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def tool_list(args: Dict[str, Any], root: str) -> str:
    path = resolve_path(args.get("path", ""), root)
    if not os.path.isdir(path):
        raise ValueError("path is not a directory")
    items = sorted(os.listdir(path))
    return "\n".join(items)


def tool_write(args: Dict[str, Any], root: str) -> str:
    path = resolve_path(args.get("path", ""), root)
    content = args.get("content", "")
    append = bool(args.get("append", False))
    os.makedirs(os.path.dirname(path) or root, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as handle:
        handle.write(content)
    return "ok"


def tool_grep(args: Dict[str, Any], root: str) -> str:
    path = resolve_path(args.get("path", ""), root)
    pattern = args.get("pattern", "")
    if not pattern:
        raise ValueError("pattern required")
    case_sensitive = bool(args.get("case_sensitive", False))
    if not os.path.isfile(path):
        raise ValueError("path is not a file")
    matches = []
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            hay = line if case_sensitive else line.lower()
            needle = pattern if case_sensitive else pattern.lower()
            if needle in hay:
                matches.append(f"{idx}:{line.rstrip()}")
    return "\n".join(matches)


def main() -> None:
    try:
        call = read_call()
        name = call.get("name") or call.get("tool") or ""
        args = call.get("args") or {}
        root = workspace_root()

        if name == "filesystem.read":
            out = tool_read(args, root)
        elif name == "filesystem.list":
            out = tool_list(args, root)
        elif name == "filesystem.write":
            out = tool_write(args, root)
        elif name == "filesystem.grep":
            out = tool_grep(args, root)
        else:
            raise ValueError(f"unknown tool: {name}")

        emit(True, stdout=out)
    except Exception as exc:
        emit(False, stderr=str(exc), error=str(exc))


if __name__ == "__main__":
    main()
