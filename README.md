# EquiCode / varctx_proto

Prototype for a variable-context store + DSPy-style assertions/retry + local LLM (llama.cpp).

## What it does
- Stores large context externally in chunks and binds them to variables.
- Retrieves small working sets for prompts.
- Runs assertion/suggestion retries with feedback (paper-inspired).
- Includes a minimal coding agent (plan -> answer) that can run on a local GGUF model.

## Quick start
1. Place a GGUF model in the repo root, or set `VARCTX_MODEL_PATH`.
2. Run tests:
   - `cargo test --lib`
3. Run the demo pipeline:
   - `cargo run --release`

## Coding agent
Example:

```
cargo run --bin coding_agent -- \
  --task "Refactor the store layer to add var summaries" \
  --context-file README.md
```

Using the context store + retrieval:

```
cargo run --bin coding_agent -- \
  --task "Refactor the store layer to add var summaries" \
  --store ./varctx_db \
  --vars V:demo_doc \
  --top-k 8
```

Update summaries automatically:

```
cargo run --bin coding_agent -- \
  --summarize-var V:demo_doc \
  --store ./varctx_db
```

Ingest a file and bind to a var:

```
cargo run --bin coding_agent -- \
  --ingest-file README.md \
  --bind-var V:demo_doc \
  --store ./varctx_db
```

If the store is corrupted, rebuild it during ingest:

```
cargo run --bin coding_agent -- \
  --ingest-file README.md \
  --bind-var V:demo_doc \
  --store ./varctx_db \
  --rebuild-store
```

Optional env vars for model config:
- `VARCTX_MODEL_PATH`
- `VARCTX_N_CTX`, `VARCTX_N_BATCH`, `VARCTX_N_UBATCH`
- `VARCTX_N_GPU_LAYERS`
- `VARCTX_TEMP`, `VARCTX_TOP_K`, `VARCTX_TOP_P`
- `VARCTX_USE_CHAT_TEMPLATE`, `VARCTX_SYSTEM_PROMPT`

## TUI (Claude Code style)
Run the interactive TUI:

```
cargo run --bin coding_agent_tui -- \
  --store ./varctx_db \
  --vars V:demo_doc \
  --top-k 8
```

Optional flags:
- `--mode yolo|confirm|paranoid` (tool approval policy)
- `--skills-dir ./skills` (enable `/skills` tools)
- `--allow-network` (enable network tools)

### Skills
This repo includes a minimal `/skills` example:
- `skills/filesystem` with `filesystem.read`, `filesystem.list`, `filesystem.write`, `filesystem.grep`.

The skill entrypoint is a subprocess that reads a JSON tool call from stdin and returns JSON on stdout.
It requires `python3` to be available on your system.

### Monty tool
The TUI always exposes a built-in `code.exec` tool powered by `pydantic/monty` (Rust).
It runs Monty code directly in-process without Python.

Monty helper functions (hosted by EquiCode, subject to safety policy):
- `read(path)`
- `write(path, content)`
- `list(path)`
- `grep(path, pattern)`
- `exists(path)`
