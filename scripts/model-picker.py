#!/usr/bin/env python3
"""Two-step picker for DevAI: pick a downloaded model, then pick an agent.

Discovery is disk-only — the picker walks the host model cache that is
mounted into the container and lists exactly what is on disk. The yaml
catalog (deploy/models.yaml) is consulted only to enrich display metadata
(purpose). It is never used to decide whether a model is "downloaded",
and the picker performs no VRAM-fit math: that gate happens upstream in
`select-models.py` before a model ever reaches disk.

Layout scanned:
    OLLAMA_MANIFESTS_DIR/<library>/<tag>          → ollama tag <library>:<tag>
    VLLM_MODELS_DIR/<name>/config.json            → HF model <name>

Usage:
    model-picker                  Step 1: model · Step 2: agent
    model-picker --agent claude   Pre-select agent, only show step 1
    model-picker --agent gemini   Shortcut: launch gemini directly
    model-picker --agent bash     Shortcut: drop to bash immediately

Errors propagate verbatim. No exception swallowing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


# ── Configuration ────────────────────────────────────────────────────────────

_CATALOG_PATHS = [
    "/etc/devai/models.yaml",
    str(Path(__file__).resolve().parent.parent / "deploy" / "models.yaml"),
]

# Active catalog with precomputed VRAM breakdown (written by select-models.py).
# We read this for display only — never to decide what's downloaded.
_ACTIVE_CATALOG_PATHS = [
    "/etc/devai/active-models.yaml",
    str(Path(__file__).resolve().parent.parent / "deploy" / "active-models.yaml"),
]

_ROUTER = os.environ.get("DEVAI_ROUTER_HOST", "devai-router")
_VLLM_DIR = os.environ.get("VLLM_MODELS_DIR", "/var/cache/devai/ollama/models/vllm")
_OLLAMA_MANIFESTS = os.environ.get(
    "OLLAMA_MANIFESTS_DIR",
    "/var/cache/devai/ollama/models/manifests/registry.ollama.ai/library",
)
# Default backend per source. Ollama-source models are ollama-only.
# HF-source models can run on vllm or sglang; we pick vllm by default.
# Override via env DEVAI_HF_BACKEND=sglang.
_HF_BACKEND = os.environ.get("DEVAI_HF_BACKEND", "vllm")

#                     label        reason                                              port
_BACKENDS: dict[str, tuple[str, str, int]] = {
    "ollama": ("Ollama", "GGUF quantized — wide compatibility, CPU+GPU",              11434),
    "vllm":   ("vLLM",   "NVFP4 tensor cores — high throughput, paged attention",     11435),
    "sglang": ("SGLang", "NVFP4 tensor cores — RadixAttention, multi-turn optimized", 11436),
}

#                  id             display name          description
_AGENTS: list[tuple[str, str, str]] = [
    ("claude",      "Claude Code",       "AI coding assistant with agentic terminal"),
    ("aider",       "Aider",             "Git-aware pair programming"),
    ("codex",       "Codex",             "OpenAI terminal coding agent"),
    ("late",        "LATE",              "Lightweight AI Terminal Environment — ephemeral subagents"),
    ("interpreter", "Open Interpreter",  "Natural language computer control"),
]

# ANSI helpers
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


# ── Catalog (display metadata only) ──────────────────────────────────────────

def _load_catalog() -> dict[str, dict]:
    """Optional metadata lookup. Returns {} if no yaml found — discovery
    still works, rows just lose declared size / purpose / arch."""
    for path in _CATALOG_PATHS:
        if os.path.isfile(path):
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
            return {m["name"]: m for m in data.get("models", [])}
    return {}


def _load_active_breakdowns() -> dict[str, dict]:
    """Lookup of name → precomputed VRAM breakdown (weights/KV/overhead/total)
    written by scripts/select-models.py into active-models.yaml.

    Returns {} if the file isn't mounted or doesn't have `vram` blocks yet.
    """
    for path in _ACTIVE_CATALOG_PATHS:
        if os.path.isfile(path):
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
            return {
                m["name"]: m["vram"]
                for m in data.get("models", []) or []
                if isinstance(m, dict) and m.get("vram")
            }
    return {}


# ── Disk discovery ───────────────────────────────────────────────────────────

def _ollama_disk_size_gb(library: str, tag: str) -> float:
    """Read ollama manifest, sum layer sizes."""
    manifest = Path(_OLLAMA_MANIFESTS) / library / tag
    try:
        data = json.loads(manifest.read_text())
        total = sum(int(L.get("size", 0)) for L in data.get("layers", []))
        return total / (1024 ** 3)
    except (OSError, ValueError, KeyError):
        return 0.0


def _hf_disk_size_gb(model_dir: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(model_dir):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 ** 3)


def _discover_models() -> list[dict]:
    """Walk the cache dirs and return one entry per model on disk.

    Disk is the source of truth for existence. Catalog metadata (purpose,
    declared size) and the precomputed VRAM breakdown are layered on top
    when available — never recomputed here.
    """
    catalog = _load_catalog()
    breakdowns = _load_active_breakdowns()
    out: list[dict] = []

    # Ollama: <library>/<tag> manifest files
    base = Path(_OLLAMA_MANIFESTS)
    if base.is_dir():
        for lib_dir in sorted(base.iterdir()):
            if not lib_dir.is_dir():
                continue
            for tag_file in sorted(lib_dir.iterdir()):
                if not tag_file.is_file():
                    continue
                name = f"{lib_dir.name}:{tag_file.name}"
                meta = catalog.get(name, {})
                disk_gb = _ollama_disk_size_gb(lib_dir.name, tag_file.name)
                out.append({
                    "name": name,
                    "source": "ollama",
                    "backend": "ollama",
                    "size": meta.get("size") or f"{disk_gb:.2f} GB",
                    "purpose": meta.get("purpose", ""),
                    "vram": breakdowns.get(name),
                })

    # HF/vLLM/SGLang: <name>/config.json
    vbase = Path(_VLLM_DIR)
    if vbase.is_dir():
        for d in sorted(vbase.iterdir()):
            if not (d.is_dir() and (d / "config.json").is_file()):
                continue
            name = d.name
            meta = catalog.get(name, {})
            disk_gb = _hf_disk_size_gb(d)
            # HF models can run on vllm or sglang; default per env.
            backend = _HF_BACKEND if _HF_BACKEND in ("vllm", "sglang") else "vllm"
            out.append({
                "name": name,
                "source": "hf",
                "backend": backend,
                "size": meta.get("size") or f"{disk_gb:.2f} GB",
                "purpose": meta.get("purpose", ""),
                "vram": breakdowns.get(name),
            })

    return out


def _check_cache_visible() -> None:
    """Bail early with a clear message if the cache mount is missing."""
    if not (os.path.isdir(_OLLAMA_MANIFESTS) or os.path.isdir(_VLLM_DIR)):
        sys.exit(
            f"error: model cache not visible inside the container.\n"
            f"  Looked for: {_OLLAMA_MANIFESTS}\n"
            f"          and {_VLLM_DIR}\n"
            f"  Re-run via `make shell-gpu` / `make lab-gpu` (or equivalent CPU "
            f"target) so the host cache dir is mounted read-only."
        )


# ── fzf wrapper ──────────────────────────────────────────────────────────────

def _fzf(lines: list[str], header: str) -> int | None:
    """Run fzf; return selected index or None on cancel."""
    indexed = [f"{i}\t{line}" for i, line in enumerate(lines)]
    try:
        result = subprocess.run(
            [
                "fzf", "--reverse", "--no-sort", "--ansi", "--no-info",
                "--delimiter", "\t", "--with-nth", "2..",
                "--header", header,
                "--pointer", "▶", "--prompt", "  ",
                "--margin", "1,2",
                "--color", "pointer:3,header:4,hl:3,hl+:3,gutter:-1",
            ],
            input="\n".join(indexed),
            stdout=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return _numbered_fallback(lines, header)
    if result.returncode != 0:
        return None
    return int(result.stdout.strip().split("\t", 1)[0])


def _numbered_fallback(lines: list[str], header: str) -> int | None:
    print(f"\n  {header}\n")
    for i, line in enumerate(lines, 1):
        print(f"  [{i}] {line}")
    print()
    idx = int(input(f"  Select [1-{len(lines)}]: ")) - 1
    if 0 <= idx < len(lines):
        return idx
    return None


# ── Row formatters ───────────────────────────────────────────────────────────

def _format_model_row(m: dict) -> str:
    backend_label = _BACKENDS[m["backend"]][0]
    v = m.get("vram")
    if v:
        ctx_k = v["context"] // 1024
        vram_col = (
            f"w {v['weights_gb']:>5.2f}G  "
            f"kv {v['kv_gb']:>5.2f}G  "
            f"ctx {ctx_k:>3d}K  "
            f"= {_BOLD}{v['total_gb']:>5.2f}G{_RESET}"
        )
    elif m.get("purpose"):
        # In catalog but no breakdown yet — user just needs to refresh.
        vram_col = f"{_DIM}(no breakdown — run `make model-select`){_RESET}"
    else:
        # Not in catalog at all (orphan download); fall back to disk size.
        vram_col = f"w {m['size']:>10s}  {_DIM}(not in catalog){_RESET}"
    return (
        f" {_BOLD}{m['name']:<38s}{_RESET} "
        f"{backend_label:<7s}  "
        f"{vram_col}"
    )


def _format_agent_row(agent: tuple[str, str, str]) -> str:
    _aid, name, desc = agent
    return f" {_BOLD}{name:<22s}{_RESET} {_DIM}{desc}{_RESET}"


# ── Command builder ──────────────────────────────────────────────────────────

def _build(agent_id: str, model_name: str, backend: str) -> list[str]:
    _, _, port = _BACKENDS[backend]
    base = f"http://{_ROUTER}:{port}"

    if agent_id == "claude":
        if backend != "ollama":
            os.environ["ANTHROPIC_BASE_URL"] = base
        return ["claude", "--model", model_name]

    if agent_id == "aider":
        if backend == "ollama":
            return ["aider", "--model", f"ollama_chat/{model_name}"]
        return [
            "aider", "--model", f"openai/{model_name}",
            "--openai-api-base", f"{base}/v1",
            "--openai-api-key", "local",
        ]

    if agent_id == "codex":
        # Codex 0.124+ requires --oss for non-OpenAI endpoints (chat-completions
        # wire was removed). The built-in `ollama` provider is reserved and
        # hard-codes localhost, so we use custom providers `router-<backend>`
        # defined in ~/.codex/config.toml (seeded by the container entrypoint).
        # Trust for the default workdir is seeded in the same config; codex
        # will prompt once for any other cwd.
        os.environ.setdefault("OPENAI_API_KEY", "local")
        return [
            "codex",
            "--oss",
            "--local-provider", f"router-{backend}",
            "-c", f'model="{model_name}"',
        ]

    if agent_id == "late":
        os.environ["OPENAI_BASE_URL"] = f"{base}/v1"
        os.environ["OPENAI_API_KEY"] = "local"
        os.environ["OPENAI_MODEL"] = model_name
        cmd = ["late"]
        if model_name.startswith("Gemma-4-"):
            cmd.append("--gemma-thinking")
        return cmd

    if agent_id == "interpreter":
        if backend == "ollama":
            return ["interpreter", "--model", f"ollama/{model_name}"]
        return [
            "interpreter", "--model", f"openai/{model_name}",
            "--api-base", f"{base}/v1",
        ]

    sys.exit(f"error: unknown agent '{agent_id}'")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    agent_filter: str | None = None
    if "--agent" in sys.argv:
        pos = sys.argv.index("--agent")
        if pos + 1 < len(sys.argv):
            agent_filter = sys.argv[pos + 1]

    if agent_filter == "bash":
        os.execvp("bash", ["bash"])
    if agent_filter == "gemini":
        os.execvp("gemini", ["gemini"])

    _check_cache_visible()
    models = _discover_models()
    if not models:
        sys.exit(
            "error: no downloaded models found on disk.\n"
            f"  Scanned: {_OLLAMA_MANIFESTS}\n"
            f"       and {_VLLM_DIR}\n"
            "  Pull models first (e.g. `make ollama-pull`, `make vllm-pull`)."
        )

    # Step 1 — pick a model
    lines = [_format_model_row(m) for m in models]
    header = f"DevAI  ▸  Step 1/2: pick a model  ({len(models)} downloaded)"
    idx = _fzf(lines, header)
    if idx is None:
        os.execvp("bash", ["bash"])
    model = models[idx]

    # Step 2 — pick an agent (skipped when --agent was passed)
    if agent_filter:
        agent = next((a for a in _AGENTS if a[0] == agent_filter), None)
        if agent is None:
            sys.exit(
                f"error: unknown agent '{agent_filter}' "
                f"(known: {', '.join(a[0] for a in _AGENTS)})"
            )
    else:
        alines = [_format_agent_row(a) for a in _AGENTS]
        header = (
            f"Step 2/2: pick an agent for "
            f"{_BOLD}{model['name']}{_RESET}"
            f" via {_BACKENDS[model['backend']][0]}"
        )
        idx = _fzf(alines, header)
        if idx is None:
            os.execvp("bash", ["bash"])
        agent = _AGENTS[idx]

    cmd = _build(agent[0], model["name"], model["backend"])
    print(
        f"\n  {_BOLD}{agent[0]}{_RESET}"
        f" → {model['name']} via {_BACKENDS[model['backend']][0]}\n"
    )
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
