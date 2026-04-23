#!/usr/bin/env python3
"""Interactive (model × backend × agent) picker for DevAI.

Shows every valid combination in a single filterable fzf. Only sensible
rows appear: Ollama models pair only with :ollama, vLLM models only with
:vllm, etc. Each row is annotated with a VRAM-fit flag computed from the
in-repo vram-fit library using the current GPU_MEMORY_GB / MAX_CONTEXT_LEN
environment settings.

Usage:
    model-picker                  Full single-screen picker
    model-picker --agent claude   Pre-select agent, keep model+backend in picker
    model-picker --agent late     Same, for LATE
    model-picker --agent gemini   Shortcut: launch gemini directly
    model-picker --agent bash     Shortcut: drop to bash immediately

Errors propagate verbatim. No exception swallowing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

# Allow the picker to call helpers from the sibling vram-fit script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util as _importlib_util
_spec = _importlib_util.spec_from_file_location(
    "vram_fit", Path(__file__).resolve().parent / "vram-fit.py"
)
_vram = _importlib_util.module_from_spec(_spec)
sys.modules["vram_fit"] = _vram  # required so @dataclass can resolve the module
_spec.loader.exec_module(_vram)  # type: ignore[union-attr]


# ── Configuration ────────────────────────────────────────────────────────────

_CATALOG_PATHS = [
    "/etc/devai/models.yaml",
    str(Path(__file__).resolve().parent.parent / "deploy" / "models.yaml"),
]

_ROUTER = os.environ.get("DEVAI_ROUTER_HOST", "devai-router")
_VLLM_DIR = os.environ.get("VLLM_MODELS_DIR", "/var/cache/devai/ollama/models/vllm")
_VRAM_GB = float(os.environ.get("GPU_MEMORY_GB", "24"))
_CONTEXT = int(os.environ.get("MAX_CONTEXT_LEN", "131072"))
_KV_DTYPE = os.environ.get("KV_DTYPE", "fp16")

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
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


# ── Catalog ──────────────────────────────────────────────────────────────────

def _load() -> tuple[list[dict], dict]:
    for path in _CATALOG_PATHS:
        if os.path.isfile(path):
            with open(path) as fh:
                data = yaml.safe_load(fh)
            return data.get("models", []), data.get("defaults", {})
    sys.exit("error: models.yaml not found")


# ── Fit annotation ───────────────────────────────────────────────────────────

def _estimate_total_gb(model: dict, all_models: list[dict]) -> tuple[float, bool]:
    """Return (estimated_total_VRAM_gb, is_exact).
    is_exact=True when arch came from config.json, False when heuristic."""
    weight_gb = _vram.parse_size_gb(model["size"])
    arch = _vram.resolve_arch(model, all_models, Path(_VLLM_DIR))
    if arch:
        kv_gb = (arch.kv_per_token_bytes(_KV_DTYPE) * _CONTEXT) / (1024 ** 3)
        return weight_gb + kv_gb + _vram.OVERHEAD_GB, True
    return _vram.heuristic_vram(weight_gb, _CONTEXT, _KV_DTYPE), False


# ── Selection UI ─────────────────────────────────────────────────────────────

def _fzf(lines: list[str], header: str) -> int | None:
    """Run fzf; return selected index or None on cancel (ESC/ctrl-c)."""
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
        # Legitimate cancel (ESC/ctrl-c). Any other exception from here on
        # propagates verbatim to the caller.
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


# ── Command builders ─────────────────────────────────────────────────────────

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
        if backend != "ollama":
            os.environ["OPENAI_BASE_URL"] = f"{base}/v1"
            os.environ["OPENAI_API_KEY"] = "local"
        return ["codex", "--model", model_name]

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


# ── Row enumeration ──────────────────────────────────────────────────────────

def _enumerate_rows(
    models: list[dict], defaults: dict, agent_filter: str | None
) -> list[dict]:
    """Build the flat list of valid (model, backend, agent) combinations."""
    default_names = set(defaults.values())
    rows: list[dict] = []
    agents = [a for a in _AGENTS if (agent_filter is None or a[0] == agent_filter)]
    if agent_filter and not agents:
        sys.exit(f"error: unknown agent '{agent_filter}' (known: "
                 + ", ".join(a[0] for a in _AGENTS) + ")")
    for m in models:
        total_gb, exact = _estimate_total_gb(m, models)
        fits = total_gb <= _VRAM_GB
        for backend in m.get("backend", []):
            backend_default = defaults.get(backend) == m["name"]
            for aid, aname, _adesc in agents:
                rows.append({
                    "model": m["name"],
                    "backend": backend,
                    "agent_id": aid,
                    "agent_name": aname,
                    "model_size": m["size"],
                    "model_purpose": m["purpose"],
                    "total_gb": total_gb,
                    "fits": fits,
                    "exact": exact,
                    "default_model": m["name"] in default_names,
                    "default_pair": backend_default,
                })
    return rows


def _format_row(row: dict) -> str:
    fit_mark = (
        f"{_GREEN}✓{_RESET}" if row["fits"]
        else f"{_RED}✗{_RESET}"
    )
    approx = "~" if not row["exact"] else " "
    star = f"{_YELLOW}★{_RESET}" if row["default_pair"] else " "
    return (
        f" {fit_mark} "
        f"{_BOLD}{row['model']:<38s}{_RESET} "
        f"{_CYAN}{row['model_size']:>7s}{_RESET}  "
        f"{row['backend']:<7s}  "
        f"{_BOLD}{row['agent_name']:<18s}{_RESET} "
        f"{_DIM}{approx}{row['total_gb']:>5.1f}G{_RESET}  "
        f"{star}{row['model_purpose']}"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    agent_filter: str | None = None
    if "--agent" in sys.argv:
        pos = sys.argv.index("--agent")
        if pos + 1 < len(sys.argv):
            agent_filter = sys.argv[pos + 1]

    # Agents that skip model selection entirely.
    if agent_filter == "bash":
        os.execvp("bash", ["bash"])
    if agent_filter == "gemini":
        os.execvp("gemini", ["gemini"])

    models, defaults = _load()
    rows = _enumerate_rows(models, defaults, agent_filter)
    if not rows:
        sys.exit("error: no valid (model × backend × agent) combinations in catalog")

    lines = [_format_row(r) for r in rows]
    header = (
        f"DevAI · {_VRAM_GB:g}GB VRAM · {_CONTEXT // 1024}K ctx · KV={_KV_DTYPE} · "
        f"✓/✗ fit · ~=approx · ★=default  ▸  filter by model | backend | agent | size"
    )
    idx = _fzf(lines, header)
    if idx is None:
        # Clean user cancel → bash.
        os.execvp("bash", ["bash"])

    choice = rows[idx]
    cmd = _build(choice["agent_id"], choice["model"], choice["backend"])
    backend_label = _BACKENDS[choice["backend"]][0]
    fit_tag = "fits" if choice["fits"] else f"{_RED}OVER VRAM{_RESET}"
    print(
        f"\n  {_BOLD}{choice['agent_id']}{_RESET}"
        f" → {choice['model']} via {backend_label}"
        f"  ({choice['total_gb']:.1f}G est · {fit_tag})\n"
    )
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
