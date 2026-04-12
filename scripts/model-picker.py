#!/usr/bin/env python3
"""Interactive model → backend → agent selection for DevAI.

Reads the model catalog from models.yaml and presents a three-step fzf-based
picker: model → backend → agent.  Pre-select an agent with --agent to skip
the third step.

Usage:
    model-picker                  Full: model → backend → agent
    model-picker --agent claude   Pre-select agent, pick model + backend
    model-picker --agent aider
    model-picker --agent codex
    model-picker --agent interpreter
    model-picker --agent bash     Shortcut: launch bash immediately
    model-picker --agent gemini   Shortcut: launch gemini immediately
"""

import os
import subprocess
import sys

import yaml

# ── Configuration ────────────────────────────────────────────────────────────

_CATALOG_PATHS = [
    "/etc/devai/models.yaml",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "deploy", "models.yaml"
    ),
]

_ROUTER = os.environ.get("DEVAI_ROUTER_HOST", "devai-router")

#                     label       reason                                              port
_BACKENDS = {
    "ollama": ("Ollama", "GGUF quantized — wide compatibility, CPU+GPU",              11434),
    "vllm":   ("vLLM",   "NVFP4 tensor cores — high throughput, paged attention",     11435),
    "sglang": ("SGLang",  "NVFP4 tensor cores — RadixAttention, multi-turn optimized", 11436),
}

#                  id             display name          description
_AGENTS = [
    ("claude",      "Claude Code",       "AI coding assistant with agentic terminal"),
    ("aider",       "Aider",             "Git-aware pair programming"),
    ("codex",       "Codex",             "OpenAI terminal coding agent"),
    ("interpreter", "Open Interpreter",  "Natural language computer control"),
    ("bash",        "Bash",              "Interactive shell, no agent"),
]

# ANSI helpers
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


# ── Catalog ──────────────────────────────────────────────────────────────────

def _load():
    """Return (models_list, defaults_dict) from the first models.yaml found."""
    for path in _CATALOG_PATHS:
        if os.path.isfile(path):
            with open(path) as fh:
                data = yaml.safe_load(fh)
            return data["models"], data.get("defaults", {})
    sys.exit("error: models.yaml not found")


# ── Selection UI ─────────────────────────────────────────────────────────────

def _pick(lines: list[str], header: str) -> int | None:
    """Show *lines* in fzf and return selected index, or None on cancel.

    Each line is prefixed with a tab-separated index that fzf hides via
    --with-nth.  The index survives regardless of ANSI-code handling.
    """
    indexed = [f"{i}\t{line}" for i, line in enumerate(lines)]
    try:
        result = subprocess.run(
            [
                "fzf",
                "--reverse",
                "--no-sort",
                "--ansi",
                "--no-info",
                "--delimiter",
                "\t",
                "--with-nth",
                "2..",
                "--header",
                header,
                "--pointer",
                "▶",
                "--prompt",
                "  ",
                "--margin",
                "1,2",
                "--color",
                "pointer:3,header:4,hl:3,hl+:3,gutter:-1",
            ],
            input="\n".join(indexed),
            stdout=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return _pick_fallback(lines, header)
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip().split("\t", 1)[0])
    except (ValueError, IndexError):
        return None


def _pick_fallback(lines: list[str], header: str) -> int | None:
    """Numbered menu when fzf is not available."""
    print(f"\n  {header}\n")
    for i, line in enumerate(lines, 1):
        print(f"  [{i}] {line}")
    print()
    try:
        idx = int(input(f"  Select [1-{len(lines)}]: ")) - 1
        if 0 <= idx < len(lines):
            return idx
    except (ValueError, EOFError, KeyboardInterrupt):
        pass
    return None


def _bail():
    """Exit gracefully to bash."""
    os.execvp("bash", ["bash"])


# ── Command builders ─────────────────────────────────────────────────────────

def _build(agent_id: str, model_name: str, backend: str) -> list[str]:
    """Return the argv list to exec for the given agent/model/backend."""
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
            "aider",
            "--model",
            f"openai/{model_name}",
            "--openai-api-base",
            f"{base}/v1",
            "--openai-api-key",
            "local",
        ]

    if agent_id == "codex":
        if backend != "ollama":
            os.environ["OPENAI_BASE_URL"] = f"{base}/v1"
            os.environ["OPENAI_API_KEY"] = "local"
        return ["codex", "--model", model_name]

    if agent_id == "interpreter":
        if backend == "ollama":
            return ["interpreter", "--model", f"ollama/{model_name}"]
        return [
            "interpreter",
            "--model",
            f"openai/{model_name}",
            "--api-base",
            f"{base}/v1",
        ]

    return ["bash"]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Parse --agent flag
    agent_id = None
    if "--agent" in sys.argv:
        pos = sys.argv.index("--agent")
        if pos + 1 < len(sys.argv):
            agent_id = sys.argv[pos + 1]

    # Short-circuit agents that don't need model selection
    if agent_id == "bash":
        _bail()
    if agent_id == "gemini":
        os.execvp("gemini", ["gemini"])

    models, defaults = _load()
    default_names = set(defaults.values())

    # ── Step 1: Select model ─────────────────────────────────────────────

    lines = []
    for m in models:
        star = f"{_YELLOW}★{_RESET}" if m["name"] in default_names else " "
        tags = ", ".join(m["backend"])
        lines.append(
            f"{star} {_BOLD}{m['name']:<35s}{_RESET} "
            f"{_CYAN}{m['size']:>7s}{_RESET}  "
            f"{m['purpose']}  "
            f"{_YELLOW}[{tags}]{_RESET}"
        )

    idx = _pick(lines, "Select Model  (type to filter)")
    if idx is None:
        _bail()
    model = models[idx]

    # ── Step 2: Select backend ───────────────────────────────────────────

    available = model["backend"]

    if len(available) == 1:
        backend = available[0]
    else:
        blines = []
        for b in available:
            label, reason, _ = _BACKENDS[b]
            star = (
                f"{_YELLOW}★{_RESET}"
                if defaults.get(b) == model["name"]
                else " "
            )
            blines.append(f"{star} {_BOLD}{label:<12s}{_RESET} {reason}")

        bidx = _pick(
            blines, f"Select Backend — {model['name']} ({model['size']})"
        )
        if bidx is None:
            _bail()
        backend = available[bidx]

    # ── Step 3: Select agent (unless pre-selected via --agent) ───────────

    if not agent_id:
        alines = []
        for aid, name, desc in _AGENTS:
            alines.append(f"  {_BOLD}{name:<20s}{_RESET} {desc}")

        backend_label = _BACKENDS[backend][0]
        aidx = _pick(
            alines, f"Select Agent — {model['name']} via {backend_label}"
        )
        if aidx is None:
            _bail()
        agent_id = _AGENTS[aidx][0]

    if agent_id == "bash":
        _bail()

    # ── Launch ───────────────────────────────────────────────────────────

    cmd = _build(agent_id, model["name"], backend)
    backend_label = _BACKENDS[backend][0]
    print(
        f"\n  {_BOLD}{agent_id}{_RESET}"
        f" → {model['name']} via {backend_label}\n"
    )
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
