#!/usr/bin/env python3
"""Assert every flag in deploy/backend-flags.yaml is exposed by the
pinned vLLM and SGLang images.

Run by `make verify-backend-flags` after bumping either image. Fails
fast if any flag has been renamed or removed upstream — emits a
pointer to deploy/backend-flags.yaml so the operator can update the
single source of truth without chasing references through the router
and probers.

Errors propagate verbatim. No exception swallowing.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
FLAGS_YAML = REPO_ROOT / "deploy" / "backend-flags.yaml"

VLLM_IMAGE = os.environ.get(
    "VLLM_IMAGE", "docker.io/vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404"
)
SGLANG_IMAGE = os.environ.get(
    "SGLANG_IMAGE", "docker.io/lmsysorg/sglang:v0.5.16-cu130"
)
RUNTIME = os.environ.get("CONTAINER_RUNTIME", "podman")


def _help_text(image: str, args: list[str], gpu: bool) -> str:
    """Run an image with `--help` and return the captured output.

    vLLM requires GPU device exposure to import its CUDA stack before
    arg-parsing — without --device the CLI faults during parser
    construction. SGLang's launch_server has no such dependency.
    """
    cmd = [RUNTIME, "run", "--rm", "--entrypoint", "python3"]
    if gpu:
        cmd.extend(["--device", "nvidia.com/gpu=all"])
    cmd.append(image)
    cmd.extend(args)
    cmd.append("--help")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    # Both backends print --help to stdout but tolerate stderr for
    # the rare case where pkg-resources warnings break the contract.
    return (r.stdout or "") + "\n" + (r.stderr or "")


def _flag_present(flag: str, help_text: str) -> bool:
    """True iff `flag` appears in `help_text` as a WHOLE CLI token.

    A plain substring test silently passes a flag that upstream REMOVED
    whenever its name is a prefix of a surviving one: `--kv-cache-dtype`
    would "verify" against a help text that only offers
    `--kv-cache-dtype-foo`, and `--model` against `--model-path`. That
    defeats the entire purpose of this gate. Require the flag to be
    bounded by a non-[word/dash] character on both sides, so `--flag=X`,
    `--flag VALUE`, and `--flag, --no-flag` all still match.
    """
    return re.search(
        rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text) is not None


def _abbreviation_hint(flag: str, help_text: str) -> str:
    """If `flag` is a strict prefix of some advertised long option, name it.

    argparse resolves an unambiguous prefix, so such a flag still *works*
    today -- but only by accident: the moment upstream adds a second
    option with the same prefix, every launch breaks with 'ambiguous
    option'. And a renamed flag looks identical from here. Either way the
    operator needs the real spelling, so surface it.
    """
    candidates = sorted({
        m for m in re.findall(r"--[\w-]+", help_text)
        if m != flag and m.startswith(flag)
    })
    if not candidates:
        return ""
    return f" (not advertised; prefix of {', '.join(candidates)})"


def _check_flags(label: str, flags: dict, help_text: str) -> list[str]:
    """Return the list of flag names that are missing from help_text."""
    missing: list[str] = []
    for key, flag in sorted(flags.items()):
        if not isinstance(flag, str):
            continue
        if not _flag_present(flag, help_text):
            missing.append(f"{key}={flag}{_abbreviation_hint(flag, help_text)}")
    return missing


def main() -> None:
    with FLAGS_YAML.open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        print(f"[fatal] {FLAGS_YAML}: top-level must be a mapping",
              file=sys.stderr)
        sys.exit(2)

    print(f"verifying {FLAGS_YAML}")
    print(f"  VLLM_IMAGE   = {VLLM_IMAGE}")
    print(f"  SGLANG_IMAGE = {SGLANG_IMAGE}")
    print()

    failed: list[str] = []

    # --- vLLM ---
    print(f"→ vLLM ({VLLM_IMAGE}) ...")
    vllm_help = _help_text(
        VLLM_IMAGE, ["-m", "vllm.entrypoints.openai.api_server"], gpu=True
    )
    vllm_missing = _check_flags("vllm", cfg.get("vllm") or {}, vllm_help)
    if vllm_missing:
        print(f"  MISSING: {', '.join(vllm_missing)}")
        failed.append("vllm")
    else:
        print(f"  OK ({len(cfg.get('vllm') or {})} flags present)")

    # --- SGLang ---
    print(f"→ SGLang ({SGLANG_IMAGE}) ...")
    sglang_help = _help_text(
        SGLANG_IMAGE, ["-m", "sglang.launch_server"], gpu=False
    )
    sglang_missing = _check_flags("sglang", cfg.get("sglang") or {}, sglang_help)
    if sglang_missing:
        print(f"  MISSING: {', '.join(sglang_missing)}")
        failed.append("sglang")
    else:
        print(f"  OK ({len(cfg.get('sglang') or {})} flags present)")

    if failed:
        print()
        print(f"[fail] {len(failed)} backend(s) have flag drift: "
              f"{', '.join(failed)}")
        print(f"  edit {FLAGS_YAML} to match the pinned image's --help "
              f"output, then update consumers.")
        sys.exit(1)

    print()
    print("all flags verified.")


if __name__ == "__main__":
    main()
