#!/usr/bin/env python3
"""Assert every flag in deploy/backend-flags.yaml is exposed by the
pinned vLLM and SGLang images, and by the home-built vLLM.

Run by `make verify-backend-flags` after bumping either image. Fails
fast if any flag has been renamed or removed upstream — emits a
pointer to deploy/backend-flags.yaml so the operator can update the
single source of truth without chasing references through the router
and probers.

The home-built image (VLLM_DEVAI_IMAGE, the vllm-devai backend; plan
Phase 4 step 4v) is checked against the same vLLM flag names PLUS the
flag names of every recovery entry scoped to that backend -- curated
deploy/recovery-flags.json and generated deploy/recovery-flags.generated.json,
gated exactly as a launch gates them (_probe_hf_common
recovery_entries_for_backend). Those flags are appended to every launch of
their model, so a name the build no longer accepts stops that model at
startup. An entry pinning its own `image` launches there instead and is
listed, not checked.

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
# Same default as the Makefile's VLLM_DEVAI_IMAGE (exported there).
VLLM_DEVAI_IMAGE = os.environ.get(
    "VLLM_DEVAI_IMAGE", "docker.io/devai/vllm-devai:latest"
)
VLLM_DEVAI_BACKEND = "vllm-devai"
RUNTIME = os.environ.get("CONTAINER_RUNTIME", "podman")


def _help_text(image: str, args: list[str], gpu: bool,
               help_arg: str = "--help") -> str:
    """Run an image with `--help` and return the captured output.

    vLLM requires GPU device exposure to import its CUDA stack before
    arg-parsing — without --device the CLI faults during parser
    construction. SGLang's launch_server has no such dependency.
    `help_arg` "--help=all" asks vLLM's FlexibleArgumentParser for every
    argument with all its option strings (0.28.0 and 0.30.0
    vllm/utils/argparse_utils.py); a plain --help prints the usage line and
    the config-group titles only.
    """
    cmd = [RUNTIME, "run", "--rm", "--entrypoint", "python3"]
    if gpu:
        cmd.extend(["--device", "nvidia.com/gpu=all"])
    cmd.append(image)
    cmd.extend(args)
    cmd.append(help_arg)
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


def _flag_name(token: str) -> str | None:
    """`--flag` or `--flag=value` -> `--flag`; a value token -> None."""
    if not token.startswith("--") or token == "--":
        return None
    return token.split("=", 1)[0]


def recovery_flags(entries: list[tuple[str, str, list[str], str | None]]
                   ) -> tuple[dict[str, str], list[str]]:
    """({check key: flag name}, [skipped entries]) for the recovery entries
    of one backend (as _probe_hf_common.recovery_entries_for_backend
    returns them). Flag NAMES only, never values; one key per distinct
    flag, `recovery<flag>[<models>]`, naming every model that appends it. An entry with its own
    `image` launches on that image, so its flags are not this image's to
    check: it is listed in the second element instead."""
    users: dict[str, set[str]] = {}
    skipped: list[str] = []
    for model, origin, flags, image in entries:
        if image:
            skipped.append(f"{model} ({origin}; launches on {image})")
            continue
        for token in flags:
            name = _flag_name(token)
            if name:
                users.setdefault(name, set()).add(f"{model} ({origin})")
    # The key carries the flag: two flags appended by the same models must
    # not collapse into one check.
    return ({f"recovery{flag}[{'; '.join(sorted(models))}]": flag
             for flag, models in users.items()}, skipped)


def check_vllm_devai(cfg: dict, help_text: str,
                     entries: list[tuple[str, str, list[str], str | None]]
                     ) -> tuple[list[str], list[str], int]:
    """(missing, skipped entries, number of flags checked) for the home-built
    vLLM: the pinned vLLM flag names, the vllm-devai-only names, and the
    vllm-devai recovery flags."""
    flags = {k: v for k, v in (cfg.get("vllm") or {}).items() if isinstance(v, str)}
    flags.update({k: v for k, v in (cfg.get("vllm-devai") or {}).items() if isinstance(v, str)})
    rec, skipped = recovery_flags(entries)
    pinned = set(flags.values())
    flags.update({k: v for k, v in rec.items() if v not in pinned})
    return _check_flags(VLLM_DEVAI_BACKEND, flags, help_text), skipped, len(flags)


def _vllm_devai_entries() -> list[tuple[str, str, list[str], str | None]]:
    """The recovery entries a vllm-devai launch applies (both files)."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _probe_hf_common import recovery_entries_for_backend
    return recovery_entries_for_backend(VLLM_DEVAI_BACKEND)


def main() -> None:
    with FLAGS_YAML.open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        print(f"[fatal] {FLAGS_YAML}: top-level must be a mapping",
              file=sys.stderr)
        sys.exit(2)

    print(f"verifying {FLAGS_YAML}")
    print(f"  VLLM_IMAGE       = {VLLM_IMAGE}")
    print(f"  SGLANG_IMAGE     = {SGLANG_IMAGE}")
    print(f"  VLLM_DEVAI_IMAGE = {VLLM_DEVAI_IMAGE}")
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

    # --- the home-built vLLM (vllm-devai) ---
    print(f"→ vLLM devai ({VLLM_DEVAI_IMAGE}) ...")
    devai_help = _help_text(
        VLLM_DEVAI_IMAGE, ["-m", "vllm.entrypoints.openai.api_server"],
        gpu=True, help_arg="--help=all",
    )
    devai_missing, devai_skipped, n_checked = check_vllm_devai(
        cfg, devai_help, _vllm_devai_entries())
    for entry in devai_skipped:
        print(f"  not checked here: {entry}")
    if devai_missing:
        print(f"  MISSING: {', '.join(devai_missing)}")
        failed.append(VLLM_DEVAI_BACKEND)
    else:
        print(f"  OK ({n_checked} flags present, recovery flags included)")

    if failed:
        print()
        print(f"[fail] {len(failed)} backend(s) have flag drift: "
              f"{', '.join(failed)}")
        print(f"  edit {FLAGS_YAML} to match the pinned image's --help "
              f"output, then update consumers; a missing recovery[...] flag "
              f"is a deploy/recovery-flags*.json entry to fix.")
        sys.exit(1)

    print()
    print("all flags verified.")


if __name__ == "__main__":
    main()
