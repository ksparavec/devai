#!/usr/bin/env python3
"""Diff llmfit `recommend --json` output against scripts/model-families.yaml.

Reads llmfit JSON on stdin, prints a table of GGUF repos that llmfit
considers a good fit but are NOT yet referenced in any family's
`gguf_repos:` block.

This is a HINT. llmfit is a static heuristic; the probe cache is truth.
Treat suggestions as candidates worth running through `make probe` before
adding to model-families.yaml. Never wire this output into automatic edits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
FAMILIES_FILE = REPO_ROOT / "scripts" / "model-families.yaml"


def load_existing_gguf_repos() -> set[str]:
    """Return lowercased set of every gguf repo already in the catalog."""
    if not FAMILIES_FILE.exists():
        print(f"error: {FAMILIES_FILE} not found", file=sys.stderr)
        sys.exit(1)
    with FAMILIES_FILE.open() as f:
        data = yaml.safe_load(f)
    repos: set[str] = set()
    for fam in data.get("families", []) or []:
        for entry in fam.get("gguf_repos", []) or []:
            if isinstance(entry, dict):
                r = entry.get("repo")
            else:
                r = entry
            if r:
                repos.add(r.strip().lower())
    return repos


def load_existing_hf_repos() -> set[str]:
    """Return lowercased set of every hf_repos entry — useful when the
    llmfit candidate is a non-GGUF mirror of an HF repo we already track.
    """
    with FAMILIES_FILE.open() as f:
        data = yaml.safe_load(f)
    repos: set[str] = set()
    for fam in data.get("families", []) or []:
        for r in fam.get("hf_repos", []) or []:
            if r:
                repos.add(r.strip().lower())
    return repos


def family_hint(model_name: str, family_names: list[str]) -> str:
    """Best-effort: return the catalog family whose name is a substring
    of the model name (case-insensitive). Empty string if no match.
    """
    n = model_name.lower()
    for fam in family_names:
        if fam.lower() in n:
            return fam
    return ""


def main() -> int:
    raw = sys.stdin.read().strip()
    if not raw:
        print("error: no JSON on stdin. Pipe `llmfit recommend --json` here.",
              file=sys.stderr)
        return 1
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON on stdin ({exc})", file=sys.stderr)
        return 1

    models = payload.get("models", [])
    system = payload.get("system", {})

    with FAMILIES_FILE.open() as f:
        families = yaml.safe_load(f).get("families", []) or []
    family_names = [fam["name"] for fam in families if "name" in fam]

    existing_gguf = load_existing_gguf_repos()
    existing_hf = load_existing_hf_repos()

    candidates: list[dict] = []
    skipped_already_present = 0
    skipped_no_gguf = 0
    skipped_non_llamacpp = 0

    for m in models:
        if (m.get("runtime") or "").lower() != "llama.cpp":
            skipped_non_llamacpp += 1
            continue
        sources = m.get("gguf_sources") or []
        if not sources:
            skipped_no_gguf += 1
            continue
        new_repos = []
        for src in sources:
            repo = (src.get("repo") or "").strip()
            if not repo:
                continue
            if repo.lower() in existing_gguf:
                continue
            new_repos.append(repo)
        if not new_repos:
            skipped_already_present += 1
            continue

        candidates.append({
            "name": m.get("name"),
            "params": m.get("parameter_count"),
            "best_quant": m.get("best_quant"),
            "fit_level": m.get("fit_level"),
            "score": m.get("score"),
            "est_tps": m.get("estimated_tps"),
            "mem_gb": m.get("memory_required_gb"),
            "ctx": m.get("effective_context_length"),
            "use_case": m.get("use_case"),
            "caps": m.get("capabilities") or [],
            "new_gguf_repos": new_repos,
            "hint_family": family_hint(m.get("name") or "", family_names),
            "hf_already_tracked": (m.get("name") or "").lower() in existing_hf,
        })

    print(f"# llmfit catalog suggestions (read-only)")
    print(f"# system: {system.get('gpu_name','?')} "
          f"{system.get('gpu_vram_gb','?'):.2f}GB VRAM, "
          f"backend={system.get('backend','?')}")
    print(f"# llmfit returned {len(models)} ranked models; "
          f"{len(candidates)} have GGUF repos not in scripts/model-families.yaml")
    print(f"# skipped: {skipped_already_present} already-present, "
          f"{skipped_no_gguf} no GGUF source, "
          f"{skipped_non_llamacpp} non-llama.cpp runtime")
    print()

    if not candidates:
        print("# No new GGUF candidates. Catalog already covers llmfit's top picks.")
        return 0

    print(f"{'#':<3} {'model':<55} {'params':<7} {'quant':<10} "
          f"{'fit':<8} {'score':<6} {'tps':<6} {'mem':<6} "
          f"{'family-hint':<14} {'caps'}")
    print("-" * 140)
    for i, c in enumerate(candidates, 1):
        flag = "*" if c["hf_already_tracked"] else " "
        print(f"{i:<3}{flag}{c['name'][:54]:<55} "
              f"{(c['params'] or '?'):<7} "
              f"{(c['best_quant'] or '?'):<10} "
              f"{(c['fit_level'] or '?'):<8} "
              f"{(c['score'] or 0):<6.1f} "
              f"{(c['est_tps'] or 0):<6.1f} "
              f"{(c['mem_gb'] or 0):<6.2f} "
              f"{(c['hint_family'] or '-'):<14} "
              f"{','.join(c['caps']) or '-'}")
        for repo in c["new_gguf_repos"]:
            print(f"        + gguf_repo: {repo}")

    print()
    print("# legend:  * = HF mirror already tracked under family.hf_repos")
    print("#")
    print("# next steps for any candidate worth pursuing:")
    print("#   1. Sanity-check tag list: huggingface-cli scan-cache or HF web UI")
    print("#   2. Add `- repo: <repo>` (with `tag_prefix:` and `include:` quants)")
    print("#      under the matching family.gguf_repos in scripts/model-families.yaml")
    print("#   3. make catalog-regen   # refresh deploy/models.yaml")
    print("#   4. make model-pull FAMILY=<name>   # download the candidates")
    print("#   5. make probe           # write probe-cache truth (NOT optional)")
    print("#")
    print("# Reminder: llmfit is a static estimate. The probe-cache is the")
    print("# only authoritative source of fit. Do not skip step 5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
