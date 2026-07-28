#!/usr/bin/env python3
"""Predicted-vs-curated parser report. READ-ONLY.

For every checkpoint on disk, print what `_card_hints` predicts from the
model's own chat template, what `scripts/model-families.yaml` curates, and
what the probe cache actually recorded -- with the evidence substring that
produced each prediction.

This exists to MEASURE the derivation before anything depends on it. The
discriminator rules were written after reading five downloaded
checkpoints, so a 5/5 match on those five is in-sample and proves nothing;
the value is in the disagreements and in families validated from fetched
metadata they were not fitted on.

Changes nothing. Launches nothing. Needs no GPU.

    make card-hints
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _card_hints import hints_for_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VLLM_DIR = Path(os.environ.get("VLLM_MODELS_DIR", "/var/cache/devai/vllm"))
SGLANG_DIR = Path(os.environ.get("SGLANG_MODELS_DIR", "/var/cache/devai/sglang"))
# Metadata-only checkouts staged by `make card-hints-fetch` (a few hundred
# KB per family, no weights). Under the USER cache, not /var/cache/devai --
# per CLAUDE.md's mount-point convention a new top-level directory there is
# not volume-backed.
FETCH_DIR = Path(
    os.environ.get("DEVAI_CARD_HINTS_DIR")
    or (Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
        / "devai" / "card-hints")
)


def _families() -> list[dict]:
    try:
        import yaml
    except ImportError:
        return []
    try:
        data = yaml.safe_load((REPO_ROOT / "scripts" / "model-families.yaml").read_text())
    except (OSError, Exception):  # noqa: BLE001
        return []
    return (data or {}).get("families") or []


def _curated_index() -> dict[str, dict]:
    """repo-substring -> curated parsers, so a model dir can be matched to
    the family that curates it."""
    out: dict[str, dict] = {}
    for fam in _families():
        parsers = (fam or {}).get("parsers") or {}
        if not parsers:
            continue
        for repo in (fam.get("hf_repos") or []):
            name = str(repo).split("/")[-1] if isinstance(repo, str) else ""
            if name:
                out[name] = {"family": fam.get("name"), "parsers": parsers}
        # arch_ref covers families whose members are listed elsewhere.
        ar = fam.get("arch_ref")
        if isinstance(ar, str) and ar:
            out.setdefault(ar.split("/")[-1],
                           {"family": fam.get("name"), "parsers": parsers})
    return out


def _probe_values(name: str) -> dict[str, dict]:
    """What each probe cache actually recorded for this model."""
    out: dict[str, dict] = {}
    for backend, fn in (("vllm", ".vllm-reasoning-cache.json"),
                        ("sglang", ".sglang-reasoning-cache.json")):
        try:
            data = json.loads((REPO_ROOT / "deploy" / fn).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for key, entry in data.items():
            if key.startswith("_") or not isinstance(entry, dict):
                continue
            if key.split("/")[-1].split("@")[0] != name:
                continue
            out[backend] = {
                "tool": entry.get("tool_parser"),
                "reasoning": entry.get("reasoning_parser"),
            }
            break
    return out


def _fmt(v) -> str:
    return "-" if v in (None, "") else str(v)


def _agree(pred, cur) -> str:
    if pred is None and cur is None:
        return "  "
    if pred == cur:
        return "OK"
    if pred is None:
        return "no-pred"
    if cur is None:
        return "no-cur"
    return "DIFFER"


def main() -> int:
    curated = _curated_index()

    dirs: list[Path] = []
    for base in (VLLM_DIR, SGLANG_DIR, FETCH_DIR):
        if base.is_dir():
            dirs.extend(sorted(p for p in base.iterdir() if p.is_dir()))
    # Deduplicate by basename -- the two stores are hard-linked copies.
    seen: set[str] = set()
    unique: list[Path] = []
    for d in dirs:
        if d.name in seen:
            continue
        seen.add(d.name)
        unique.append(d)

    if not unique:
        print("no checkpoints found. Looked in:")
        for b in (VLLM_DIR, SGLANG_DIR, FETCH_DIR):
            print(f"  {b}")
        print("\nRun `make card-hints-fetch` to stage metadata-only checkouts.")
        return 0

    matches = comparisons = 0
    print(f"card-hints: {len(unique)} checkpoint(s)\n")
    for d in unique:
        h = hints_for_model(d)
        cur = (curated.get(d.name) or {}).get("parsers") or {}
        fam = (curated.get(d.name) or {}).get("family")
        probed = _probe_values(d.name)

        src = h["template_source"] or "NONE"
        print(f"  {d.name}")
        print(f"    family={_fmt(fam)}  template={src} ({h['template_bytes']} bytes)")
        if not h["template_bytes"]:
            print("    (no chat template -- nothing to derive)\n")
            continue
        print(f"    tool format     : {_fmt(h['tool_format'])}"
              f"   evidence: {_fmt(h['tool_evidence'])}")
        print(f"    reasoning format: {_fmt(h['reasoning_format'])}"
              f"   evidence: {_fmt(h['reasoning_evidence'])}")

        for backend in ("vllm", "sglang"):
            cb = (cur.get(backend) or {})
            for kind, pred_map in (("tool", h["tool_parser"]),
                                   ("reasoning", h["reasoning_parser"])):
                pred = pred_map[backend]
                curval = cb.get(kind)
                pv = (probed.get(backend) or {}).get(kind)
                verdict = _agree(pred, curval)
                if verdict in ("OK", "DIFFER"):
                    comparisons += 1
                    if verdict == "OK":
                        matches += 1
                print(f"      {backend:6} {kind:9} pred={_fmt(pred):15}"
                      f" curated={_fmt(curval):15} probed={_fmt(pv):15} {verdict}")
        if h["sampling"]:
            print(f"    sampling: {h['sampling']}")
        print()

    if comparisons:
        print(f"agreement where both a prediction and a curated value exist: "
              f"{matches}/{comparisons}")
    print("\nREAD-ONLY: no launch argument was changed. Wiring derived parsers "
          "in as a fallback is Phase 3, gated on this result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
