#!/usr/bin/env python3
"""Write one catalog row's reviewed recipe flags into the generated recovery file.

`make recipe-apply NAME=<row>` (plan Phase 17a, I18, I25). Reads the row's
seed from deploy/recipe-seeds.yaml (`make fetch-vllm-recipes`, reviewed in
`git diff` first) and writes ONE group into
deploy/recovery-flags.generated.json:

    {"backends": ["vllm-devai"], "engine_flags": [...], "engine_env": {...},
     "source": "recipe", "reason": "<recipe path> at <commit>", "added": "<UTC>"}

Rules:

  - Only when the recipe's min_vllm_version admits the vLLM in service: the
    default VLLM_VERSION of scripts/build-vllm.sh (0.28.0 today; Phase 4 step
    4v moves the default). A `nightly` or missing minimum admits nothing.
  - Scoped to `vllm-devai`, the backend that runs that build, until Phase 4
    step 4e renames it to `vllm` (which also rewrites the groups written
    here). Backends match by engine, so a ["vllm"] group would also reach
    stock vLLM 0.22.1.
  - Only flags the argument builders never emit. The last flag wins, so a
    recipe flag the builders also emit (--max-model-len, --kv-cache-dtype,
    --gpu-memory-utilization, the parser flags, ...) would pin every launch
    of the model and override the launch settings (C2, I18). Such flags are
    REPORTED, not written. The excluded set is every flag in
    gpu-arbiter/testdata/argv-golden.txt except those a golden case injects
    as RecoveryFlags (only --enforce-eager, which stays writable), plus the
    speculative-decoding and sleep flags (EXTRA_EXCLUDED).
  - The recipe's environment is written as it is.
  - Re-applying a row replaces its existing `recipe` group in place; other
    groups (Phase 18's auto-fix ones) and other models are kept, in order.
    The file is written whole to a temporary name and renamed.

Removing a group (or the file) reverts exactly what it added; the curated
deploy/recovery-flags.json is never written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import _vllm_recipes  # noqa: E402

ARGV_GOLDEN = REPO_ROOT / "gpu-arbiter" / "testdata" / "argv-golden.txt"
DEFAULT_CATALOG = REPO_ROOT / "deploy" / "models.yaml"

# What a golden case injects as RecoveryFlags (gpu-arbiter/argv_golden_test.go,
# case "recovery-flags-last"). It appears in the golden file only because the
# case appends it, not because a builder emits it, so it stays writable.
# tests/python/test_vllm_recipes.py pins this against the Go source.
GOLDEN_RECOVERY_FLAGS = frozenset({"--enforce-eager"})

# Builder flags the golden corpus does not show (it renders no MTP case and
# predates sleep mode): vLLM's speculative config, SGLang's speculative
# flags, and the sleep flags (D15).
EXTRA_EXCLUDED = frozenset({
    "--speculative-config",
    "--speculative-algorithm",
    "--speculative-num-steps",
    "--speculative-num-draft-tokens",
    "--speculative-eagle-topk",
    "--speculative-draft-model-path",
    "--enable-sleep-mode",
    "--enable-memory-saver",
})

GENERATED_VERSION = 1

# Never written, whatever the recipe says: devai uses models as delivered,
# with no operator rope or config surgery (CLAUDE.md, "Use models as
# delivered -- no operator YaRN / rope_scaling edits").
REFUSED_FLAGS = frozenset({"--hf-overrides", "--rope-scaling"})

# Short spellings the engines accept for a long flag, compared as the long
# one. `-tp` is vLLM's --tensor-parallel-size; an alias missing here is
# still caught when it is a prefix of an excluded long flag (below).
FLAG_ALIASES = {"-tp": "--tensor-parallel-size"}

CURATED_FLAGS_PATH = REPO_ROOT / "deploy" / "recovery-flags.json"

# A token that starts with '-' but is a number is a value, not a flag.
_NUMERIC = re.compile(r"^-(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")


def golden_flags(path: Path = ARGV_GOLDEN) -> frozenset[str]:
    """Every flag token in the argv golden corpus (a JSON string element
    that starts with '-')."""
    flags = set()
    for line in path.read_text().splitlines():
        tok = line.strip().rstrip(",")
        if tok.startswith('"-'):
            try:
                flags.add(json.loads(tok))
            except ValueError:
                continue
    return frozenset(flags)


def excluded_flags(golden: Path = ARGV_GOLDEN) -> frozenset[str]:
    """Normalised names of every flag the argument builders emit."""
    return frozenset(normalise_flag(f) for f in
                     (golden_flags(golden) - GOLDEN_RECOVERY_FLAGS) | EXTRA_EXCLUDED)


def split_flags(tokens: list[str]) -> list[tuple[str, list[str]]]:
    """argv tokens -> [(flag, [values...])]. A token that starts with '-'
    (and is not a number such as `-1`) begins a flag; the tokens up to the
    next one are its values. A `--flag=value` token is one flag with its
    value attached. A leading token that is not a flag is kept as a flag of
    its own (and is then written as it is, never silently dropped)."""
    groups: list[tuple[str, list[str]]] = []
    for tok in tokens:
        if (tok.startswith("-") and not _NUMERIC.match(tok)) or not groups:
            groups.append((tok, []))
        else:
            groups[-1][1].append(tok)
    return groups


def normalise_flag(flag: str) -> str:
    """The name an engine resolves a flag token to, for comparison:
    `=value` and a `.subkey` suffix stripped (`--compilation-config.mode=3`,
    `-cc.pass_config.x=True`), `_` read as `-`, a known alias mapped to its
    long flag."""
    name = flag.split("=", 1)[0]
    dashes = "--" if name.startswith("--") else "-"
    body = name[len(dashes):].split(".", 1)[0].replace("_", "-")
    name = dashes + body
    return FLAG_ALIASES.get(name, name)


def flag_name(flag: str) -> str:
    return normalise_flag(flag)


def curated_flag_names(model: str, backend: str,
                       path: Path = CURATED_FLAGS_PATH) -> tuple[frozenset[str], frozenset[str]]:
    """(flag names, env keys) the model's curated entry already sets on
    `backend` (contract C2: absent, null or non-list `backends` applies
    everywhere; a list names backends or their engine, vllm-devai being a
    vLLM engine). Empty when the file or entry is missing or malformed."""
    try:
        entry = (json.loads(path.read_text()).get("models") or {}).get(model)
    except (OSError, ValueError, AttributeError):
        return frozenset(), frozenset()
    if not isinstance(entry, dict):
        return frozenset(), frozenset()
    backends = entry.get("backends")
    engine = "vllm" if backend == "vllm-devai" else backend
    if isinstance(backends, list) and backend not in backends and engine not in backends:
        return frozenset(), frozenset()
    flags = entry.get("engine_flags") if isinstance(entry.get("engine_flags"), list) else []
    env = entry.get("engine_env") if isinstance(entry.get("engine_env"), dict) else {}
    names = frozenset(normalise_flag(f) for f, _ in split_flags([str(t) for t in flags])
                      if f.startswith("-"))
    return names, frozenset(str(k) for k in env)


def why_not_written(flag: str, excluded: frozenset[str],
                    curated: frozenset[str] = frozenset()) -> str | None:
    """Why a recipe flag is reported instead of written, or None."""
    name = normalise_flag(flag)
    if name in REFUSED_FLAGS:
        return "models are used as delivered (no rope or config overrides)"
    if name in excluded:
        return "the argument builders emit it"
    if name in curated:
        return "the model's curated entry already sets it"
    if name.startswith("--"):
        for protected in sorted(excluded | REFUSED_FLAGS | curated):
            if protected.startswith(name) and protected != name:
                return f"an abbreviation of {protected}, which is not written"
    return None


def partition_flags(tokens: list[str], excluded: frozenset[str],
                    curated: frozenset[str] = frozenset(),
                    ) -> tuple[list[str], list[tuple[str, list[str], str]]]:
    """(tokens to write, [(flag, values, reason)] reported and not written)."""
    keep: list[str] = []
    reported: list[tuple[str, list[str], str]] = []
    for flag, values in split_flags(tokens):
        reason = why_not_written(flag, excluded, curated)
        if reason:
            reported.append((flag, values, reason))
        else:
            keep += [flag, *values]
    return keep, reported


def read_generated(path: Path) -> dict:
    """The generated file, or its empty form when missing. Refuses (raises
    SystemExit) a file that does not parse or has another version: writing
    over it would discard content this tool cannot read."""
    try:
        doc = json.loads(path.read_text())
    except FileNotFoundError:
        return {"version": GENERATED_VERSION, "models": {}}
    except ValueError as exc:
        raise SystemExit(f"error: {path} does not parse ({exc}); not writing over it")
    version = doc.get("version") if isinstance(doc, dict) else None
    if type(version) is not int or version != GENERATED_VERSION \
            or not isinstance(doc.get("models", {}), dict):
        raise SystemExit(f"error: {path} is not a version-{GENERATED_VERSION} "
                         f"generated recovery file; not writing over it")
    doc.setdefault("models", {})
    return doc


def write_generated_atomic(path: Path, doc: dict) -> None:
    """The whole file to a temporary name in the same directory, then a
    rename: a reader sees the old file or the new one, never a half."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(doc, indent=2) + "\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def place_group(groups: list, group: dict) -> tuple[list, str]:
    """(new group list, what happened). The row's existing recipe group is
    replaced in place; with none, the group is appended."""
    out = list(groups)
    for i, g in enumerate(out):
        if isinstance(g, dict) and g.get("source") == "recipe":
            same = {k: v for k, v in g.items() if k != "added"} == \
                   {k: v for k, v in group.items() if k != "added"}
            if same:
                return out, "unchanged"
            out[i] = group
            return out, "replaced"
    out.append(group)
    return out, "added"


def catalog_names(path: Path) -> set[str] | None:
    try:
        import yaml
        doc = yaml.safe_load(path.read_text()) or {}
    except Exception:  # noqa: BLE001 -- the check is advisory
        return None
    return {r.get("name") for r in doc.get("models") or [] if isinstance(r, dict)}


def apply(name: str, *, seeds_path: Path, generated_path: Path, golden: Path,
          vllm_version: str | None, backend: str, catalog: Path | None,
          dry_run: bool, now: dt.datetime | None = None,
          curated_path: Path = CURATED_FLAGS_PATH) -> int:
    seeds = _vllm_recipes.load_seeds(seeds_path)
    seed = seeds.get(name)
    if seed is None:
        names = catalog_names(catalog) if catalog else None
        where = "" if names is None or name in names else " (and no catalog row has that name)"
        print(f"error: no recipe seed for {name!r} in {seeds_path}{where}; "
              f"run `make fetch-vllm-recipes` and review the result first", file=sys.stderr)
        return 1
    if vllm_version is None:
        print(f"error: cannot read the vLLM version in service from "
              f"{_vllm_recipes.BUILD_VLLM_SH}; nothing written", file=sys.stderr)
        return 1
    minimum = seed.get("min_vllm_version")
    if not _vllm_recipes.version_admits(minimum, vllm_version):
        print(f"refused: {name}: the recipe needs vLLM >= {minimum!r}, the "
              f"vLLM in service is {vllm_version}; nothing written", file=sys.stderr)
        return 1

    flags = [str(t) for t in seed.get("engine_flags") or []]
    curated_flags, curated_env = curated_flag_names(name, backend, curated_path)
    keep, reported = partition_flags(flags, excluded_flags(golden), curated_flags)
    # The same for env: a key the curated entry sets would be overridden by
    # the group (later wins), so it is reported, not written.
    env, env_reported = {}, {}
    for k, v in (seed.get("engine_env") or {}).items():
        (env_reported if str(k) in curated_env else env)[str(k)] = str(v)

    print(f"{name}: recipe {seed.get('recipe')} at {str(seed.get('commit'))[:12]} "
          f"(min vLLM {minimum}, in service {vllm_version})")
    for flag, values, reason in reported:
        print(f"  reported, not written ({reason}): {' '.join([flag, *values])}")
    for k, v in env_reported.items():
        print(f"  reported, not written (the model's curated entry already sets it): {k}={v}")
    print(f"  flags to write: {' '.join(keep) if keep else '(none)'}")
    print(f"  env to write:   {json.dumps(env) if env else '(none)'}")
    if not keep and not env:
        print("  nothing to write; the generated file is unchanged")
        return 0

    now = now or dt.datetime.now(dt.timezone.utc)
    group = {
        "backends": [backend],
        "engine_flags": keep,
        "engine_env": env,
        "source": "recipe",
        "reason": f"vllm-project/recipes {seed.get('recipe')} at "
                  f"{str(seed.get('commit'))[:12]} ({seed.get('match')} match)",
        "added": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    doc = read_generated(generated_path)
    models = dict(doc["models"])
    groups, what = place_group(models.get(name) or [], group)
    if what == "unchanged":
        print(f"  already applied; {generated_path} unchanged")
        return 0
    if dry_run:
        print(f"  dry run: would have {what} the recipe group in {generated_path}")
        return 0
    models[name] = groups
    write_generated_atomic(generated_path, {**doc, "models": models})
    print(f"  {what} the recipe group for {name} (backend {backend}) in {generated_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True, help="catalog row name")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    ap.add_argument("--seeds", type=Path, default=_vllm_recipes.SEEDS_PATH)
    ap.add_argument("--generated", type=Path, default=_vllm_recipes.GENERATED_FLAGS_PATH)
    ap.add_argument("--golden", type=Path, default=ARGV_GOLDEN)
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = ap.parse_args(argv)
    # The gate reads the build script, never a command-line value.
    version = _vllm_recipes.service_vllm_version()
    return apply(args.name, seeds_path=args.seeds, generated_path=args.generated,
                 golden=args.golden, vllm_version=version,
                 backend=_vllm_recipes.SERVICE_BACKEND, catalog=args.catalog,
                 dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
