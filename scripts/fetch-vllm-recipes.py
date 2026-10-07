#!/usr/bin/env python3
"""Fetch vLLM's published launch recipes for the catalog's HF rows.

`make fetch-vllm-recipes`. Plan Phase 17a (from odysseus CB-3): a model
starts probing from the vendor's recipe -- parsers, engine flags, environment
-- instead of from a bare command.

For every HF / derived row of deploy/models.yaml this looks up
github.com/vllm-project/recipes (`models/<org>/<repo>.yaml`), normalises the
recipe -- base args and env, the tool-calling and reasoning parser args, the
per-precision variants (VRAM minimum, extra flags and env), min_vllm_version
-- and writes deploy/recipe-seeds.yaml: one entry per row that has a recipe.

  - The probers read a seed's parsers as the THIRD step of the parser order
    (curated, card-derived, recipe, name heuristic, none), on vLLM only and
    only where min_vllm_version admits the backend's vLLM.
  - `make recipe-apply NAME=<row>` writes a reviewed row's engine flags into
    deploy/recovery-flags.generated.json.
  - The picker shows a badge for rows with a seed.

Matching a catalog row to a recipe, in order (recorded as `match`):
  exact      the row's repo IS the recipe's model (path, case-insensitive)
  variant    the row's repo is one of the recipe's `variants.*.model_id`
  base-name  the row's repo name minus quantisation / packaging suffixes
             (-NVFP4, -FP8, -MTP, -W4A16-AutoRound, -devai, ...) equals one
             recipe's name minus the same suffixes. Two recipes with the same
             stripped name match nothing.
A row that matches no recipe by name still gets a `variant` match when its
repo is a variant of a recipe loaded for another row.
A derived row (`make model-prepare`) is matched through its source's repo.

Engine flags and env are the recipe's base args and env, plus the variant's
extra args and env when the row IS that variant's checkpoint (exact: the
`default` variant; variant: that one). A base-name match takes the base args
alone: a variant's extras describe its own checkpoint (some name Ascend NPU
settings), not another publisher's quantisation of the same model. Its
`vram_minimum_gb` is still read from the recipe's variants of the row's
precision (NVFP4 -> nvfp4, FP8 -> fp8, ...), the smallest one.

Offline-first, never a failure: the recipe index (the repository's commit
and its file list) is cached for --max-age-hours (12 h) under
~/.cache/devai/vllm-recipes (--cache-dir or DEVAI_VLLM_RECIPES_CACHE), and a
recipe file is cached by commit, so it never goes stale. When the network is
unreachable a stale cache is used; with no cache at all nothing is seeded --
an existing deploy/recipe-seeds.yaml is then left as it is, and a missing one
is written in its empty form. The exit status is 0 in every such case.

Reads only the catalog and the network; writes only the cache and --out.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import _vllm_recipes  # noqa: E402

RECIPES_REPO = "vllm-project/recipes"
RECIPES_BRANCH = "main"
RECIPES_URL = f"https://github.com/{RECIPES_REPO}"
API = f"https://api.github.com/repos/{RECIPES_REPO}"
RAW = f"https://raw.githubusercontent.com/{RECIPES_REPO}"
DEFAULT_CATALOG = REPO_ROOT / "deploy" / "models.yaml"
DEFAULT_MAX_AGE_HOURS = 12.0
FETCH_ATTEMPTS = 3
FETCH_RETRY_DELAY_S = 2.0
FETCH_TIMEOUT_S = 20.0

# A recipe path from the repository's tree is untrusted data: accept only
# models/<org>/<repo>.yaml with plain name characters, so it can never name
# a file outside the cache directory.
RECIPE_PATH_RE = re.compile(r"^models/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.yaml$")
PARSER_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

# Trailing name tokens that say how a checkpoint was quantised or packaged,
# not which model it is. Stripped from both sides before a base-name match.
PACKAGING_TOKENS = frozenset({
    "nvfp4", "fp4", "mxfp4", "fp8", "bf16", "fp16", "int4", "int8",
    "w4a16", "w8a8", "w4a4", "autoround", "awq", "gptq", "dynamic",
    "mtp", "text", "devai",
})

# Row precision, from its name tokens -> the recipe variant precisions that
# describe it. A row naming none of these is matched to the `default` variant.
PRECISION_TOKENS: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    (frozenset({"mxfp4"}), ("mxfp4",)),
    (frozenset({"nvfp4", "fp4"}), ("nvfp4", "fp4")),
    (frozenset({"fp8"}), ("fp8",)),
    (frozenset({"w4a16", "autoround", "int4"}), ("int4",)),
    (frozenset({"awq"}), ("awq",)),
    (frozenset({"gptq"}), ("gptq",)),
    (frozenset({"int8", "w8a8"}), ("int8",)),
    (frozenset({"bf16"}), ("bf16",)),
)

Fetch = Callable[[str, str | None], bytes]


# ── Network and cache ────────────────────────────────────────────────────────

def http_get(url: str, accept: str | None = None) -> bytes:
    """GET with FETCH_ATTEMPTS tries; raises the last error."""
    headers = {"User-Agent": "devai-fetch-vllm-recipes"}
    if accept:
        headers["Accept"] = accept
    last: Exception | None = None
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
            if attempt < FETCH_ATTEMPTS:
                time.sleep(FETCH_RETRY_DELAY_S)
    assert last is not None
    raise last


def default_cache_dir() -> Path:
    env = os.environ.get("DEVAI_VLLM_RECIPES_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "devai" / "vllm-recipes"


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def fetch_index(fetch: Fetch) -> dict:
    """The recipes repository's current commit and its recipe paths."""
    commit = fetch(f"{API}/commits/{RECIPES_BRANCH}", "application/vnd.github.sha").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"unexpected commit id from GitHub: {commit[:60]!r}")
    tree = json.loads(fetch(f"{API}/git/trees/{commit}?recursive=1", "application/vnd.github+json"))
    if tree.get("truncated"):
        print("warning: GitHub truncated the recipe tree listing; some recipes "
              "may be missing", file=sys.stderr)
    paths = sorted(item["path"] for item in tree.get("tree", [])
                   if item.get("type") == "blob"
                   and RECIPE_PATH_RE.match(str(item.get("path", ""))))
    return {"commit": commit, "fetched_at": time.time(), "paths": paths}


def load_index(cache_dir: Path, *, fetch: Fetch, max_age_s: float,
               offline: bool, refresh: bool) -> dict | None:
    """The index from the cache when fresh, else from the network, else a
    stale cache, else None. Never raises for a network or cache problem."""
    path = cache_dir / "index.json"
    cached = None
    try:
        cached = json.loads(path.read_text())
        if not (isinstance(cached, dict) and isinstance(cached.get("paths"), list)
                and re.fullmatch(r"[0-9a-f]{40}", str(cached.get("commit", "")))):
            cached = None
    except (OSError, ValueError):
        cached = None
    age = time.time() - float(cached.get("fetched_at", 0)) if cached else None
    if cached and not refresh and age is not None and age < max_age_s:
        return cached
    if offline:
        if cached:
            print(f"offline: using the cached recipe index ({age / 3600:.1f} h old)",
                  file=sys.stderr)
        return cached
    try:
        index = fetch_index(fetch)
    except Exception as exc:  # noqa: BLE001 -- offline-first: never a failure
        if cached:
            print(f"warning: recipe index fetch failed ({exc}); using the cached "
                  f"index ({age / 3600:.1f} h old)", file=sys.stderr)
        else:
            print(f"warning: recipe index fetch failed ({exc}) and nothing is "
                  f"cached; no recipes", file=sys.stderr)
        return cached
    try:
        _write_atomic(path, json.dumps(index, indent=1).encode())
    except OSError as exc:
        print(f"warning: cannot cache the recipe index in {cache_dir}: {exc}",
              file=sys.stderr)
    return index


def load_recipe(cache_dir: Path, commit: str, path: str, *, fetch: Fetch,
                offline: bool) -> dict | None:
    """One recipe YAML at `commit`, from the cache or the network; None when
    neither has it or it does not parse."""
    if not RECIPE_PATH_RE.match(path):
        return None
    cached = cache_dir / "files" / commit / path
    try:
        text = cached.read_text()
    except OSError:
        if offline:
            return None
        try:
            data = fetch(f"{RAW}/{commit}/{path}", None)
        except Exception as exc:  # noqa: BLE001 -- offline-first
            print(f"warning: {path}: fetch failed ({exc})", file=sys.stderr)
            return None
        try:
            _write_atomic(cached, data)
        except OSError:
            pass
        text = data.decode("utf-8", errors="replace")
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        print(f"warning: {path}: not YAML ({exc})", file=sys.stderr)
        return None
    return doc if isinstance(doc, dict) else None


# ── Normalisation ────────────────────────────────────────────────────────────

def _str_list(value: object) -> list[str]:
    """A recipe's arg list as strings; non-scalar items are dropped."""
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, bool):
            out.append("true" if item else "false")
        elif isinstance(item, (str, int, float)):
            out.append(str(item))
    return out


def _str_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items()
            if isinstance(v, (str, int, float, bool))}


def _flag_value(args: list[str], flag: str) -> str | None:
    for i, tok in enumerate(args):
        if tok == flag and i + 1 < len(args):
            return args[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def _has_flag(args: list[str], flag: str) -> bool:
    return any(t == flag or t.startswith(flag + "=") for t in args)


def extract_parsers(recipe: dict) -> tuple[dict[str, str], list[str]]:
    """({"tool": ..., "reasoning": ...}, notes) from a recipe's args.

    Looked for in base_args, then the default-on features, then the opt-in
    ones; the first value wins. A parser whose args also load a parser
    PLUGIN file is not seeded: the plugin is a file the recipe's user
    supplies, and a launch naming a parser vLLM does not ship fails.
    """
    model = recipe.get("model") if isinstance(recipe.get("model"), dict) else {}
    features = recipe.get("features") if isinstance(recipe.get("features"), dict) else {}
    opt_in = set(_str_list(recipe.get("opt_in_features")))
    arg_lists = [_str_list(model.get("base_args"))]
    for wanted_opt_in in (False, True):
        for key, feature in features.items():
            if (key in opt_in) == wanted_opt_in and isinstance(feature, dict):
                arg_lists.append(_str_list(feature.get("args")))
    parsers: dict[str, str] = {}
    notes: list[str] = []
    for kind, flag, plugin in (("tool", "--tool-call-parser", "--tool-parser-plugin"),
                               ("reasoning", "--reasoning-parser", "--reasoning-parser-plugin")):
        for args in arg_lists:
            value = _flag_value(args, flag)
            if value is None:
                continue
            if _has_flag(args, plugin):
                notes.append(f"{kind} parser {value!r} needs a parser plugin file; not seeded")
            elif PARSER_NAME_RE.match(value):
                parsers[kind] = value
            else:
                notes.append(f"{kind} parser {value!r} is not a plain name; not seeded")
            break
    return parsers, notes


def normalise_recipe(recipe: dict, path: str) -> dict:
    """The parts of a recipe devai uses, in one flat shape."""
    model = recipe.get("model") if isinstance(recipe.get("model"), dict) else {}
    variants_raw = recipe.get("variants") if isinstance(recipe.get("variants"), dict) else {}
    variants = {}
    for key, v in variants_raw.items():
        if not isinstance(v, dict):
            continue
        vram = v.get("vram_minimum_gb")
        variants[str(key)] = {
            "model_id": v.get("model_id") if isinstance(v.get("model_id"), str) else None,
            "precision": str(v.get("precision") or "").lower() or None,
            "vram_minimum_gb": vram if isinstance(vram, (int, float)) and not isinstance(vram, bool) else None,
            "extra_args": _str_list(v.get("extra_args")),
            "extra_env": _str_map(v.get("extra_env")),
        }
    parsers, notes = extract_parsers(recipe)
    # A string only: an unquoted `0.10` is the YAML float 0.1 and would admit
    # every vLLM; a non-string minimum is read as missing, which admits none.
    min_version = model.get("min_vllm_version")
    if not isinstance(min_version, str):
        if min_version is not None:
            notes.append(f"min_vllm_version {min_version!r} is not a string; ignored")
        min_version = None
    return {
        "path": path,
        "model_id": model.get("model_id") if isinstance(model.get("model_id"), str) else None,
        "min_vllm_version": min_version,
        "base_args": _str_list(model.get("base_args")),
        "base_env": _str_map(model.get("base_env")),
        "parsers": parsers,
        "notes": notes,
        "variants": variants,
    }


# ── Matching ─────────────────────────────────────────────────────────────────

def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[-_]", name.lower()) if t]


def base_name(repo_or_name: str) -> str:
    """`nvidia/Qwen3.8-27B-MTP-NVFP4` -> `qwen3.8-27b`: the repo name, lower
    case, without trailing quantisation / packaging tokens."""
    tokens = _tokens(repo_or_name.rsplit("/", 1)[-1])
    while tokens and tokens[-1] in PACKAGING_TOKENS:
        tokens.pop()
    # Interior packaging tokens too (`-MTP-pi-tune-NVFP4` keeps `pi-tune`, so
    # it matches nothing; `-Text-NVFP4-MTP` reduces to the model).
    return "-".join(t for t in tokens if t not in PACKAGING_TOKENS)


def row_precisions(repo_or_name: str) -> tuple[str, ...] | None:
    """Recipe variant precisions a row's name describes, or None."""
    tokens = set(_tokens(repo_or_name.rsplit("/", 1)[-1]))
    for markers, precisions in PRECISION_TOKENS:
        if tokens & markers:
            return precisions
    return None


def build_lookup(paths: list[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """({"org/repo" lower: path}, {base name: [paths]})."""
    by_id: dict[str, str] = {}
    by_base: dict[str, list[str]] = {}
    for path in paths:
        model_id = path[len("models/"):-len(".yaml")]
        by_id[model_id.lower()] = path
        by_base.setdefault(base_name(model_id), []).append(path)
    return by_id, by_base


def match_repo(repo: str, by_id: dict[str, str],
               by_base: dict[str, list[str]]) -> tuple[str | None, str | None]:
    """(recipe path, how) for a catalog repo; (None, reason) when none."""
    path = by_id.get(repo.lower())
    if path:
        return path, "exact"
    candidates = by_base.get(base_name(repo)) or []
    if len(candidates) == 1:
        return candidates[0], "base-name"
    if len(candidates) > 1:
        return None, "ambiguous: " + ", ".join(candidates)
    return None, None


def choose_variant(norm: dict, repo: str, how: str) -> tuple[str | None, str]:
    """(variant key or None, match label) for a row matched to `norm`: the
    variant whose checkpoint the row IS, or None for a base-name match."""
    for key, v in norm["variants"].items():
        if v["model_id"] and v["model_id"].lower() == repo.lower():
            return key, "variant"
    if how == "exact" and "default" in norm["variants"]:
        return "default", how
    return None, how


def precision_vram(norm: dict, repo: str) -> tuple[str | None, float | None]:
    """(precision, the recipe's smallest VRAM minimum for it) for a row whose
    checkpoint is none of the recipe's variants; (None, None) when the row's
    name states no precision the recipe has."""
    wanted = row_precisions(repo)
    if wanted is None:
        return None, None
    figures = [v["vram_minimum_gb"] for v in norm["variants"].values()
               if v["precision"] in wanted and v["vram_minimum_gb"] is not None]
    return wanted[0], (min(figures) if figures else None)


def seed_for_row(row: dict, repo: str, norm: dict, how: str, commit: str) -> dict:
    variant_key, how = choose_variant(norm, repo, how)
    variant = norm["variants"].get(variant_key) if variant_key else None
    engine_env = dict(norm["base_env"])
    if variant:
        engine_env.update(variant["extra_env"])
        precision, vram = variant["precision"], variant["vram_minimum_gb"]
    else:
        precision, vram = precision_vram(norm, repo)
    seed = {
        "repo": row.get("repo"),
        "recipe": norm["path"],
        "recipe_model_id": norm["model_id"],
        "match": how,
        "commit": commit,
        "min_vllm_version": norm["min_vllm_version"],
        "parsers": dict(norm["parsers"]),
        "variant": variant_key,
        "precision": precision,
        "vram_minimum_gb": vram,
        "engine_flags": norm["base_args"] + (variant["extra_args"] if variant else []),
        "engine_env": engine_env,
    }
    if norm["notes"]:
        seed["notes"] = list(norm["notes"])
    return seed


def catalog_hf_rows(catalog: Path) -> list[dict]:
    doc = yaml.safe_load(catalog.read_text()) or {}
    rows = doc.get("models") if isinstance(doc, dict) else None
    return [r for r in (rows or []) if isinstance(r, dict)
            and r.get("source") in ("hf", "derived") and r.get("name")]


def matching_repo(row: dict, by_name: dict[str, dict]) -> str:
    """The repo a row is matched by: a derived row's source repo."""
    source = by_name.get(str(row.get("derived_from") or ""))
    if row.get("source") == "derived" and source and source.get("repo"):
        return str(source["repo"])
    return str(row.get("repo") or "")


def stamp_applicability(name: str, seed: dict) -> dict:
    """Record, per vLLM backend, which vLLM a launch of this row runs and
    whether the recipe's min_vllm_version admits it (`applies_to`). The
    picker shows its badge from this: it runs inside the lab, where neither
    build-vllm.sh nor the image in service can be read. Recomputed at every
    fetch (`make fetch-vllm-recipes OFFLINE=1` after an image change)."""
    versions = {b: _vllm_recipes.backend_vllm_version(name, b) for b in _vllm_recipes.VLLM_BACKENDS}
    seed["vllm_version"] = versions
    seed["applies_to"] = {b: _vllm_recipes.version_admits(seed.get("min_vllm_version"), v)
                          for b, v in versions.items()}
    return seed


def build_seeds(rows: list[dict], index: dict, *, cache_dir: Path, fetch: Fetch,
                offline: bool, previous: dict[str, dict] | None = None,
                ) -> tuple[dict[str, dict], list[str]]:
    """({row name: seed}, report lines), both in catalog order.

    A second pass gives a row that matched no recipe by name a `variant`
    match when its repo is a variant's model_id in a recipe already loaded
    for another row (nvidia/Gemma-4-26B-A4B-NVFP4 is a variant of
    google/gemma-4-26B-A4B-it, whose name it does not share).

    A row whose recipe file cannot be had (offline, or the fetch failed)
    keeps its seed from `previous` (the seeds file as it was), which still
    names the commit it came from.
    """
    by_id, by_base = build_lookup(index["paths"])
    by_name = {r["name"]: r for r in rows}
    commit = index["commit"]
    loaded: dict[str, dict] = {}
    outcome: dict[str, tuple[str, object]] = {}
    for row in rows:
        repo = matching_repo(row, by_name)
        path, how = match_repo(repo, by_id, by_base)
        if path is None:
            outcome[row["name"]] = ("none", how or "no recipe")
            continue
        recipe = load_recipe(cache_dir, commit, path, fetch=fetch, offline=offline)
        if recipe is None:
            outcome[row["name"]] = ("missing", path)
            continue
        loaded[path] = normalise_recipe(recipe, path)
        outcome[row["name"]] = ("seed", seed_for_row(row, repo, loaded[path], how, commit))
    variant_of = {v["model_id"].lower(): path for path, norm in loaded.items()
                  for v in norm["variants"].values() if v["model_id"]}
    for row in rows:
        kind, _ = outcome[row["name"]]
        repo = matching_repo(row, by_name)
        if kind == "none" and repo.lower() in variant_of:
            path = variant_of[repo.lower()]
            outcome[row["name"]] = ("seed", seed_for_row(row, repo, loaded[path], "variant", commit))

    seeds: dict[str, dict] = {}
    report: list[str] = []
    for row in rows:
        kind, detail = outcome[row["name"]]
        if kind == "none":
            report.append(f"  -  {row['name']}: {detail}")
        elif kind == "missing" and (previous or {}).get(row["name"]):
            seeds[row["name"]] = stamp_applicability(row["name"], dict(previous[row["name"]]))
            report.append(f"  =  {row['name']}: {detail} not available (offline or fetch "
                          f"failed); kept the previous seed")
        elif kind == "missing":
            report.append(f"  ?  {row['name']}: {detail} not available (offline or fetch failed)")
        else:
            seed = stamp_applicability(row["name"], detail)
            seeds[row["name"]] = seed
            report.append(f"  +  {row['name']}: {seed['recipe']} ({seed['match']}, variant "
                          f"{seed['variant'] or '-'}, parsers {seed['parsers'] or '-'}, "
                          f"min vLLM {seed['min_vllm_version'] or '?'})")
    return seeds, report


# ── Output ───────────────────────────────────────────────────────────────────

HEADER = """\
# vLLM launch-recipe seeds -- GENERATED by scripts/fetch-vllm-recipes.py
# (`make fetch-vllm-recipes`) from {url}. Do not edit by hand.
#
# One entry per catalog row (deploy/models.yaml) that has an official recipe:
#   parsers           the recipe's tool / reasoning parser; the HF probers use
#                     it as the third step of the parser order (curated,
#                     card-derived, recipe, name heuristic, none), on vLLM
#                     only and only where min_vllm_version admits the vLLM
#   engine_flags/env  base args (+ the variant's extra args when the row IS
#                     that variant's checkpoint); nothing launches with them
#                     until `make recipe-apply NAME=<row>` writes the
#                     reviewed ones to deploy/recovery-flags.generated.json
#   match             exact | variant | base-name (see the script's docstring)
#   vram_minimum_gb   the recipe's figure for the row's precision
#   vllm_version /    per vLLM backend: the vLLM a launch of the row runs on
#   applies_to        this host (a curated image pin, the backend's image,
#                     build-vllm.sh for vllm-devai) and whether
#                     min_vllm_version admits it; the picker's badge reads it
"""


def render_seeds(commit: str | None, seeds: dict[str, dict]) -> str:
    doc = {
        "version": _vllm_recipes.SEEDS_VERSION,
        "source": RECIPES_URL,
        "commit": commit,
        "seeds": seeds,
    }
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False,
                          allow_unicode=False, width=100)
    return HEADER.format(url=RECIPES_URL) + body


def main(argv: list[str] | None = None, *, fetch: Fetch = http_get) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--out", type=Path, default=_vllm_recipes.SEEDS_PATH)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="recipe cache (default: $DEVAI_VLLM_RECIPES_CACHE or "
                         "~/.cache/devai/vllm-recipes)")
    ap.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS,
                    help="re-fetch the index when the cached one is older (default 12)")
    ap.add_argument("--offline", action="store_true", help="never touch the network")
    ap.add_argument("--refresh", action="store_true", help="re-fetch the index now")
    args = ap.parse_args(argv)
    cache_dir = args.cache_dir or default_cache_dir()

    rows = catalog_hf_rows(args.catalog)
    index = load_index(cache_dir, fetch=fetch, max_age_s=args.max_age_hours * 3600,
                       offline=args.offline, refresh=args.refresh)
    if index is None:
        if args.out.exists():
            print(f"no recipe data (offline, nothing cached): {args.out} left as it is")
        else:
            _write_atomic(args.out, render_seeds(None, {}).encode())
            print(f"no recipe data (offline, nothing cached): wrote the empty {args.out}")
        return 0

    seeds, report = build_seeds(rows, index, cache_dir=cache_dir, fetch=fetch,
                                offline=args.offline,
                                previous=_vllm_recipes.load_seeds(args.out))
    _write_atomic(args.out, render_seeds(index["commit"], seeds).encode())
    print(f"recipes {RECIPES_URL} at {index['commit'][:12]} "
          f"({len(index['paths'])} recipes); {len(rows)} catalog HF rows")
    print("\n".join(report))
    print(f"wrote {args.out}: {len(seeds)} seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
