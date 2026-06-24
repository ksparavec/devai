#!/usr/bin/env python3
"""Discover newer-version members of already-tracked model lineages.

Read-only HINT generator. Reads scripts/model-families.yaml, groups the
tracked families into *lineages* (same brand + sub-lineage, differing
version -- e.g. qwen3 / qwen3.5 / qwen3.6 are one "qwen" lineage), then
queries upstream for variants whose version is NOT yet tracked:

  - HuggingFace: searches each lineage's already-trusted authors for
    repos matching the lineage's name pattern. Scoping to authors the
    catalog already references is the quality gate -- discovery never
    surfaces a repo from an author you haven't vetted unless a
    `discover:` block opts that author in.
  - Ollama: probes ollama.com for the next library-name versions a
    lineage would publish (qwen3.5 -> qwen3.6, qwen3.7, qwen4, ...) and
    reports which already exist but are untracked, plus which are not
    yet published.

Lineage definition is HYBRID:
  - By default each family's lineage (brand, version, sub-lineage suffix)
    and its trusted authors are auto-derived from the family `name` and
    its existing `hf_repos` / `ollama_repos` / `gguf_repos`.
  - An optional per-family `discover:` block overrides/constrains the
    auto-derivation (authors, name pattern, minimum version, ollama
    naming, or disabling discovery entirely). See model-families.yaml.

VRAM-band filter: the catalog was curated for the host GPU, so a candidate
is only useful if it lands in a usable VRAM BAND -- big enough to use the
GPU well, small enough to fit. Each candidate's weight VRAM is ESTIMATED
from its parameter count times an approximate bytes-per-parameter for its
quant format (params alone are not enough -- a 9B is ~5 GB at NVFP4 but
~18 GB at BF16, and only the BF16 one uses the card well). The band is:
  floor   = min_vram_frac x GPU budget (default 50% -- below this a 1B
            model just wastes the GPU)
  ceiling = the family's largest tracked model x tolerance, capped at the
            GPU budget (and falling back to the full GPU budget for a small
            family whose own max sits below the floor)
For an UNMARKED ('?') format the param x bytes estimate would have to guess
the precision, so the real on-disk weight size is fetched instead (one
extra API call; shown with '=' instead of '~'). Candidates outside the band
are hidden by default with a count.

Base / pretraining checkpoints are also hidden by default (the lab wants
chat/instruct models). A repo is base if its name says so (-Base/-pretrain)
or the HF `conversational` tag is absent and the name does not say instruct
-- conservative, so a tag-less named-instruct quant survives.

`--include-oversized` / `--include-undersized` / `--include-base` show the
hidden ones flagged. The estimate is a coarse pre-filter, not a fit
guarantee -- `make probe` stays authoritative.

Discovery is READ-ONLY: the report never edits anything. Adding a
candidate is a separate, explicit step (`--add`) that writes
model-families.yaml only after a per-entry confirmation, and only for
EXISTING families (a NEWER version that would need a new family is
refused -- new families need arch_ref/parsers curation the tool cannot
infer). Either way the probe cache stays truth: run `make probe` before
relying on anything added. Same review contract as `make catalog-suggest`.

Usage:
    python3 scripts/catalog-discover.py            # full report (read-only)
    python3 scripts/catalog-discover.py --family qwen3.5
    python3 scripts/catalog-discover.py --json
    python3 scripts/catalog-discover.py --add nvidia/Qwen3.6-35B-A3B-NVFP4 --yes
    python3 scripts/catalog-discover.py --add      # interactive confirm loop
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
FAMILIES_YAML = REPO_ROOT / "scripts" / "model-families.yaml"

HF_API = "https://huggingface.co/api/models"
OLLAMA_WEB = "https://ollama.com/library"
USER_AGENT = "devai-catalog-discover/1.0"

# VRAM-range filter. A discovered candidate is only useful if it fits the
# same VRAM envelope as the models the family already tracks -- the catalog
# was curated for the host GPU, so a 397B checkpoint on a 24 GB card is
# noise, not a discovery. We ESTIMATE weight VRAM from the parameter count
# in the model name times an approximate bytes-per-parameter for its quant
# format (params alone are not enough: a 35B is ~19 GB at NVFP4 but ~70 GB
# at BF16). The estimate is a coarse pre-filter, not a fit guarantee --
# `make probe` remains the source of truth.
DEFAULT_GPU_MEMORY_GB = 24.0
DEFAULT_VRAM_TOLERANCE = 1.25  # allow the next size step up from the family max
# Lower bound: a model that uses less than this fraction of the GPU wastes
# it (a 1B model on a 24 GB card). Discovery surfaces a BAND -- big enough
# to use the GPU well, small enough to fit.
DEFAULT_MIN_VRAM_FRAC = 0.50

# Approximate weight bytes per parameter, keyed by the format label that
# detect_format() returns. NVFP4/MXFP4/INT4 ~ 4-bit + per-block scales;
# FP8/INT8 ~ 1 byte; BF16/FP16 = 2 bytes. GGUF is a mix of 3-5 bit k-quants.
_BYTES_PER_PARAM = {
    "NVFP4": 0.55, "MXFP4": 0.55, "FP4": 0.55, "INT4": 0.55,
    "AWQ": 0.55, "GPTQ": 0.55, "GGUF": 0.60,
    "FP8": 1.10, "INT8": 1.10,
    "BF16": 2.0, "FP16": 2.0,
}
# Unmarked safetensors repos are almost always full-precision -> assume the
# worst (2 bytes) so an unquantized giant is not waved through.
_DEFAULT_BYTES_PER_PARAM = 2.0

# Quant / format markers we surface in the report so a candidate's
# serving backend is obvious at a glance (NVFP4 -> vllm, gguf -> ollama).
_FORMAT_MARKERS = (
    "nvfp4", "mxfp4", "fp4", "fp8", "awq", "gptq", "gguf", "int4", "int8",
    "bf16", "fp16",
)

# A size token: digits-then-'b' (8b, 27b, 235b), optionally with a single
# leading letter for MoE active-param notation (a3b, a10b). Lets us tell a
# size apart from a foreign product-line word when scanning a repo name.
_SIZE_TOKEN_RE = re.compile(r"^[a-z]?\d+(?:\.\d+)?b$")

# Line-neutral tuning qualifiers that may sit between the version and the
# size token without signalling a different product line. A token that is
# NOT one of these (and not a size or format marker) -- e.g. 'vl', 'next',
# 'omni', 'embedding', 'guard', 'coder' -- marks a DIFFERENT lineage.
_TUNING_QUALIFIERS = frozenset({
    "instruct", "instruction", "it", "chat", "base", "thinking", "reasoning",
    "text",
})

# Name tokens that positively identify a chat/instruct model vs a base
# (pretraining) checkpoint. A repo named instruct is kept even if its HF
# `conversational` tag is missing (some NVFP4 quants strip it); a repo named
# base is dropped; otherwise we fall back to the conversational tag.
_INSTRUCT_NAME_TOKENS = frozenset({
    "instruct", "instruction", "it", "chat", "thinking", "reasoning",
})
_BASE_NAME_TOKENS = frozenset({"base", "pretrain", "pretrained", "pretraining"})


# == Pure logic (no network; unit-tested in tests/python) =====================

def parse_version(text: str) -> tuple[int, ...] | None:
    """Parse a dotted numeric version into a comparable int tuple.

    '3' -> (3,), '3.5' -> (3, 5), '3.10' -> (3, 10). Tuple comparison
    gives the right ordering (3,) < (3, 5) < (3, 10) -- using a float
    would wrongly collapse 3.10 onto 3.1. Returns None for non-numeric
    input.
    """
    text = str(text).strip()
    if not re.fullmatch(r"\d+(?:\.\d+)*", text):
        return None
    return tuple(int(p) for p in text.split("."))


def version_str(ver: tuple[int, ...]) -> str:
    """Render a version tuple the way model names spell it: (3, 5) -> '3.5'."""
    return ".".join(str(p) for p in ver)


def parse_family_lineage(name: str) -> tuple[str, tuple[int, ...] | None, str]:
    """Split a family `name` into (brand, version, sub-lineage suffix).

    The version follows a single letter-run brand, optionally separated by
    one delimiter (qwen3.5, gemma4, llama3.1, nemotron-3-nano), then an
    optional sub-lineage suffix (qwen3-coder, -nano). A name whose version
    is not reachable that way -- a multi-segment brand (gpt-oss) or a
    non-numeric version token (deepseek-r1-distill, nemotron-nano-v2) or
    none at all (diffusiongemma) -- returns version=None; those only get
    discovery through an explicit `discover:` block.

      'qwen3.5'         -> ('qwen', (3, 5), '')
      'qwen3'           -> ('qwen', (3,),   '')
      'qwen3-coder'     -> ('qwen', (3,),   '-coder')
      'gemma4'          -> ('gemma', (4,),  '')
      'llama3.1'        -> ('llama', (3, 1), '')
      'nemotron-3-nano' -> ('nemotron', (3,), '-nano')
      'gpt-oss'         -> ('gpt-oss', None, '')
    """
    m = re.match(r"^([a-z][a-z]*)[-_.]?(\d+(?:\.\d+)*)(.*)$", name)
    if not m:
        return name, None, ""
    return m.group(1), parse_version(m.group(2)), m.group(3)


def lineage_key(brand: str, suffix: str) -> str:
    """Stable grouping key: families sharing brand+suffix are one lineage."""
    return f"{brand}|{suffix}"


def repo_author(repo: str) -> str:
    """'nvidia/Qwen3-8B-NVFP4' -> 'nvidia'. Bare repo -> ''."""
    return repo.split("/", 1)[0] if "/" in repo else ""


def repo_basename(repo: str) -> str:
    """'nvidia/Qwen3-8B-NVFP4' -> 'Qwen3-8B-NVFP4'."""
    return repo.rsplit("/", 1)[-1]


def extract_repo_version(basename: str, brand: str) -> tuple[int, ...] | None:
    """Extract the LINEAGE version from an upstream repo basename.

    Finds the numeric token attached to the brand word that is a COMPLETE
    delimited token (followed by '-', '_', '.', or end of string) -- so a
    size like 8B / 20b / 26B (digits glued to a letter) is never mistaken
    for a version. The brand may glue directly to the version (Qwen3) or
    be separated by a single delimiter (Llama-3.1, gemma-4).

      ('Qwen3.6-35B-A3B-NVFP4',       'qwen')   -> (3, 6)
      ('Qwen3-8B-NVFP4',              'qwen')    -> (3,)
      ('Llama-3.1-8B-Instruct',       'llama')   -> (3, 1)
      ('gemma-4-26b-a4b-it',          'gemma')   -> (4,)
      ('gpt-oss-20b',                 'gpt-oss') -> None   # 20b is a size
      ('diffusiongemma-26B-A4B-it',   'diffusiongemma') -> None
    """
    pat = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(brand)}[-_.]?(\d+(?:\.\d+)*)(?=[-_.]|$)",
        re.IGNORECASE,
    )
    m = pat.search(basename)
    if not m:
        return None
    return parse_version(m.group(1))


def detect_format(basename: str, tags: list[str]) -> str:
    """Best-effort quant/format label from the repo name + HF tags."""
    hay = (basename + " " + " ".join(tags)).lower()
    found = [mk for mk in _FORMAT_MARKERS if mk in hay]
    # Prefer the most specific marker (nvfp4 over fp4 over fp16).
    for pref in ("nvfp4", "mxfp4", "fp8", "fp4", "awq", "gptq", "gguf",
                 "int4", "int8", "bf16", "fp16"):
        if pref in found:
            return pref.upper()
    return "?"


def parse_param_count(name: str) -> float | None:
    """Total parameter count in billions parsed from a model name.

    Takes the largest plain `<num>B` token, which is the TOTAL parameter
    count -- the letter-prefixed MoE active-param token (A3B, A22B) is
    deliberately ignored because weight VRAM scales with total, not active.

      '8B'        -> 8.0      '0.6B'         -> 0.6
      '35B-A3B'   -> 35.0     '397B-A17B'    -> 397.0
      '405B'      -> 405.0    'Qwen3-30B-A3B'-> 30.0
    """
    nums: list[float] = []
    for m in re.finditer(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)[Bb](?=[-_.]|$)", name):
        nums.append(float(m.group(1)))
    return max(nums) if nums else None


def is_base_model(basename: str, conversational: bool) -> bool:
    """True for a base (pretraining) checkpoint, not a chat/instruct model.

    Conservative so a tag-less instruct quant survives:
      - name token says base/pretrain   -> base
      - name token says instruct/chat/...-> NOT base (even without the tag)
      - otherwise                        -> trust the HF conversational tag
    """
    toks = {t.lower() for t in re.split(r"[^A-Za-z0-9]+", basename) if t}
    if toks & _BASE_NAME_TOKENS:
        return True
    if toks & _INSTRUCT_NAME_TOKENS:
        return False
    return not conversational


def bytes_per_param(fmt: str) -> float:
    return _BYTES_PER_PARAM.get(fmt.upper(), _DEFAULT_BYTES_PER_PARAM)


def estimate_vram_gb(params_b: float | None, fmt: str) -> float | None:
    """Approximate weight VRAM in GB (params x bytes-per-param), or None
    when the parameter count cannot be parsed from the name."""
    if params_b is None:
        return None
    return params_b * bytes_per_param(fmt)


def candidate_next_versions(
    tracked: list[tuple[int, ...]], minor_steps: int = 4, major_steps: int = 2,
) -> list[tuple[int, ...]]:
    """Plausible next library versions forward from the tracked maximum.

    A lineage that has EVER used minor versions (any tracked tuple has a
    fractional part) is treated as minor-versioned: bump the current
    major's last minor (3.6 -> 3.7/3.8; and if the max is a bare new
    major like 4, that yields 4.1/4.2) AND the next whole majors (4, 5).
    A lineage that has only ever used integers (gemma: {(4,)}) bumps the
    integer (5, 6, ...). Deciding on `uses_minors` rather than on the
    shape of max() alone matters when a bare-integer release (llama4)
    coexists with earlier minors (llama3.1/3.2): max() is (4,) but we
    still want 4.1/4.2 probed.
    """
    if not tracked:
        return []
    mx = max(tracked)
    major = mx[0]
    last_minor = mx[1] if len(mx) >= 2 else 0
    uses_minors = any(len(v) >= 2 for v in tracked)
    out: list[tuple[int, ...]] = []
    if uses_minors:
        for i in range(1, minor_steps + 1):
            out.append((major, last_minor + i))
        for i in range(1, major_steps + 1):
            out.append((major + i,))
    else:
        for i in range(1, minor_steps + major_steps + 1):
            out.append((major + i,))
    return out


@dataclass
class DiscoverConfig:
    """Parsed (and validated) per-family `discover:` override block."""
    enabled: bool = True
    hf_authors: list[str] = field(default_factory=list)
    name_regex: str | None = None
    min_version: tuple[int, ...] | None = None
    ollama_names: list[str] = field(default_factory=list)


def parse_discover_block(raw: object) -> DiscoverConfig:
    """Normalize a family's `discover:` mapping. Unknown/absent -> defaults.

    min_version is read as a STRING ('3.5') to dodge YAML float coercion
    (3.10 would parse as the float 3.1); we also accept ints/floats and
    coerce defensively.
    """
    if not isinstance(raw, dict):
        return DiscoverConfig()
    cfg = DiscoverConfig()
    if "enabled" in raw:
        cfg.enabled = bool(raw["enabled"])
    authors = raw.get("hf_authors")
    if isinstance(authors, list):
        cfg.hf_authors = [str(a).strip() for a in authors if str(a).strip()]
    nr = raw.get("name_regex")
    if isinstance(nr, str) and nr.strip():
        # Compile-validate now so a typo in one family's block can't crash
        # discovery for every lineage (the other fields are coerced too).
        try:
            re.compile(nr.strip())
            cfg.name_regex = nr.strip()
        except re.error as e:
            print(f"  [warn] invalid discover.name_regex {nr!r}: {e} "
                  f"-- ignoring", file=sys.stderr)
    mv = raw.get("min_version")
    if mv is not None:
        cfg.min_version = parse_version(str(mv))
    onames = raw.get("ollama_names")
    if isinstance(onames, list):
        cfg.ollama_names = [str(n).strip() for n in onames if str(n).strip()]
    return cfg


@dataclass
class Lineage:
    """A brand+sub-lineage aggregated across one or more catalog families."""
    key: str
    brand: str
    suffix: str
    family_names: list[str] = field(default_factory=list)
    # version -> the family name that already tracks it (for mapping a
    # discovered same-version repo back to its home family).
    version_to_family: dict[tuple[int, ...], str] = field(default_factory=dict)
    hf_versions: set[tuple[int, ...]] = field(default_factory=set)
    ollama_versions: set[tuple[int, ...]] = field(default_factory=set)
    hf_authors: set[str] = field(default_factory=set)
    tracked_repos: set[str] = field(default_factory=set)   # lowercased repo ids
    ollama_libs: set[str] = field(default_factory=set)      # lowercased lib names
    discover: DiscoverConfig = field(default_factory=DiscoverConfig)

    @property
    def all_versions(self) -> set[tuple[int, ...]]:
        return self.hf_versions | self.ollama_versions

    @property
    def suffix_token_list(self) -> list[str]:
        """Ordered alphabetic tokens of this lineage's own suffix.

        '-coder' -> ['coder']; '-nano' -> ['nano']; '' -> [].
        """
        return [t for t in re.split(r"[^a-z0-9]+", self.suffix.lower()) if t.isalpha()]

    def tracked_vram_estimates(self) -> list[float]:
        """Estimated weight VRAM (GB) of each tracked repo whose size is
        parseable from its name -- the family's known-fitting envelope."""
        out: list[float] = []
        for repo in self.tracked_repos:
            base = repo_basename(repo)
            est = estimate_vram_gb(parse_param_count(base), detect_format(base, []))
            if est is not None:
                out.append(est)
        return out

    def vram_band_gb(self, gpu_budget: float, tolerance: float,
                     min_frac: float) -> tuple[float, float]:
        """The (floor, ceiling) weight-VRAM band a candidate must land in.

        floor   = min_frac x GPU budget -- below this the model wastes the
                  card (a 1B model on 24 GB).
        ceiling = the family's biggest tracked model x tolerance, capped at
                  the GPU budget. When the family's largest model is itself
                  below the floor (a small family), that family ceiling would
                  make the band empty, so we fall back to the full GPU budget
                  -- the floor still keeps tiny models out.
        """
        floor = min_frac * gpu_budget
        ests = self.tracked_vram_estimates()
        if ests:
            family_ceiling = max(ests) * tolerance
            ceiling = (gpu_budget if family_ceiling < floor
                       else min(gpu_budget, family_ceiling))
        else:
            ceiling = gpu_budget
        return floor, ceiling


def build_lineages(families: list[dict]) -> dict[str, Lineage]:
    """Group families into lineages and aggregate their tracked metadata."""
    lineages: dict[str, Lineage] = {}

    for fam in families:
        name = str(fam.get("name", "")).strip()
        if not name:
            continue
        brand, ver, suffix = parse_family_lineage(name)
        key = lineage_key(brand, suffix)
        lin = lineages.get(key)
        if lin is None:
            lin = Lineage(key=key, brand=brand, suffix=suffix)
            lin.discover = parse_discover_block(fam.get("discover"))
            lineages[key] = lin
        else:
            # Merge discover blocks across families of one lineage: a
            # later family's block fills gaps but does not clobber.
            _merge_discover(lin.discover, parse_discover_block(fam.get("discover")))
        lin.family_names.append(name)

        # hf_repos entries may be a bare string or a {repo: ...} mapping.
        for spec in fam.get("hf_repos") or []:
            repo = spec if isinstance(spec, str) else (spec or {}).get("repo")
            if not repo:
                continue
            _absorb_hf_repo(lin, repo, brand)
        for spec in fam.get("gguf_repos") or []:
            repo = (spec or {}).get("repo") if isinstance(spec, dict) else spec
            if repo:
                lin.tracked_repos.add(str(repo).lower())
                if repo_author(str(repo)):
                    lin.hf_authors.add(repo_author(str(repo)))
        for lib in fam.get("ollama_repos") or []:
            lib = str(lib).strip()
            if not lib:
                continue
            lin.ollama_libs.add(lib.lower())
            ob, ov, _osuf = parse_family_lineage(lib)
            if ov is not None:
                lin.ollama_versions.add(ov)

        if ver is not None:
            lin.version_to_family.setdefault(ver, name)

    return lineages


def _merge_discover(base: DiscoverConfig, other: DiscoverConfig) -> None:
    if not other.enabled:
        base.enabled = False
    for a in other.hf_authors:
        if a not in base.hf_authors:
            base.hf_authors.append(a)
    if base.name_regex is None and other.name_regex:
        base.name_regex = other.name_regex
    if base.min_version is None and other.min_version:
        base.min_version = other.min_version
    for n in other.ollama_names:
        if n not in base.ollama_names:
            base.ollama_names.append(n)


def _absorb_hf_repo(lin: Lineage, repo: str, brand: str) -> None:
    lin.tracked_repos.add(repo.lower())
    author = repo_author(repo)
    if author:
        lin.hf_authors.add(author)
    rv = extract_repo_version(repo_basename(repo), brand)
    if rv is not None:
        lin.hf_versions.add(rv)


def is_size_token(tok: str) -> bool:
    return bool(_SIZE_TOKEN_RE.fullmatch(tok.lower()))


def _as_int(value: object) -> int:
    """Coerce an upstream numeric field to int, defaulting to 0.

    HF /api/models returns integer likes/downloads, but a contract
    violation (None, a float-as-string, 'NaN') must not abort the run.
    """
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def is_clean_lineage_member(basename: str, lin: Lineage,
                            author: str = "") -> bool:
    """Structurally confirm a repo is THIS lineage's line, not a cousin.

    Rejects foreign product lines that merely share the brand+version
    string -- Qwen3-VL (vision), Qwen3-Next (different arch), Qwen3-Coder
    (sibling lineage), DeepSeek-R1-...-Qwen3 (a distill), KVzap-mlp-Qwen3
    (research artifact), and finetune brands like OpenMath2-Llama3.1 or
    Dolphin-Qwen3. The shape we accept is:

        [<org prefix>-] <brand><ver> -<suffix tokens?> -<size> ...

    i.e. after the brand+version we must see exactly this lineage's own
    sub-lineage suffix tokens (none for a base lineage) followed by a size
    token, with only line-neutral tuning qualifiers allowed in between. Any
    text BEFORE the brand must be a single token related to the publishing
    `author` (an org self-prefix like 'NVIDIA-Nemotron' from author nvidia,
    or 'Meta-Llama' from meta-llama) -- a foreign brand word ('OpenMath2-',
    'Dolphin-') or a multi-token derivative prefix ('DeepSeek-R1-0528-') is
    rejected even when the author is otherwise trusted.
    """
    pat = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(lin.brand)}[-_.]?(\d+(?:\.\d+)*)(?=[-_.]|$)",
        re.IGNORECASE,
    )
    m = pat.search(basename)
    if not m:
        return False
    # Any text before the brand must be a single org self-prefix token
    # (related to the publishing author), else it is a finetune/derivative
    # brand and this is not the base line.
    prefix_alpha = [t for t in re.split(r"[^A-Za-z]+", basename[:m.start()]) if t]
    if prefix_alpha:
        a = (author or "").lower()
        tok = prefix_alpha[0].lower()
        if len(prefix_alpha) > 1 or not a or not (tok in a or a in tok):
            return False
    # Split the post-version tail, KEEPING decimals inside size tokens
    # (0.6B must survive as one token, not split into '0' and '6b').
    toks = [t.strip(".").lower()
            for t in re.split(r"[^A-Za-z0-9.]+", basename[m.end():])]
    toks = [t for t in toks if t]
    i = 0
    # Consume this lineage's own suffix tokens, in order.
    for exp in lin.suffix_token_list:
        if i >= len(toks) or toks[i] != exp:
            return False
        i += 1
    # Everything up to the first size token must be a neutral qualifier.
    while i < len(toks):
        tok = toks[i]
        if is_size_token(tok):
            return True
        if tok in _TUNING_QUALIFIERS or tok in _FORMAT_MARKERS:
            i += 1
            continue
        return False  # foreign product-line token (vl, next, omni, ...)
    return False  # no size token at all -- not a normal weight repo


def repo_matches_lineage(basename: str, lin: Lineage,
                         name_re: re.Pattern[str], strict: bool,
                         author: str = "") -> bool:
    """True when an upstream repo belongs to THIS lineage.

    `strict` (auto mode, default brand pattern) adds the structural
    line-membership check (which uses `author` to validate any org
    prefix). When the family supplies a custom `discover.name_regex` we
    trust the operator's pattern and skip the structural filter.
    """
    if not name_re.search(basename):
        return False
    if strict:
        return is_clean_lineage_member(basename, lin, author)
    return True


def classify_candidate(ver: tuple[int, ...], lin: Lineage) -> tuple[str, str]:
    """Return (class, family-mapping hint) for a discovered version.

    class is one of NEWER / SAME / GAP. The hint names the family to add
    it under, or proposes a new family name for an untracked version.
    """
    tracked = lin.all_versions
    if ver in lin.version_to_family:
        return "SAME", f"add under family {lin.version_to_family[ver]}"
    proposed = f"{lin.brand}{version_str(ver)}{lin.suffix}"
    if tracked and ver > max(tracked):
        return "NEWER", f"new family {proposed} likely needed"
    return "GAP", f"new family {proposed} likely needed"


# == Network I/O ==============================================================

def http_json(url: str, timeout: int) -> object:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_status(url: str, timeout: int) -> int:
    """GET a URL and return its HTTP status (404 instead of raising)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0  # network failure -- treat as "unknown"


# Weight-file suffixes the loader actually reads, and mirror subdirs some
# repos ship (bf16/Metal/PyTorch copies) that would double-count the size.
# Mirrors generate-catalog.py's hf_weight_bytes.
_HF_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth")
_HF_NON_PRIMARY_DIRS = ("original/", "metal/", "consolidated/")


def hf_weight_gb(repo: str, timeout: int) -> float | None:
    """Actual on-disk weight size (GB) from the HF blob list, or None.

    One extra API call; used only for unmarked ('?') repos where the
    bytes-per-param estimate would otherwise have to guess the precision.
    Sums top-level weight files, excluding alternate-format mirror dirs.
    """
    try:
        data = http_json(f"{HF_API}/{repo}?blobs=true", timeout)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    total = 0
    for f in data.get("siblings") or []:
        rfn = (f.get("rfilename") or "") if isinstance(f, dict) else ""
        if not rfn.endswith(_HF_WEIGHT_SUFFIXES):
            continue
        if any(rfn.startswith(d) for d in _HF_NON_PRIMARY_DIRS):
            continue
        total += int(f.get("size") or 0)
    return total / (1024 ** 3) if total else None


def hf_search_author(author: str, query: str, limit: int,
                     timeout: int) -> list[dict]:
    from urllib.parse import quote
    url = (f"{HF_API}?author={quote(author)}&search={quote(query)}"
           f"&limit={limit}&full=false&config=false")
    try:
        data = http_json(url, timeout)
    except Exception as e:
        print(f"  [warn] HF search author={author} query={query}: {e}",
              file=sys.stderr)
        return []
    return data if isinstance(data, list) else []


# == Discovery ================================================================

@dataclass
class HFCandidate:
    repo: str
    author: str
    version: tuple[int, ...]
    klass: str
    mapping: str
    fmt: str
    likes: int
    downloads: int
    created_at: str
    conversational: bool
    params_b: float | None = None    # total params (billions) from the name
    est_vram_gb: float | None = None  # weight VRAM (estimated or measured)
    vram_measured: bool = False       # True when est_vram_gb is the real size
    oversized: bool = False           # above the VRAM band ceiling
    undersized: bool = False          # below the VRAM band floor (wastes GPU)
    base: bool = False                # base/pretraining checkpoint, not chat

    @property
    def in_range(self) -> bool:
        return not self.oversized and not self.undersized


@dataclass
class OllamaCandidate:
    library: str
    version: tuple[int, ...]
    status: int       # 200 exists / 404 not published / 0 unknown
    klass: str
    mapping: str


def effective_authors(lin: Lineage) -> list[str]:
    """The author allowlist discovery actually uses.

    A `discover.hf_authors` block REPLACES the auto-derived set (so you can
    *restrict* discovery to vetted authors, e.g. drop community quants).
    Absent -> the authors auto-derived from the family's tracked repos.
    """
    if lin.discover.hf_authors:
        return sorted(set(lin.discover.hf_authors))
    return sorted(lin.hf_authors)


def lineage_name_re(lin: Lineage) -> re.Pattern[str]:
    """The name pattern matching repos of this lineage: the operator's custom
    `discover.name_regex`, else brand+version as a left-bounded word anywhere
    in the basename (repos often org-prefix the brand: NVIDIA-Nemotron-3-)."""
    if lin.discover.name_regex:
        return re.compile(lin.discover.name_regex, re.IGNORECASE)
    return re.compile(rf"(?<![A-Za-z]){re.escape(lin.brand)}[-_.]?\d",
                      re.IGNORECASE)


def discover_hf(lin: Lineage, *, hf_limit: int, timeout: int,
                gpu_budget: float, vram_tolerance: float,
                min_vram_frac: float) -> list[HFCandidate]:
    authors = effective_authors(lin)
    if not authors:
        return []
    name_re = lineage_name_re(lin)
    min_ver = lin.discover.min_version
    if min_ver is None and lin.all_versions:
        min_ver = min(lin.all_versions)
    floor, ceiling = lin.vram_band_gb(gpu_budget, vram_tolerance, min_vram_frac)

    seen: set[str] = set()
    out: list[HFCandidate] = []
    for author in authors:
        for entry in hf_search_author(author, lin.brand, hf_limit, timeout):
            if not isinstance(entry, dict):
                continue  # never trust the upstream API shape
            repo = str(entry.get("id") or "")
            if not repo or repo.lower() in lin.tracked_repos:
                continue
            if repo.lower() in seen:
                continue
            base = repo_basename(repo)
            repo_auth = repo_author(repo)
            if not repo_matches_lineage(base, lin, name_re,
                                        strict=lin.discover.name_regex is None,
                                        author=repo_auth):
                continue
            ver = extract_repo_version(base, lin.brand)
            if ver is None:
                continue
            if min_ver is not None and ver < min_ver:
                continue
            tags = entry.get("tags") or []
            tags = tags if isinstance(tags, list) else []
            fmt = detect_format(base, tags)
            conv = "conversational" in tags
            is_base = is_base_model(base, conv)
            params = parse_param_count(base)
            est_vram = estimate_vram_gb(params, fmt)
            measured = False
            # For an unmarked format, the param x bytes estimate has to guess
            # the precision; read the real on-disk size instead (one extra
            # call). Skip base repos -- they are hidden anyway.
            if fmt == "?" and not is_base:
                real = hf_weight_gb(repo, timeout)
                if real is not None:
                    est_vram, measured = real, True
            klass, mapping = classify_candidate(ver, lin)
            seen.add(repo.lower())
            out.append(HFCandidate(
                repo=repo,
                author=repo_auth,
                version=ver,
                klass=klass,
                mapping=mapping,
                fmt=fmt,
                likes=_as_int(entry.get("likes")),
                downloads=_as_int(entry.get("downloads")),
                created_at=str(entry.get("createdAt") or ""),
                conversational=conv,
                params_b=params,
                est_vram_gb=est_vram,
                vram_measured=measured,
                # A parseable estimate outside the band is out of range; an
                # UNparseable size is kept (flagged) so we never silently
                # drop a candidate we could not measure.
                oversized=(est_vram is not None and est_vram > ceiling),
                undersized=(est_vram is not None and est_vram < floor),
                base=is_base,
            ))
    out.sort(key=lambda c: (c.version, c.downloads), reverse=True)
    return out


def _ollama_name_templates(lin: Lineage) -> list[str]:
    """Templates with a '{V}' placeholder for the version, derived from the
    lineage's tracked ollama library names (so naming matches upstream)."""
    templates: list[str] = []
    for lib in sorted(lin.ollama_libs):
        b, v, suf = parse_family_lineage(lib)
        if v is None:
            continue
        tmpl = f"{b}{{V}}{suf}"
        if tmpl not in templates:
            templates.append(tmpl)
    return templates


def discover_ollama(lin: Lineage, *, probe_count: int,
                    timeout: int) -> list[OllamaCandidate]:
    templates = _ollama_name_templates(lin)
    explicit = lin.discover.ollama_names
    if not templates and not explicit:
        return []

    # Versions to probe: forward from the ollama-tracked max, plus any
    # version tracked on the HF side but missing an ollama library (a
    # cross-backend gap, e.g. you track qwen3.6 weights but not its
    # ollama lib).
    forward = candidate_next_versions(
        sorted(lin.ollama_versions), minor_steps=probe_count, major_steps=2)
    gap = sorted(lin.hf_versions - lin.ollama_versions)
    versions: list[tuple[int, ...]] = []
    for v in gap + forward:
        if v not in versions:
            versions.append(v)

    out: list[OllamaCandidate] = []
    seen: set[str] = set()
    # Templated (version-bearing) candidates.
    for ver in versions:
        for tmpl in templates:
            lib = tmpl.replace("{V}", version_str(ver))
            if lib.lower() in lin.ollama_libs or lib.lower() in seen:
                continue
            seen.add(lib.lower())
            status = http_status(f"{OLLAMA_WEB}/{lib}/tags", timeout)
            klass, mapping = classify_candidate(ver, lin)
            out.append(OllamaCandidate(lib, ver, status, klass, mapping))
    # Explicit names from a discover: block (no version inference).
    for lib in explicit:
        if lib.lower() in lin.ollama_libs or lib.lower() in seen:
            continue
        seen.add(lib.lower())
        status = http_status(f"{OLLAMA_WEB}/{lib}/tags", timeout)
        out.append(OllamaCandidate(lib, (), status, "EXPLICIT",
                                   "from discover.ollama_names"))
    out.sort(key=lambda c: (c.status == 200, c.version), reverse=True)
    return out


# == Reporting ================================================================

def _status_label(code: int) -> str:
    return {200: "EXISTS", 404: "not yet published"}.get(code, f"http {code}")


def _vram_label(c: "HFCandidate") -> str:
    if c.est_vram_gb is None:
        return "~?GB"
    prefix = "=" if c.vram_measured else "~"   # = measured, ~ estimated
    return f"{prefix}{c.est_vram_gb:.0f}GB"


def _fmt_hf_row(c: "HFCandidate") -> str:
    conv = "IT" if c.conversational else "base"
    flag = (" OVERSIZED" if c.oversized
            else " UNDERSIZED" if c.undersized
            else " BASE" if c.base else "")
    return (f"     [{c.klass:5}] {c.repo:<46} {c.fmt:<6} "
            f"v{version_str(c.version):<5} {_vram_label(c):>6} {c.likes:>4}❤ "
            f"{c.downloads:>9}↓ {conv:<4} -> {c.mapping}{flag}")


def _is_shown(c: "HFCandidate", *, include_oversized: bool,
              include_undersized: bool, include_base: bool) -> bool:
    """A candidate is shown only if it passes BOTH the VRAM-band gate and
    the base-model gate."""
    if c.oversized and not include_oversized:
        return False
    if c.undersized and not include_undersized:
        return False
    if c.base and not include_base:
        return False
    return True


# How many SAME-version repos to show per home-family before collapsing to
# a count (use --include-same to list them all).
SAME_PER_FAMILY = 3


def _render_hf(hf: list["HFCandidate"], lines: list[str], *,
               include_same: bool, include_oversized: bool,
               include_undersized: bool, include_base: bool) -> None:
    # Hide candidates outside the family's usable VRAM band (too big to load,
    # too small to be worth the GPU) and base/pretraining checkpoints, unless
    # explicitly requested.
    oversized = [c for c in hf if c.oversized]
    undersized = [c for c in hf if c.undersized]
    base_hidden = [c for c in hf if c.base and c.in_range]  # hidden only by base gate
    shown_hf = [c for c in hf if _is_shown(
        c, include_oversized=include_oversized,
        include_undersized=include_undersized, include_base=include_base)]

    newer = [c for c in shown_hf if c.klass == "NEWER"]
    gap = [c for c in shown_hf if c.klass == "GAP"]
    same = [c for c in shown_hf if c.klass == "SAME"]
    if not shown_hf:
        hidden = []
        if oversized and not include_oversized:
            hidden.append(f"{len(oversized)} too big")
        if undersized and not include_undersized:
            hidden.append(f"{len(undersized)} too small")
        if base_hidden and not include_base:
            hidden.append(f"{len(base_hidden)} base")
        msg = "   HuggingFace candidates: none usable for this GPU"
        if hidden:
            msg += (f" ({', '.join(hidden)} hidden; "
                    f"--include-oversized/-undersized/-base to show)")
        lines.append(msg)
        return
    if newer:
        lines.append("   HuggingFace -- NEWER versions (not yet a family):")
        for c in sorted(newer, key=lambda c: (c.version, c.downloads),
                        reverse=True):
            lines.append(_fmt_hf_row(c))
    if gap:
        lines.append("   HuggingFace -- version GAPs (untracked between tracked):")
        for c in sorted(gap, key=lambda c: (c.version, c.downloads),
                        reverse=True):
            lines.append(_fmt_hf_row(c))
    if same:
        # Group by the home family the SAME repo maps to; show the most
        # downloaded few unless --include-same.
        by_family: dict[str, list[HFCandidate]] = {}
        for c in same:
            by_family.setdefault(c.mapping, []).append(c)
        lines.append("   HuggingFace -- new repos at versions you already track:")
        for mapping in sorted(by_family):
            group = sorted(by_family[mapping], key=lambda c: c.downloads,
                           reverse=True)
            shown = group if include_same else group[:SAME_PER_FAMILY]
            for c in shown:
                lines.append(_fmt_hf_row(c))
            extra = len(group) - len(shown)
            if extra > 0:
                lines.append(f"            (+{extra} more {mapping}; "
                             f"--include-same to list)")
    hidden_notes = []
    if oversized and not include_oversized:
        hidden_notes.append(f"{len(oversized)} too big (--include-oversized)")
    if undersized and not include_undersized:
        hidden_notes.append(f"{len(undersized)} too small "
                            f"(--include-undersized)")
    if base_hidden and not include_base:
        hidden_notes.append(f"{len(base_hidden)} base/non-chat "
                            f"(--include-base)")
    if hidden_notes:
        lines.append(f"   (hidden: {'; '.join(hidden_notes)})")


def render_report(results: list[dict], *, do_hf: bool, do_ollama: bool,
                  include_same: bool = False, include_oversized: bool = False,
                  include_undersized: bool = False, include_base: bool = False,
                  gpu_budget: float = DEFAULT_GPU_MEMORY_GB,
                  vram_tolerance: float = DEFAULT_VRAM_TOLERANCE,
                  min_vram_frac: float = DEFAULT_MIN_VRAM_FRAC) -> str:
    lines: list[str] = []
    lines.append("# catalog-discover (read-only) -- newer-version candidates")
    lines.append("# source of truth: scripts/model-families.yaml")
    lines.append("# probe-cache is truth; treat every row below as a candidate")
    lines.append("# to run through `make probe` before adding. NO auto-edits.")
    lines.append(f"# VRAM band: GPU budget {gpu_budget:.0f} GB; keep candidates "
                 f">= {min_vram_frac:.0%} of it and <= {vram_tolerance:g}x the "
                 f"family's largest model.")
    lines.append("")

    total_hf = sum(1 for r in results for c in r["hf"]
                   if c.in_range and not c.base)
    total_ol = sum(1 for r in results for c in r["ollama"] if c.status == 200)
    skipped = [r for r in results if r.get("skip_reason")]

    for r in results:
        lin: Lineage = r["lineage"]
        head = (f"== lineage {lin.brand}{lin.suffix or ''}  "
                f"(families: {', '.join(sorted(lin.family_names))})")
        lines.append(head)
        tracked = ", ".join(version_str(v) for v in sorted(lin.all_versions)) or "-"
        authors = ", ".join(effective_authors(lin)) or "-"
        lines.append(f"   tracked versions: {tracked}")
        lines.append(f"   trusted authors : {authors}")
        floor, ceiling = lin.vram_band_gb(gpu_budget, vram_tolerance,
                                          min_vram_frac)
        ests = lin.tracked_vram_estimates()
        basis = (f" (family max ~{max(ests):.0f} GB)" if ests
                 else " (GPU budget; no tracked size)")
        lines.append(f"   VRAM band       : ~{floor:.0f}-{ceiling:.0f} GB{basis}")
        if r.get("skip_reason"):
            lines.append(f"   [skipped] {r['skip_reason']}")
            lines.append("")
            continue

        if do_hf:
            _render_hf(r["hf"], lines, include_same=include_same,
                       include_oversized=include_oversized,
                       include_undersized=include_undersized,
                       include_base=include_base)
        if do_ollama:
            ol: list[OllamaCandidate] = r["ollama"]
            existing = [c for c in ol if c.status == 200]
            # The next couple of not-yet-published NUMERIC versions, as a
            # "watch" signal that they were probed (qwen3.7 -> 404). Filter
            # to versioned candidates so version=() EXPLICIT names (from
            # discover.ollama_names) can't crowd out the numeric signal.
            pending = sorted((c for c in ol if c.status == 404 and c.version),
                             key=lambda c: c.version)[:2]
            explicit_404 = [c for c in ol if c.status == 404 and not c.version]
            # Probes that neither resolved nor 404'd (network/DNS/proxy) --
            # surface them so an outage never reads as "nothing to discover".
            failed = [c for c in ol if c.status not in (200, 404)]
            shown = existing + pending + explicit_404
            if shown:
                lines.append("   Ollama candidates:")
                for c in shown:
                    vs = version_str(c.version) if c.version else "-"
                    tag = "-> " + c.mapping if c.status == 200 else ""
                    lines.append(
                        f"     {c.library:<24} v{vs:<5} "
                        f"{_status_label(c.status):<18} {tag}")
            if failed:
                lines.append(f"   Ollama: {len(failed)} probe(s) failed "
                             f"(network/proxy?) -- re-run; not 'no candidates'.")
        lines.append("")

    total_over = sum(1 for r in results for c in r["hf"] if c.oversized)
    total_under = sum(1 for r in results for c in r["hf"] if c.undersized)
    total_base = sum(1 for r in results for c in r["hf"]
                     if c.base and c.in_range)
    lines.append("# -----------------------------------------------------------")
    lines.append(f"# {total_hf} usable HF candidate(s) (hidden: {total_over} "
                 f"too big + {total_under} too small + {total_base} base), "
                 f"{total_ol} existing untracked Ollama librar(ies) across "
                 f"{len(results) - len(skipped)} active lineage(s); "
                 f"{len(skipped)} skipped.")
    if skipped:
        lines.append("# skipped lineages (no derivable version + no discover: "
                     "block):")
        for r in skipped:
            lin = r["lineage"]
            lines.append(f"#   {lin.brand}{lin.suffix or ''} "
                         f"({', '.join(sorted(lin.family_names))})")
        lines.append("# -> add a `discover:` block to enable discovery for "
                     "these (see model-families.yaml header).")
    lines.append("#")
    lines.append("# next steps for any candidate worth pursuing:")
    lines.append("#   1. Verify the repo/tag on the HF or Ollama web UI.")
    lines.append("#   2. Add it (confirmed) under the matching family:")
    lines.append("#        make catalog-discover-add ADD=<repo>      # one repo")
    lines.append("#        make catalog-discover-add                 # confirm each")
    lines.append("#      (or edit scripts/model-families.yaml by hand: HF -> "
                 "hf_repos, GGUF -> gguf_repos, Ollama -> ollama_repos)")
    lines.append("#   3. make catalog-regen     # refresh deploy/models.yaml")
    lines.append("#   4. make model-pull FAMILY=<name>")
    lines.append("#   5. make probe / probe-vllm / probe-sglang   # NOT optional")
    return "\n".join(lines)


def to_json(results: list[dict], *, do_hf: bool, do_ollama: bool) -> str:
    payload: list[dict] = []
    for r in results:
        lin: Lineage = r["lineage"]
        item: dict = {
            "lineage": f"{lin.brand}{lin.suffix or ''}",
            "families": sorted(lin.family_names),
            "tracked_versions": [version_str(v) for v in sorted(lin.all_versions)],
            "trusted_authors": effective_authors(lin),
            "skipped": r.get("skip_reason"),
        }
        if do_hf:
            item["hf_candidates"] = [{
                "repo": c.repo, "version": version_str(c.version),
                "class": c.klass, "mapping": c.mapping, "format": c.fmt,
                "likes": c.likes, "downloads": c.downloads,
                "created_at": c.created_at, "conversational": c.conversational,
                "params_b": c.params_b, "est_vram_gb": c.est_vram_gb,
                "vram_measured": c.vram_measured,
                "oversized": c.oversized, "undersized": c.undersized,
                "base": c.base, "in_range": c.in_range,
            } for c in r["hf"]]
        if do_ollama:
            item["ollama_candidates"] = [{
                "library": c.library,
                "version": version_str(c.version) if c.version else None,
                "status": c.status, "exists": c.status == 200,
                "class": c.klass, "mapping": c.mapping,
            } for c in r["ollama"]]
        payload.append(item)
    return json.dumps({"lineages": payload}, indent=2)


# == Add (confirmed mutation of model-families.yaml) ==========================
#
# The ONLY writer of model-families.yaml. Inserts a single repo entry under an
# EXISTING family, preserving every comment (line-based insertion, not a YAML
# round-trip). New families (NEWER versions) are out of scope -- they need
# arch_ref/parsers/thinking curation the tool cannot infer.

_GGUF_SIZE_RE = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?[Bb](?:-[Aa]\d+(?:\.\d+)?[Bb])?)")


def _gguf_tag_prefix(basename: str) -> str | None:
    """Derive a gguf tag_prefix from the size token: 'Qwen3.6-35B-A3B' ->
    '35b-a3b', 'Qwen3.5-27B' -> '27b'."""
    m = _GGUF_SIZE_RE.search(basename)
    return m.group(1).lower() if m else None


def format_entry(repo: str, list_key: str) -> list[str]:
    """The YAML lines (6-space indent) for a new repo entry."""
    if list_key == "gguf_repos":
        out = [f"      - repo: {repo}"]
        tp = _gguf_tag_prefix(repo_basename(repo))
        if tp:
            out.append(f"        tag_prefix: {tp}")
        out.append("        include: []   # TODO: quant allowlist e.g. "
                   "UD-Q3_K_XL ([] emits nothing)")
        return out
    return [f"      - {repo}"]   # hf_repos / ollama_repos: bare string item


def insert_repo_entry(text: str, family: str, list_key: str,
                      entry_lines: list[str]) -> str:
    """Insert entry_lines under family's list_key, creating the key if absent.

    Line-based so every comment survives. Raises KeyError if the family is
    not present.
    """
    lines = text.split("\n")
    fam_re = re.compile(rf"^  - name:\s*['\"]?{re.escape(family)}['\"]?\s*$")
    start = next((i for i, l in enumerate(lines) if fam_re.match(l)), None)
    if start is None:
        raise KeyError(f"family {family!r} not found")
    # The family block ends at the next top-level list item ('  - ') or EOF.
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"^  - ", lines[i]):
            end = i
            break
    key_re = re.compile(rf"^    {re.escape(list_key)}:\s*$")
    key_idx = next((i for i in range(start, end) if key_re.match(lines[i])), None)
    if key_idx is None:
        # Create the key right after the `- name:` line (key order is free).
        lines[start + 1:start + 1] = [f"    {list_key}:"] + entry_lines
        return "\n".join(lines)
    # Insert after the last real (non-comment, non-blank) item line of the list.
    insert_at = key_idx
    for i in range(key_idx + 1, end):
        stripped = lines[i].strip()
        if not stripped:
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        if indent <= 4:
            break  # next family-level key -> the list ended
        if not stripped.startswith("#"):
            insert_at = i
    lines[insert_at + 1:insert_at + 1] = entry_lines
    return "\n".join(lines)


def _family_for_version(lineages: dict[str, Lineage], brand: str, suffix: str,
                        version: tuple[int, ...] | None) -> str | None:
    lin = lineages.get(lineage_key(brand, suffix))
    if lin and version is not None and version in lin.version_to_family:
        return lin.version_to_family[version]
    return None


def plan_add(repo: str, lineages: dict[str, Lineage],
             tracked_all: set[str]) -> dict:
    """Resolve a repo id to {family, list_key, entry} or {error}.

    Existing families only: a repo whose version has no family yet (a NEWER
    version) is refused rather than creating a family.
    """
    repo = repo.strip()
    if not repo:
        return {"error": "empty repo id"}
    if repo.lower() in tracked_all:
        return {"error": "already tracked"}
    if "/" not in repo:   # Ollama library name, e.g. 'qwen3.6'
        brand, version, suffix = parse_family_lineage(repo)
        fam = _family_for_version(lineages, brand, suffix, version)
        if not fam:
            return {"error": f"no existing family tracks '{repo}' "
                             f"(new family -- out of scope)"}
        return {"family": fam, "list_key": "ollama_repos",
                "entry": format_entry(repo, "ollama_repos")}
    base = repo_basename(repo)
    author = repo_author(repo)
    matches = []
    for lin in lineages.values():
        ver = extract_repo_version(base, lin.brand)
        if ver is None or ver not in lin.version_to_family:
            continue
        if not repo_matches_lineage(base, lin, lineage_name_re(lin),
                                    strict=lin.discover.name_regex is None,
                                    author=author):
            continue
        matches.append((lin, ver))
    if not matches:
        return {"error": f"no existing family for {repo} "
                         f"(version not tracked -> new family, out of scope)"}
    lin, ver = max(matches, key=lambda m: len(m[0].suffix))  # prefer sub-lineage
    list_key = "gguf_repos" if _is_gguf_repo(base) else "hf_repos"
    return {"family": lin.version_to_family[ver], "list_key": list_key,
            "entry": format_entry(repo, list_key)}


def all_tracked(lineages: dict[str, Lineage]) -> set[str]:
    """Every already-tracked id (lowercased) across families -- hf/gguf repo
    ids AND ollama library names. ollama libs live in lin.ollama_libs, not
    tracked_repos, so they must be unioned in or an already-tracked library
    slips past plan_add's guard and gets appended twice."""
    out: set[str] = set()
    for lin in lineages.values():
        out |= lin.tracked_repos
        out |= lin.ollama_libs
    return out


def _is_gguf_repo(basename: str) -> bool:
    """gguf-vs-safetensors from the name alone (no tags on the add path).

    A '-GGUF' token is the usual signal; also treat a k-quant / imatrix
    marker (Q4_K_M, IQ3_XXS, UD-Q3, imat) as gguf when no other format
    marker won -- safetensors quants (NVFP4/FP8/AWQ/GPTQ) take precedence
    in detect_format, so this only fires on otherwise-unmarked names.
    """
    fmt = detect_format(basename, [])
    if fmt == "GGUF":
        return True
    return fmt == "?" and bool(re.search(
        r"(?i)(?:^|[-_.])(?:i?q\d[_a-z0-9]*|imat)(?:$|[-_.])", basename))


def _count_in_family(data: object, family: str, list_key: str,
                     repo: str) -> int:
    families = (data or {}).get("families", []) if isinstance(data, dict) else []
    n = 0
    for fam in families:
        if fam.get("name") != family:
            continue
        for item in fam.get(list_key) or []:
            r = item.get("repo") if isinstance(item, dict) else item
            if str(r).strip().lower() == repo.lower():
                n += 1
    return n


def _repo_in_family(data: object, family: str, list_key: str,
                    repo: str) -> bool:
    return _count_in_family(data, family, list_key, repo) > 0


def _validate_insertion(text: str, family: str, list_key: str,
                        repo: str) -> tuple[bool, str]:
    try:
        data = yaml.safe_load(text)
    except Exception as e:
        return False, f"result no longer parses as YAML: {e}"
    # A fresh add must leave EXACTLY one occurrence: 0 means the insert was
    # lost, >1 means we duplicated an already-tracked entry. Both fail closed.
    count = _count_in_family(data, family, list_key, repo)
    if count == 0:
        return False, "entry not found after insert"
    if count > 1:
        return False, f"{repo} already present in {family} ({list_key})"
    return True, ""


def _confirm(msg: str) -> bool:
    try:
        return input(f"{msg} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False   # non-interactive without --yes -> decline


def do_add(repo: str, families_path: Path, *, assume_yes: bool) -> int:
    """Add one specific repo (the --add <repo> path)."""
    text = families_path.read_text()
    lineages = build_lineages(load_families(families_path))
    tracked_all = all_tracked(lineages)
    plan = plan_add(repo, lineages, tracked_all)
    if "error" in plan:
        print(f"cannot add {repo}: {plan['error']}", file=sys.stderr)
        return 1
    fam, key, entry = plan["family"], plan["list_key"], plan["entry"]
    print(f"add {repo}")
    print(f"  -> family {fam} ({key})")
    for line in entry:
        print(f"  | {line}")
    if not assume_yes and not _confirm(f"Write to {families_path.name}?"):
        print("skipped (no changes).")
        return 0
    new_text = insert_repo_entry(text, fam, key, entry)
    ok, msg = _validate_insertion(new_text, fam, key, repo)
    if not ok:
        print(f"refusing to write -- {msg}", file=sys.stderr)
        return 1
    families_path.write_text(new_text)
    print(f"  added to {families_path}")
    if key == "gguf_repos":
        print("  NOTE: fill in the `include:` quant allowlist (now [] = nothing).")
    print("  next: make catalog-regen && make probe   # probe before relying on it")
    return 0


def interactive_add(results: list[dict], families_path: Path, *,
                    assume_yes: bool) -> int:
    """Walk the discovery results and confirm each addable candidate."""
    text = families_path.read_text()
    lineages = build_lineages(load_families(families_path))
    tracked_all = all_tracked(lineages)
    ids: list[str] = []
    for r in results:
        for c in sorted((c for c in r["hf"]
                         if c.klass == "SAME" and c.in_range and not c.base),
                        key=lambda c: c.downloads, reverse=True):
            ids.append(c.repo)
        ids.extend(c.library for c in r["ollama"] if c.status == 200)
    seen: set[str] = set()
    ids = [x for x in ids if not (x.lower() in seen or seen.add(x.lower()))]
    if not ids:
        print("no addable in-range candidates "
              "(NEWER versions need a new family -- out of scope).")
        return 0
    added = 0
    for rid in ids:
        if rid.lower() in tracked_all:
            continue
        plan = plan_add(rid, lineages, tracked_all)
        if "error" in plan:
            continue
        fam, key = plan["family"], plan["list_key"]
        if not assume_yes and not _confirm(f"Add {rid} -> {fam} ({key})?"):
            continue
        candidate = insert_repo_entry(text, fam, key, plan["entry"])
        ok, msg = _validate_insertion(candidate, fam, key, rid)
        if not ok:
            print(f"  ! skipped {rid}: {msg}", file=sys.stderr)
            continue
        text = candidate
        families_path.write_text(text)
        tracked_all.add(rid.lower())
        print(f"  added {rid} -> {fam}")
        added += 1
    plural = "y" if added == 1 else "ies"
    print(f"\nWrote {added} entr{plural} to {families_path.name}.")
    if added:
        print("Next: make catalog-regen && make probe")
    return 0


# == Main =====================================================================

def load_families(path: Path) -> list[dict]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    return (data or {}).get("families", []) or []


def run(families: list[dict], *, family_filter: str | None, do_hf: bool,
        do_ollama: bool, hf_limit: int, ollama_probe: int, timeout: int,
        gpu_budget: float = DEFAULT_GPU_MEMORY_GB,
        vram_tolerance: float = DEFAULT_VRAM_TOLERANCE,
        min_vram_frac: float = DEFAULT_MIN_VRAM_FRAC) -> list[dict]:
    lineages = build_lineages(families)
    results: list[dict] = []
    for key in sorted(lineages):
        lin = lineages[key]
        if family_filter and not any(
                family_filter.lower() in fn.lower() for fn in lin.family_names):
            continue
        r: dict = {"lineage": lin, "hf": [], "ollama": []}
        if not lin.discover.enabled:
            r["skip_reason"] = "discovery disabled via discover.enabled: false"
            results.append(r)
            continue
        has_version = bool(lin.all_versions)
        has_override = bool(lin.discover.name_regex or lin.discover.hf_authors
                            or lin.discover.ollama_names)
        if not has_version and not has_override:
            r["skip_reason"] = ("no numeric version in family name(s) and no "
                                "discover: block")
            results.append(r)
            continue
        if do_hf:
            r["hf"] = discover_hf(lin, hf_limit=hf_limit, timeout=timeout,
                                  gpu_budget=gpu_budget,
                                  vram_tolerance=vram_tolerance,
                                  min_vram_frac=min_vram_frac)
        if do_ollama:
            r["ollama"] = discover_ollama(
                lin, probe_count=ollama_probe, timeout=timeout)
        results.append(r)
    return results


def gpu_budget_from_env(default: float = DEFAULT_GPU_MEMORY_GB) -> float:
    """Read GPU_MEMORY_GB (exported by the Makefile / .env), default 24."""
    try:
        return float(os.environ.get("GPU_MEMORY_GB") or default)
    except ValueError:
        return default


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Discover newer-version members of tracked model lineages "
                    "(read-only).")
    ap.add_argument("--family", default=None,
                    help="Only report lineages whose family name contains this "
                         "substring (case-insensitive).")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of the report.")
    ap.add_argument("--no-hf", action="store_true", help="Skip HuggingFace search.")
    ap.add_argument("--no-ollama", action="store_true",
                    help="Skip Ollama library probing.")
    ap.add_argument("--include-same", action="store_true",
                    help="List every untracked repo at an already-tracked "
                         "version (default: collapse to a per-family count).")
    ap.add_argument("--include-oversized", action="store_true",
                    help="Also show candidates above the VRAM band (too big to "
                         "load on the GPU; hidden by default).")
    ap.add_argument("--include-undersized", action="store_true",
                    help="Also show candidates below the VRAM band (too small "
                         "-- they waste the GPU; hidden by default).")
    ap.add_argument("--include-base", action="store_true",
                    help="Also show base / non-chat (pretraining) checkpoints "
                         "(hidden by default; the lab wants instruct models).")
    ap.add_argument("--max-vram", type=float, default=None,
                    help="GPU VRAM budget in GB (default: $GPU_MEMORY_GB or 24).")
    ap.add_argument("--vram-tolerance", type=float,
                    default=DEFAULT_VRAM_TOLERANCE,
                    help="How far above the family's largest model a candidate "
                         f"may go (default {DEFAULT_VRAM_TOLERANCE:g}x).")
    ap.add_argument("--min-vram-frac", type=float,
                    default=DEFAULT_MIN_VRAM_FRAC,
                    help="Hide candidates using less than this fraction of the "
                         f"GPU (default {DEFAULT_MIN_VRAM_FRAC:g} = 50%%).")
    ap.add_argument("--hf-limit", type=int, default=50,
                    help="Max HF results per author search (default 50).")
    ap.add_argument("--ollama-probe", type=int, default=4,
                    help="How many forward minor versions to probe on Ollama "
                         "(default 4).")
    ap.add_argument("--timeout", type=int, default=25,
                    help="Per-request network timeout in seconds (default 25).")
    ap.add_argument("--families-file", default=str(FAMILIES_YAML),
                    help="Path to model-families.yaml (default: repo copy).")
    ap.add_argument("--add", nargs="?", const="__INTERACTIVE__", default=None,
                    metavar="REPO",
                    help="Add a candidate to model-families.yaml (existing "
                         "families only). `--add <repo>` adds one repo by id; "
                         "`--add` with no value walks the discovered candidates "
                         "and confirms each. WRITES the file.")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="Skip the per-add confirmation prompt (for scripting).")
    args = ap.parse_args(argv)

    path = Path(args.families_file)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 1
    families = load_families(path)

    # Non-interactive single-repo add: no discovery network calls needed.
    if args.add is not None and args.add != "__INTERACTIVE__":
        return do_add(args.add, path, assume_yes=args.yes)

    do_hf = not args.no_hf
    do_ollama = not args.no_ollama
    gpu_budget = args.max_vram if args.max_vram else gpu_budget_from_env()
    results = run(
        families, family_filter=args.family, do_hf=do_hf, do_ollama=do_ollama,
        hf_limit=args.hf_limit, ollama_probe=args.ollama_probe,
        timeout=args.timeout, gpu_budget=gpu_budget,
        vram_tolerance=args.vram_tolerance, min_vram_frac=args.min_vram_frac)

    # Interactive add: confirm each discovered candidate, then exit.
    if args.add == "__INTERACTIVE__":
        return interactive_add(results, path, assume_yes=args.yes)

    if args.json:
        print(to_json(results, do_hf=do_hf, do_ollama=do_ollama))
    else:
        print(render_report(results, do_hf=do_hf, do_ollama=do_ollama,
                            include_same=args.include_same,
                            include_oversized=args.include_oversized,
                            include_undersized=args.include_undersized,
                            include_base=args.include_base,
                            gpu_budget=gpu_budget,
                            vram_tolerance=args.vram_tolerance,
                            min_vram_frac=args.min_vram_frac))
    return 0


if __name__ == "__main__":
    sys.exit(main())
