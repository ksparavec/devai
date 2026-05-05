"""Re-export of scripts/bench/_bench_core helpers for the pearls harness.

The pearls harness shares cache shape, streaming-HTTP, percentile, leak,
and router-resolution helpers with the main bench. Keeping the imports
local to this module means tasks/pearls.py and bench_runner.py can do
``from bench_pearls._bench_core import ...`` without reaching into the
sibling harness directly -- the plug-in path later is a one-line change
(switch ``bench_pearls`` to ``bench`` in the import) instead of a fork.

Two things differ from the main bench:

- ``DEFAULT_CACHE_PATH`` points at ``deploy/.bench-pearls-cache.json``.
  v1 ships a separate cache file so a bad pearls run can't pollute the
  main leaderboard. When the harness is folded into ``scripts/bench/``,
  drop this override and rows merge into ``.bench-cache.json``.
- ``DATA_DIR`` points at this package's ``data/`` so problem JSONL is
  resolved relative to the pearls harness, not the main bench.

Everything else is forwarded verbatim from the sibling module.
"""

from __future__ import annotations

from pathlib import Path

# Sibling module is on sys.path because both packages live under
# scripts/ and the runner inserts scripts/ at the front of sys.path
# before importing. No try/except: if the sibling import fails the
# harness can't run, and we want the traceback.
from bench._bench_core import (  # noqa: F401
    cache_key_for_entry,
    http_post_stream,
    load_leak_patterns,
    migrate_bench_cache_keys,
    p50,
    p95,
    percentile,
    router_url_for,
    serving_alias,
    serving_alias_with_ctx,
    stream_chat_completion,
    sweep_for_leaks,
    update_row,
)

# Override the cache path. The main bench writes
# deploy/.bench-cache.json; the pearls harness writes
# deploy/.bench-pearls-cache.json so v1 can be torn down without
# affecting the main leaderboard. Same shape, same key form.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_PATH = REPO_ROOT / "deploy" / ".bench-pearls-cache.json"

# Override the data dir so tasks/pearls.py loads JSONL from this
# package, not from scripts/bench/data/.
DATA_DIR = Path(__file__).resolve().parent / "data"
