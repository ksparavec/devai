# scripts/stats -- reproducible re-analyses of the experiment docs

Every interval, test and derived number in the experiment write-ups is
computed by the scripts in this directory. The write-ups are
[`docs/bench-results.md`](../../docs/bench-results.md),
[`docs/backends.md`](../../docs/backends.md),
[`docs/laya-trainer.md`](../../docs/laya-trainer.md),
[`docs/multi-token-prediction.md`](../../docs/multi-token-prediction.md),
[`docs/llm-tokens-and-speed.md`](../../docs/llm-tokens-and-speed.md) and
[`docs/nvfp4-coldstart.md`](../../docs/nvfp4-coldstart.md). The methods are
explained in
[`docs/statistics-primer.md`](../../docs/statistics-primer.md).

All scripts:

- use the Python 3 standard library only;
- are read-only on their inputs;
- are deterministic, with fixed bootstrap seeds (20260927 unless stated);
- write JSON (every number) plus Markdown (tables) to the output directory
  given.

The inputs are this host's retained data. The inspect_ai logs, the persisted
container logs and the laya store are listed per script below. Nothing here
starts a container, touches the GPU or uses the network.

## Shared library

| File | What |
| --- | --- |
| `statlib.py` | Clopper-Pearson (two- and one-sided), Wilson, hypergeometric (finite population), exact binomial tests, exact McNemar, Newcombe method 10 (paired and unpaired), Holm, sign-flip tests, exact McNemar power and minimum detectable effect, Garwood Poisson interval, quantile definitions, distribution-free median and quantile intervals, bootstraps (plain, two-sample, stratified, paired), AUROC with Hanley-McNeil / DeLong / cluster-bootstrap intervals. Tested against published values by `tests/python/test_stats_statlib.py`. |
| `primer_examples.py` | Recomputes every worked example in the primer; `tests/python/test_stats_primer_examples.py` fails if the primer's text drifts from it. |

## Benchmark scores (docs/bench-results.md, docs/backends.md, docs/router.md)

**Log format changed with the inspect-ai 0.3.271 pin (2026-09-27).** Logs
written by 0.3.158 (every log before that date) are deflate-compressed
zip files; 0.3.271 writes its `.eval` archives with **zstd** (zip
compression method 93), which Python's `zipfile` reads only from Python
3.14 on. The host's Python 3.13 raises `NotImplementedError: That
compression method is not supported` on them, so run the log readers
(`bench_extract_logs.py`, `bench_stats.py`, `kvquant_*`, ...) inside the
lab image (Python 3.14.7) once logs from the re-bench are in the
directory, e.g. `podman run --rm --network=none --entrypoint python3
-v $PWD:/repo:ro -v $L:$L:ro -v $O:$O localhost/devai-lab-gpu:latest
/repo/scripts/stats/bench_extract_logs.py $L $O/logs_extracted.json`.

```bash
L=/var/cache/devai/bench/inspect-logs; O=~/.cache/devai/stats/bench; mkdir -p $O
python3 scripts/stats/bench_extract_logs.py $L $O/logs_extracted.json
python3 scripts/stats/bench_stats.py $O/logs_extracted.json $L deploy/.bench-cache.json $O
python3 scripts/stats/bench_scorer_scan.py $O/logs_extracted.json $L $O/scorer_artifact_scan.json
python3 scripts/stats/bench_engine_facts.py /var/cache/devai/logs $O/engine_log_facts.json
python3 scripts/stats/bench_timeouts.py $L $O/timeouts.tsv
```

- `bench_extract_logs.py`: per-item outcomes, item content hashes, time-limit
  hits and request bodies from every inspect_ai `.eval` log.
- `bench_stats.py`: the leaderboard intervals, pairwise comparisons (Holm),
  threshold checks, change claims, vLLM-vs-SGLang tests, incidental
  replications and the current-cache table with time-out bounds.
- `bench_scorer_scan.py`: HumanEval failures caused by the v2 extractor's
  indentation stripping (parse-only; model code is never executed; fence
  path only, so it undercounts).
- `bench_rescore_humaneval.py`: re-executes every logged HumanEval /
  HumanEval+ answer whose extraction the fixed extractor changes, in the
  scorer's own sandbox, and reports each run's corrected value and the
  port-11435 vs 11437 AutoRound pair (docs/router.md). It EXECUTES model
  code and imports inspect_ai, so run it inside the lab image:
  `podman run --rm --network=none --entrypoint python3 -v $PWD:/repo:ro
  -v $L:/logs:ro -v $O:/o localhost/devai-lab-gpu:latest
  /repo/scripts/stats/bench_rescore_humaneval.py /o/logs_extracted.json
  /logs /o/humaneval_rescore.json`.
- `bench_engine_facts.py`: the sampler defaults and KV dtype each engine
  applied, from its own log.
- `bench_timeouts.py`: every log with a sample that hit the per-sample time
  limit.

## Use-case scores (owner's four use cases)

```bash
L=/var/cache/devai/bench/inspect-logs; O=~/.cache/devai/stats/usecase; mkdir -p $O
podman run --rm --network=none --entrypoint python3 -v $PWD:/repo:ro -v $L:$L:ro -v $O:$O \
    localhost/devai-lab-gpu:latest /repo/scripts/stats/bench_extract_logs.py $L $O/logs_extracted.json
python3 scripts/stats/usecase_scores.py $O/logs_extracted.json deploy/.bench-cache.json $O
```

- `usecase_scores.py` regroups benchmark ITEMS by use case, using the
  per-question tags the extractor keeps (MMLU-Pro `category`, GPQA
  `subdomain`). Mapping ("Scheme A", owner's choice 2026-09-27; no item
  counts twice, every MMLU-Pro category exactly once): coding =
  HumanEval+ and HumanEval (one cluster per problem) + MMLU-Pro computer
  science; general reasoning = MMLU-Pro law, history, philosophy,
  psychology, economics, business, health, other; problem analysis =
  GSM8K + MMLU-Pro math, physics, chemistry, engineering; complex systems
  = GPQA-Diamond + MMLU-Pro biology. Score = pooled share correct
  (time-outs scored wrong, counted beside it); Clopper-Pearson, or a
  cluster bootstrap for coding; winner vs runner-up paired on identical
  items (cluster bootstrap interval, sign-flip test, Holm over the four
  use cases). Ranked by quality, never by speed; the runner-up is the
  best row of a different base model. Each task entry is joined to its
  log by the entry's `inspect_log` name (older entries: by completion
  time); a task stopped at its deadline counts only its unbroken prefix
  (`truncated.prefix`). The extractor also reads header-less logs (left
  by SIGKILL). Writes `usecase_scores.{json,md}`.
  Tested by `tests/python/test_stats_usecase_scores.py`.

## KV-cache dtype (docs/backends.md "Per-tier KV-cache dtype")

```bash
python3 scripts/stats/kvquant_stats.py /var/cache/devai/bench/inspect-logs ~/.cache/devai/stats/kvquant
```

- `kvquant_stats.py` pairs the f16 and q8_0 (Ollama) and the fp8 and auto
  (vLLM) runs on verified-identical items.
- `kvquant_extract_samples.py` is its per-sample reader.

## Throughput, latency, MTP, cold start (the performance docs)

```bash
O=~/.cache/devai/stats/perf; mkdir -p $O; cd scripts/stats
python3 perf_bench_cache.py ../../deploy/.bench-cache.json $O/
python3 perf_engine_runs.py /var/cache/devai/logs/devai-vllm.log /var/cache/devai/logs/devai-vllm-devai.log ../bench/data/latency_prompts.jsonl $O/
python3 perf_ceiling.py /var/cache/devai/vllm $O/engine_runs.json $O/
python3 perf_coldstart.py /var/cache/devai/logs/devai-router.log /var/cache/devai/logs/devai-vllm-devai.log $O/
python3 perf_ts.py /var/cache/devai/logs/devai-vllm-devai.log ~/.cache/devai/stats/ts/measure-2026-09-26.json $O/
python3 perf_client_logs.py ~/.cache/devai /var/cache/devai/logs/devai-vllm.log $O/
```

- `perf_engine_runs.py` uses the per-request "Request finished" lines that
  the home-built vLLM image logs: engine-counted tokens and elapsed time.
  These give the paired MTP speedup, the single-stream, aggregate and depth
  figures, and the prefill rates.
- `perf_ceiling.py` re-derives the bandwidth-bound decode ceiling from the
  safetensors headers.
- `perf_coldstart.py` measures router launch-to-ready times (right-censored
  at the health timeout) and lists every launch it sets aside (never ready,
  engine failure, superseded, router restart).
- The persisted container logs contain blocks the logger sidecar replayed
  after restarts; `perf_coldstart.py`, `perf_phases.py` and
  `perf_enginelog.py` count each raw log line (timestamp included) once.
- `perf_ts.py` covers the 2026-09-26 t/S measurement for aiagent. The client
  summary is kept at `~/.cache/devai/stats/ts/measure-2026-09-26.json`.
- `perf_phases.py` (`python3 perf_phases.py /var/cache/devai/logs/devai-router.log /var/cache/devai/logs/devai-vllm.log /var/cache/devai/logs/devai-vllm-devai.log $O/ /var/cache/devai/bench/bench-vllm-run-20260505T140328Z.log`): vLLM start-up phases and memory lines (weights, model load, compile, KV pool, graph pool, peak activation, effective GPU-memory utilisation) joined to the router launches, never-ready and restart-abandoned launches, the MTP memory cost (weights, KV bytes per token, KV pool), CUDA-graph capture sizes, and the 2026-05-05 bench-run summary lines.
- `perf_enginelog.py` holds the log parsers.

## laya trainer (docs/laya-trainer.md, docs/plans/laya-trainer.md)

```bash
python3 scripts/stats/laya_stats.py      # defaults point at /var/cache/devai/laya, ~/devai-home, ~/laya-pilot, ~/git/aiagent
```

This covers the gate recomputation, the fresh500 confirmation, the shadow
batch, the zero-shot McNemar family, the AUROC intervals, the hold-time cost
model and the teacher cold starts. Output goes to
`~/.cache/devai/stats/laya/` unless `--out` is given.
