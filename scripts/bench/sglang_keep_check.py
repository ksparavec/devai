#!/usr/bin/env python3
"""Phase 15's keep-or-freeze measurement for SGLang (platform-rebuild I16).

One checkpoint served by both engines -- Qwen3.5-9B-NVFP4 (I15) -- launched
by this script through the probers' container driver, outside the router,
with the stack down:

  sglang          SGLang as the router serves it (radix cache on)
  sglang-noradix  the same with --disable-radix-cache
  vllm            the vllm-devai backend's vLLM (0.30.0, E1)

Each at one context (the smaller of the two probed maxima), with the KV
dtype its probe cell is stamped with, MTP off, the probers' command args
and recovery flags. Before each launch the card must be idle (C8: no
compute process, memory.used at most 512 MiB).

Workload (I16): reasoning off (`chat_template_kwargs` at the top level),
temperature 0, no tools.
  - steady TTFT: the latency prompts, prompt 0 as the warm-up, the median
    over prompts 1-30 (sglang and vllm);
  - turn-2+ TTFT: 20 sessions per configuration (bench_concurrency's
    run_multiturn_sessions: five turns, a 4,096-token prefix, 4,096 tokens
    added per turn, 64 out), each salted <salt>-<config>-<i>;
  - SGLang's engine-reported cache hit rate over turns 2-5 (radix on).
An uncounted dry run comes first: SGLang radix-on, 2 sessions, whose turns
2-5 must show a non-zero hit rate, or nothing is measured.

Decision (I16): keep only when all of
  (a) steady ratio SGLang/vLLM <= 3.0 (point estimate);
  (b) turn-2+ ratio SGLang/vLLM <= 0.75, its 95% interval wholly below 1.0;
  (c) turn-2+ ratio radix-on/radix-off with its interval wholly below 1.0;
  (d) the pooled radix-on hit rate above zero.
Intervals: percentile bootstrap over session values (10,000 resamples).
Any failed request, engine exit or launch failure in an SGLang
configuration ends the run as a freeze; one in the vLLM arm voids it. The
measured run is not repeated.

The JSON report is what M16 cites. The script is removed again once M16 is
recorded (Phase 15, step 4).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from bench import bench_concurrency as BC  # noqa: E402
from bench._bench_core import stream_chat_completion  # noqa: E402
from stats.statlib import boot_ci_two_sample  # noqa: E402

MODEL_REPO = "ykarout/Qwen3.5-9B-NVFP4"
SESSIONS = 20
DRY_SESSIONS = 2
TURNS = 5
STEADY_PROMPTS = 30            # prompts 1-30; prompt 0 is the warm-up
CARD_IDLE_MIB = 512
CARD_SETTLE_S = 120.0
BOOT_REPS = 10000

THRESHOLDS = {"steady_ratio_max": 3.0, "turn2plus_ratio_max": 0.75, "interval_below": 1.0}
BODY_FIELDS = {"chat_template_kwargs": {"enable_thinking": False}}

CONFIGS = ("sglang", "sglang-noradix", "vllm")
SGLANG_CONFIGS = ("sglang", "sglang-noradix")


class ArmFailure(Exception):
    """A failed request, an engine exit or a launch failure in one arm."""

    def __init__(self, config: str, what: str):
        super().__init__(f"{config}: {what}")
        self.config = config
        self.what = what


class CheckRefused(Exception):
    """The environment is not fit to measure (the card is not idle, ...)."""


# --- the decision (pure) -----------------------------------------------------

def _ratio_of_medians(xs: list[float], ys: list[float]) -> float:
    return statistics.median(xs) / statistics.median(ys)


def _ratio_with_interval(num: list[float], den: list[float], reps: int) -> dict:
    lo, hi = boot_ci_two_sample(num, den, _ratio_of_medians, reps=reps)
    return {"ratio": round(_ratio_of_medians(num, den), 4),
            "ci95": [None if lo is None else round(lo, 4), None if hi is None else round(hi, 4)],
            "n": [len(num), len(den)]}


def decide(results: dict, reps: int = BOOT_REPS) -> dict:
    """I16's four conditions over a complete measured run."""
    sg, nr, vl = results["sglang"], results["sglang-noradix"], results["vllm"]
    sv = lambda r: [s["turn2plus_median_ms"] for s in r["sessions"]["sessions"]]  # noqa: E731
    steady = round(sg["steady"]["median_ms"] / vl["steady"]["median_ms"], 4)
    t2 = _ratio_with_interval(sv(sg), sv(vl), reps)
    radix = _ratio_with_interval(sv(sg), sv(nr), reps)
    hit = sg["sessions"]["sglang_hit_rate_pooled"]
    below = THRESHOLDS["interval_below"]
    cond = {
        "a_steady_ratio_at_most_3": steady <= THRESHOLDS["steady_ratio_max"],
        "b_turn2plus_ratio_at_most_0.75_interval_below_1": (
            t2["ratio"] <= THRESHOLDS["turn2plus_ratio_max"]
            and t2["ci95"][1] is not None and t2["ci95"][1] < below),
        "c_radix_on_over_off_interval_below_1": radix["ci95"][1] is not None and radix["ci95"][1] < below,
        "d_pooled_hit_rate_above_0": bool(hit) and hit > 0,
    }
    return {
        "steady_ratio_sglang_over_vllm": steady,
        "turn2plus_sglang_over_vllm": t2,
        "turn2plus_radix_on_over_off": radix,
        "pooled_hit_rate_radix_on": hit,
        "conditions": cond,
        "outcome": "keep" if all(cond.values()) else "freeze",
    }


# --- the measurement (engines injected, so a test can fake them) -------------

@dataclass
class Engine:
    config: str
    base_url: str
    launch: dict = field(default_factory=dict)


def steady_ttft(base_url: str, model: str, prompts: list[dict], config: str,
                body_fields: dict, timeout_s: float = 300.0) -> dict:
    """TTFT of the latency prompts: prompt 0 warms up, 1..STEADY_PROMPTS count."""
    values: list[float] = []
    for i, p in enumerate(prompts[: STEADY_PROMPTS + 1]):
        body = BC._merge_body({"model": model, "messages": [{"role": "user", "content": p["prompt"]}],
                               "max_tokens": int(p.get("max_tokens", 64)), "temperature": 0}, body_fields)
        try:
            res = stream_chat_completion(base_url, body, timeout=timeout_s)
        except Exception as e:                                   # noqa: BLE001
            raise ArmFailure(config, f"steady prompt {i} ({p.get('id')}): {type(e).__name__}: {e}")
        t_first = res.get("t_first_token")
        if t_first is None:
            raise ArmFailure(config, f"steady prompt {i} ({p.get('id')}): no token streamed")
        if i > 0:
            values.append(round((t_first - res["t_open"]) * 1000, 1))
    return {"ttft_ms": values, "median_ms": round(statistics.median(values), 1)}


def run_sessions(engine: Engine, model: str, n: int, salt: str, dry: bool,
                 with_metrics: bool) -> dict:
    out = BC.run_multiturn_sessions(
        model, engine.base_url, sessions=n, engine=engine.config, salt=salt, dry=dry,
        turns=TURNS, metrics_url=f"{engine.base_url}/metrics" if with_metrics else None,
        body_fields=BODY_FIELDS)
    if out["n_failed"]:
        bad = next(s for s in out["sessions"] if s["failed"])
        raise ArmFailure(engine.config, f"session {bad['salt']}: {'; '.join(bad['problems'])}")
    return out


def measure(*, model: str, salt: str, prompts: list[dict],
            start: Callable[[str], Engine], stop: Callable[[Engine], None],
            sessions: int = SESSIONS, dry_sessions: int = DRY_SESSIONS,
            log: Callable[[str], None] = lambda m: print(m, file=sys.stderr)) -> dict:
    """The dry run, then the measured run; returns the report body.

    ``start(config)`` launches one configuration and returns it ready (or
    raises ArmFailure); ``stop`` tears it down. Order: SGLang radix-on (dry
    run, steady, sessions), SGLang radix-off (sessions), vLLM (steady,
    sessions).
    """
    results: dict = {}
    report: dict = {"salt": salt, "results": results}

    def end(outcome: str, config: str, what: str) -> bool:
        report["failure"] = {"config": config, "what": what}
        report["outcome"] = outcome
        log(f"== {config}: {what} -> {outcome}")
        return False

    def arm(config: str, body: Callable[[Engine], dict]) -> bool:
        """One configuration. I16's failure rule: a failed request, an engine
        exit or a launch failure is a freeze in an SGLang arm and voids the
        run in the vLLM arm. Anything else that stops a run -- the card not
        idle before a later launch, an unexpected error -- voids it too, so
        the operator decides; only a refusal before anything was measured
        (the first launch, the dry run) is `refused`. Completed arms are
        kept in every case."""
        engine = None
        try:
            log(f"== {config}: launching")
            engine = start(config)
            results[config] = {"launch": engine.launch}
            results[config].update(body(engine))
            return True
        except ArmFailure as f:
            return end("freeze" if f.config in SGLANG_CONFIGS else "void", f.config, f.what)
        except CheckRefused as e:
            measured = any("sessions" in r for r in results.values())
            return end("void" if measured else "refused", config, f"refused: {e}")
        except Exception as e:                                   # noqa: BLE001
            return end("void", config, f"unexpected {type(e).__name__}: {e}")
        finally:
            if engine is not None:
                stop(engine)

    def sglang_on(e: Engine) -> dict:
        log("   dry run (uncounted)")
        dry = run_sessions(e, model, dry_sessions, salt, dry=True, with_metrics=True)
        report["dry_run"] = dry
        if dry["sglang_hit_rate_pooled"] is None:
            raise CheckRefused(
                "the dry run read no SGLang cached/prompt token counters at /metrics "
                "(is --enable-metrics on?); fix it and repeat the dry run before any measured run")
        if not dry["sglang_hit_rate_pooled"]:
            raise CheckRefused(
                "the dry run's turns 2-5 show a zero SGLang hit rate: the workload does not "
                "reuse prefixes; fix the scorer and repeat the dry run before any measured run")
        log(f"   dry run hit rate {dry['sglang_hit_rate_pooled']}")
        log("   steady TTFT")
        st = steady_ttft(e.base_url, model, prompts, e.config, BODY_FIELDS)
        log(f"   steady median {st['median_ms']} ms; {sessions} sessions")
        return {"steady": st, "sessions": run_sessions(e, model, sessions, salt, dry=False, with_metrics=True)}

    def sglang_off(e: Engine) -> dict:
        log(f"   {sessions} sessions")
        return {"sessions": run_sessions(e, model, sessions, salt, dry=False, with_metrics=True)}

    def vllm(e: Engine) -> dict:
        log("   steady TTFT")
        st = steady_ttft(e.base_url, model, prompts, e.config, BODY_FIELDS)
        log(f"   steady median {st['median_ms']} ms; {sessions} sessions")
        return {"steady": st, "sessions": run_sessions(e, model, sessions, salt, dry=False, with_metrics=True)}

    for config, body in (("sglang", sglang_on), ("sglang-noradix", sglang_off), ("vllm", vllm)):
        if not arm(config, body):
            return report
    report.update(decide(results))
    return report


# --- the real engines (the probers' container driver) ------------------------

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


def card_check() -> str | None:
    """C8's card check; None when the card is idle, else why not."""
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True, check=False)
    used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=False)
    if apps.returncode != 0 or used.returncode != 0:
        return "nvidia-smi failed"
    if apps.stdout.strip():
        return f"compute processes on the card: {apps.stdout.split()}"
    mib = int(used.stdout.split()[0])
    return None if mib <= CARD_IDLE_MIB else f"memory.used {mib} MiB > {CARD_IDLE_MIB}"


class RealEngines:
    """Launch each configuration as the probers do (load probe's path)."""

    def __init__(self, runtime: str, host_vram_gb: int, ctx: int | None):
        self.runtime = runtime
        self.host_vram_gb = host_vram_gb
        self.sg = _load_module("probe_sglang_reasoning", SCRIPTS / "probe-sglang-reasoning.py")
        self.vd = _load_module("probe_vllm_devai_reasoning", SCRIPTS / "probe-vllm-devai-reasoning.py")
        import _probe_core
        import _probe_hf_common
        import _probe_load
        self.hf, self.pl, self.core = _probe_hf_common, _probe_load, _probe_core
        self.rows = {}
        self.cells = {}
        for spec in (self.sg.SPEC, self.vd.SPEC):
            row = next(r for r in self.hf.load_catalog_hf_rows(self.hf.DEFAULT_CATALOG, spec.name)
                       if r.get("repo") == MODEL_REPO)
            cache = json.loads(Path(spec.cache_path).read_text())
            key = next(k for k in cache if k.startswith(MODEL_REPO + "@"))
            entry = cache[key]
            fits = {int(c): cell for c, cell in (entry["probes"].get(str(host_vram_gb)) or {}).items()
                    if cell.get("fits") and cell.get("serving_ok") is not False}
            if not fits:
                raise CheckRefused(f"{MODEL_REPO} has no fitting {spec.name} cell")
            top = max(fits)
            self.rows[spec.name] = row
            self.cells[spec.name] = {"entry": entry, "ctx": top, "cell": fits[top]}
        self.ctx = ctx or min(c["ctx"] for c in self.cells.values())
        self.model = self.rows[self.sg.SPEC.name]["name"]

    def _spec(self, config: str):
        return self.vd.SPEC if config == "vllm" else self.sg.SPEC

    def start(self, config: str) -> Engine:
        spec = self._spec(config)
        info = self.cells[spec.name]
        self.hf.container_remove(self.runtime, spec.container_name)
        deadline = time.time() + CARD_SETTLE_S
        why = card_check()
        while why and time.time() < deadline:
            time.sleep(5)
            why = card_check()
        if why:
            raise CheckRefused(f"before launching {config}: {why}")
        models_dir = os.environ.get("VLLM_MODELS_DIR", "/var/cache/devai/vllm") if config == "vllm" \
            else os.environ.get("SGLANG_MODELS_DIR", "/var/cache/devai/sglang")
        row, entry = self.rows[spec.name], info["entry"]
        kv = self.pl._cell_kv_cache_dtype(info["cell"], spec.name)
        frac = self.hf.host_scaled_fraction(self.hf.model_size_gb_from_row(row), self.host_vram_gb,
                                            self.host_vram_gb, spec.reserve_gb)
        rp, tp = entry.get("reasoning_parser"), entry.get("tool_parser")
        plugin_volume, tool_plugin, reasoning_plugin = self.hf._resolve_plugins(spec, rp, tp)
        kwargs = dict(reasoning_parser=rp, tool_parser=tp, reasoning_parser_plugin=reasoning_plugin,
                      tool_parser_plugin=tool_plugin, speculative_config=None, kv_cache_dtype=kv)
        if config == "sglang-noradix":
            kwargs["disable_radix_cache"] = True
        args = list(spec.build_args(self.model, self.ctx, frac, **kwargs))
        flags, env_extra = self.hf.recovery_overrides(self.model, spec.name)
        args += flags
        env = {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", **env_extra}
        image = self.hf.recovery_image(self.model, spec.name) or spec.image
        launch = {"backend": spec.name, "image": image,
                  "image_digest": self.core.image_digest_via_cli(self.runtime, image),
                  "context": self.ctx, "kv_cache_dtype": kv or "(engine default)",
                  "mem_fraction": round(frac, 4), "argv": args, "env": env}
        try:
            self.hf.container_run_detached(
                self.runtime, spec.container_name, image, spec.probe_port, models_dir, env,
                spec.entrypoint, args, extra_volumes=[plugin_volume] if plugin_volume else [],
                backend=spec.name)
        except RuntimeError as e:
            raise ArmFailure(config, f"launch failed: {e}")
        base = f"http://127.0.0.1:{spec.probe_port}"
        t0 = time.time()
        while time.time() - t0 < self.hf.STARTUP_TIMEOUT:
            try:
                self.hf.http_get(f"{base}/health", timeout=5.0)
                break
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            except json.JSONDecodeError:
                break
            if self.hf.container_state(self.runtime, spec.container_name) in ("exited", "stopped", "absent"):
                raise ArmFailure(config, "the engine exited before /health: "
                                 + self.hf.container_logs(self.runtime, spec.container_name)[-1500:])
            time.sleep(self.hf.HEALTH_POLL_INTERVAL)
        else:
            raise ArmFailure(config, f"no /health within {self.hf.STARTUP_TIMEOUT:.0f} s")
        launch["startup_s"] = round(time.time() - t0, 1)
        return Engine(config, base, launch)

    def stop(self, engine: Engine) -> None:
        spec = self._spec(engine.config)
        state = self.hf.container_state(self.runtime, spec.container_name)
        if state in ("exited", "stopped"):
            engine.launch["exited_during_run"] = True
        engine.launch["log_tail"] = self.hf.container_logs(self.runtime, spec.container_name)[-3000:]
        self.hf.container_remove(self.runtime, spec.container_name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="the JSON report")
    ap.add_argument("--salt", default=None, help="the run's nonce (default: random)")
    ap.add_argument("--ctx", type=int, default=None,
                    help="override the context (default: the smaller of the two probed maxima)")
    ap.add_argument("--runtime", default=os.environ.get("CONTAINER_RUNTIME", "podman"))
    ap.add_argument("--host-vram-gb", type=int, default=int(os.environ.get("GPU_MEMORY_GB", "24")))
    a = ap.parse_args()

    why = card_check()
    if why:
        sys.exit(f"sglang-keep-check: the card is not idle ({why}); run `make cache-down slurm-down` first")
    engines = RealEngines(a.runtime, a.host_vram_gb, a.ctx)
    engines.hf.assert_no_active_backends(a.runtime)
    engines.hf.install_probe_cleanup()
    prompts = [json.loads(line) for line in
               (REPO_ROOT / "scripts" / "bench" / "data" / "latency_prompts.jsonl").read_text().splitlines()
               if line.strip()]
    salt = a.salt or BC.new_salt()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = measure(model=engines.model, salt=salt, prompts=prompts,
                     start=engines.start, stop=engines.stop)
    report = {"check": "platform-rebuild Phase 15 / I16", "model": MODEL_REPO,
              "context": engines.ctx, "thresholds": THRESHOLDS, "body_fields": BODY_FIELDS,
              "sessions": SESSIONS, "turns": TURNS, "started": started,
              "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **report}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"sglang-keep-check: {report.get('outcome')} -> {a.out}", file=sys.stderr)
    return 0 if report.get("outcome") in ("keep", "freeze") else 1


if __name__ == "__main__":
    sys.exit(main())
