"""VRAM-usage sampler for the bench harness.

Spawns a daemon thread that polls ``nvidia-smi --query-gpu=memory.used``
on a fixed cadence while a model's bench tasks run, then reports
peak and mean VRAM in GB. The polling is shellout-based to match
``scripts/_probe_hf_common.py`` style — the existing
``gpu_memory_used_mb()`` helper there does exactly the same thing for
single-shot probes.
"""

from __future__ import annotations

import subprocess
import threading
import time

DEFAULT_INTERVAL_S = 1.0


def gpu_memory_used_mb() -> int:
    """Max ``memory.used`` across visible GPUs in MB; 0 when nvidia-smi
    is unavailable. Mirrors the helper in scripts/_probe_hf_common.py
    so behaviour is identical to single-shot probes — but kept local
    here so this module can be imported without sys.path gymnastics.
    """
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    if r.returncode != 0:
        return 0
    values: list[int] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line))
        except ValueError:
            continue
    return max(values) if values else 0


class VramSampler:
    """Background nvidia-smi sampler.

    Use as a context manager or via explicit ``start()`` / ``stop()``.
    ``stop()`` returns ``{"peak_vram_gb", "mean_vram_gb",
    "n_samples"}`` derived from samples collected between start and
    stop. Samples taken every ``interval`` seconds.

    The thread is a daemon so a crashed runner doesn't leave it
    hanging. ``stop()`` is idempotent.
    """

    def __init__(self, interval: float = DEFAULT_INTERVAL_S) -> None:
        self._interval = interval
        self._samples_mb: list[int] = []
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._samples_mb.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # Wait reads return True on stop; use that to break cleanly
        # and avoid one extra sleep at shutdown.
        while not self._stop_evt.wait(self._interval):
            mb = gpu_memory_used_mb()
            if mb > 0:
                self._samples_mb.append(mb)

    def stop(self) -> dict:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 3)
            self._thread = None
        if not self._samples_mb:
            return {"peak_vram_gb": 0.0, "mean_vram_gb": 0.0, "n_samples": 0}
        peak = max(self._samples_mb) / 1024
        mean = sum(self._samples_mb) / len(self._samples_mb) / 1024
        return {
            "peak_vram_gb": round(peak, 2),
            "mean_vram_gb": round(mean, 2),
            "n_samples": len(self._samples_mb),
        }

    def __enter__(self) -> "VramSampler":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        # Discard the result here — callers using `with` should
        # call stop() themselves to capture metrics. Provided so the
        # context manager always tears down cleanly even on exception.
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join(timeout=self._interval * 3)
            self._thread = None


if __name__ == "__main__":
    # Manual smoke: sample for 5 seconds and print result.
    import json
    import sys

    s = VramSampler(interval=0.5)
    s.start()
    time.sleep(5)
    print(json.dumps(s.stop(), indent=2), file=sys.stderr)
