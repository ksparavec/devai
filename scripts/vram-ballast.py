#!/usr/bin/env python3
"""Hold a fixed amount of GPU memory so the card is PHYSICALLY smaller.

`make probe` measures every Ollama model on several VRAM bands (16G, 24G,
...) using one physical card. To probe the 16G band on a 24G card it runs
this script for the duration, holding the 8 GiB difference, so the card
genuinely has only 16 GB's worth free.

Why physical, and not an engine setting: the probe loads models with
`num_gpu: 999` ("all layers on the GPU, or fail"), mirroring the router's
warm-load. Both settings that used to or seemed to simulate a smaller card
are dead for that load -- OLLAMA_GPU_OVERHEAD no longer governs placement
(Ollama handed it to llama.cpp's automatic fit), and llama.cpp's own
--fit-target is ignored once the layer count is pinned, since fit only
adjusts UNSET parameters. With the memory physically taken, a forced-full
load that does not fit fails with a real `cudaMalloc failed: out of memory`,
exactly as on the smaller card, whatever the engine does internally.

Needs only the NVIDIA driver's own library (libcuda.so.1, present on every
GPU host) -- no CUDA toolkit, no PyTorch, no container.

Protocol: prints `ready <MiB>` on stdout once the memory is held, then
sleeps until SIGTERM / SIGINT. ANY failure exits non-zero WITHOUT printing
`ready`: a ballast that silently held nothing would bring back the very bug
this exists to fix (a "16G" band measured on the full card).

Usage: vram-ballast.py <MiB> [device-index]
Env:   VRAM_BALLAST_LIBCUDA   driver library to load (default libcuda.so.1)
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys
import time

_MIB = 1024 * 1024


def die(msg: str) -> "NoReturn":  # noqa: F821
    print(f"error: vram-ballast: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_mib(raw: str) -> int:
    try:
        mib = int(raw)
    except ValueError:
        die(f"size must be a whole number of MiB, got {raw!r}")
    if mib <= 0:
        die(f"size must be positive, got {mib}")
    return mib


def hold(mib: int, device_index: int) -> None:
    libname = os.environ.get("VRAM_BALLAST_LIBCUDA", "libcuda.so.1")
    try:
        cuda = ctypes.CDLL(libname)
    except OSError as e:
        die(f"cannot load the NVIDIA driver library {libname}: {e}")

    def ck(rc: int, what: str) -> None:
        if rc != 0:
            die(f"{what} failed with CUDA error {rc}")

    ck(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cuda.cuDeviceGet(ctypes.byref(dev), device_index),
       f"cuDeviceGet({device_index})")
    ctx = ctypes.c_void_p()
    ck(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev),
       "cuDevicePrimaryCtxRetain")
    ck(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    ptr = ctypes.c_uint64()
    ck(cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(mib * _MIB)),
       f"cuMemAlloc of {mib} MiB (is the GPU already in use?)")

    print(f"ready {mib}", flush=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: sys.exit(0))
    while True:
        time.sleep(3600)


def main(argv: list[str]) -> None:
    if len(argv) not in (2, 3):
        die("usage: vram-ballast.py <MiB> [device-index]")
    mib = parse_mib(argv[1])
    device_index = int(argv[2]) if len(argv) == 3 else 0
    hold(mib, device_index)


if __name__ == "__main__":
    main(sys.argv)
