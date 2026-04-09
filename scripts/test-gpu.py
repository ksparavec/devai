#!/usr/bin/env python3
"""GPU detection and matrix multiply benchmark."""
import time
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("No CUDA device found")
    raise SystemExit(1)

print(f"CUDA version:    {torch.version.cuda}")
print(f"Device count:    {torch.cuda.device_count()}")
print(f"Device name:     {torch.cuda.get_device_name(0)}")
mem_bytes = torch.cuda.mem_get_info(0)[1]
print(f"Device memory:   {mem_bytes / 1024**3:.1f} GB")

sizes = [1000, 2000, 4000, 8000]
runs = 10

print(f"\nMatrix multiply benchmark ({runs} runs each)")
print(f"{'Size':>6}  {'CPU':>10}  {'GPU':>10}  {'Speedup':>8}")
print("-" * 40)

for n in sizes:
    # CPU
    a_cpu = torch.randn(n, n)
    b_cpu = torch.randn(n, n)
    # warmup
    a_cpu @ b_cpu
    t0 = time.perf_counter()
    for _ in range(runs):
        a_cpu @ b_cpu
    cpu_ms = (time.perf_counter() - t0) / runs * 1000

    # GPU
    a_gpu = torch.randn(n, n, device="cuda")
    b_gpu = torch.randn(n, n, device="cuda")
    # warmup
    a_gpu @ b_gpu
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) / runs * 1000

    print(f"{n:>6}  {cpu_ms:>8.1f}ms  {gpu_ms:>8.1f}ms  {cpu_ms / gpu_ms:>7.1f}x")
