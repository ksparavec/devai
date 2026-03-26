#!/usr/bin/env python3
"""Quick Ollama benchmark — send a prompt and print timing stats."""

import argparse
import time
import ollama

parser = argparse.ArgumentParser(description="Benchmark an Ollama model")
parser.add_argument("prompt", nargs="?", default="hi")
parser.add_argument("-m", "--model", default=None, help="Model name (default: $OLLAMA_DEFAULT_MODEL or llama3.2)")
parser.add_argument("--think", action="store_true", help="Enable thinking mode")
args = parser.parse_args()

import os
model = args.model or os.environ.get("OLLAMA_DEFAULT_MODEL", "llama3.2")

# Unload any currently loaded model to free VRAM
try:
    for m in ollama.ps().get("models", []):
        ollama.generate(model=m["name"], keep_alive=0)
except Exception:
    pass

start = time.time()
response = ollama.chat(
    model=model,
    messages=[{"role": "user", "content": args.prompt}],
    think=args.think,
)
elapsed = time.time() - start

content = response["message"]["content"]
eval_count = response.get("eval_count", 0)
prompt_eval_count = response.get("prompt_eval_count", 0)
load_ns = response.get("load_duration", 0)
prompt_ns = response.get("prompt_eval_duration", 0)
eval_ns = response.get("eval_duration", 0)

print(content)
print()
print(f"model:            {model}")
print(f"total:            {elapsed:.2f}s")
print(f"load:             {load_ns / 1e9:.3f}s")
print(f"prompt eval:      {prompt_eval_count} tokens ({prompt_eval_count / max(prompt_ns / 1e9, 0.001):.1f} tok/s)")
print(f"generation:       {eval_count} tokens ({eval_count / max(eval_ns / 1e9, 0.001):.1f} tok/s)")
