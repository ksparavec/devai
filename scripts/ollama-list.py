#!/usr/bin/env python3
"""List Ollama models with catalog info (sizes, purpose)."""
import os
import subprocess
import sys
import yaml

config_path = os.environ.get('INFERENCE_CONFIG', 'deploy/models.yaml')
runtime = os.environ.get('CONTAINER_RUNTIME', 'podman')
container = os.environ.get('OLLAMA_CONTAINER', 'devai-ollama')

cfg = yaml.safe_load(open(config_path))
models = cfg.get('models', {}).get('ollama', [])

try:
    out = subprocess.check_output(
        [runtime, 'exec', container, 'ollama', 'list'], text=True)
    installed = [l.split()[0] for l in out.strip().split('\n')[1:] if l.strip()]
except Exception:
    installed = []

try:
    out = subprocess.check_output(
        [runtime, 'exec', container, 'ollama', 'ps'], text=True)
    loaded = [l.split()[0] for l in out.strip().split('\n')[1:] if l.strip()]
except Exception:
    loaded = []

print()
print(f"{'MODEL':<35s} {'SIZE':<10s} {'FIT':<6s} {'STATE':<6s} PURPOSE")
print('-' * 103)
for m in models:
    state = '-'
    if any(m['name'] in i for i in installed):
        state = 'ready'
    if any(m['name'] in l for l in loaded):
        state = '\033[32mloaded\033[0m'
    print(f"{m['name']:<35s} {m['size']:<10s} {m['fit']:<6s} {state:<6s} {m['purpose']}")
print()
