#!/usr/bin/env python3
"""Pull Ollama model(s) from catalog."""
import os
import subprocess
import sys
import yaml

config_path = os.environ.get('INFERENCE_CONFIG', 'deploy/models.yaml')
runtime = os.environ.get('CONTAINER_RUNTIME', 'podman')
container = os.environ.get('OLLAMA_CONTAINER', 'devai-ollama')
target = os.environ.get('MODEL', '')

cfg = yaml.safe_load(open(config_path))
models = cfg.get('models', {}).get('ollama', [])

for m in models:
    if target and m['name'] != target:
        continue
    print(f"Pulling {m['name']}...")
    rc = subprocess.call([runtime, 'exec', container, 'ollama', 'pull', m['name']])
    if rc != 0 and target:
        sys.exit(rc)

if target and not any(m['name'] == target for m in models):
    print(f"Error: '{target}' not in catalog. See 'make ollama-list'.")
    sys.exit(1)
