#!/usr/bin/env python3
"""Download vLLM models from HuggingFace."""
import os
import subprocess
import sys
import yaml

config_path = os.environ.get('INFERENCE_CONFIG', 'deploy/models.yaml')
models_dir = os.environ.get('VLLM_MODELS_DIR', '/var/cache/devai/ollama/models/vllm')
hf_cli = os.environ.get('HF_CLI', 'hf')
target = os.environ.get('MODEL', '')

cfg = yaml.safe_load(open(config_path))
models = cfg.get('models', {}).get('vllm', [])

for m in models:
    if target and m['name'] != target:
        continue
    path = os.path.join(models_dir, m['name'])
    print(f"Syncing {m['name']} from {m['repo']}...")
    os.makedirs(path, exist_ok=True)
    rc = subprocess.call([hf_cli, 'download', m['repo'], '--local-dir', path])
    if rc != 0:
        sys.exit(rc)

if target and not any(m['name'] == target for m in models):
    print(f"Error: '{target}' not in catalog. See 'make vllm-list'.")
    sys.exit(1)
