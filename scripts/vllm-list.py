#!/usr/bin/env python3
"""List vLLM models with status."""
import os
import yaml

config_path = os.environ.get('INFERENCE_CONFIG', 'deploy/models.yaml')
models_dir = os.environ.get('VLLM_MODELS_DIR', '/var/cache/devai/ollama/models/vllm')

cfg = yaml.safe_load(open(config_path))
models = cfg.get('models', {}).get('vllm', [])

print()
print(f"{'MODEL':<45s} {'SIZE':<10s} {'STATE':<8s} PURPOSE")
print('-' * 111)
for m in models:
    state = '-'
    if os.path.exists(os.path.join(models_dir, m['name'], 'config.json')):
        state = 'ready'
    print(f"{m['name']:<45s} {m['size']:<10s} {state:<8s} {m['purpose']}")
print()
