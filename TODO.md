# TODO

## Completed
- [x] Install Aider as local AI coding agent in JupyterLab
- [x] Configure Claude Code and Codex CLIs to use local models via router
- [x] Add JupyterLab launcher icons for Aider
- [x] Add interactive agent picker for shell-cpu/shell-gpu targets
- [x] Seed Codex config files via entrypoint
- [x] Router refactor: multi-port architecture (Ollama :11434, vLLM :11435, SGLang :11436)
- [x] GPU exclusion with graceful drain
- [x] Container liveness verification (detect externally stopped containers)
- [x] SGLang backend support added
- [x] models.yaml flat list with backend: [list] per model
- [x] Go unit tests (19) + integration tests (19) — all passing
- [x] Interactive model picker (fzf-based shell + Jupyter launcher cards)
- [x] Dynamic GPU memory fraction (model size vs VRAM, backend-aware)
- [x] Dynamic context length (128K default, auto-reduced for tight fits)
- [x] Go unit tests expanded to 31 (parseSizeGB, memFraction, computeLaunchConfig)

## Open Items
- [ ] README.md — full rewrite of remaining legacy sections (appendices, detailed GPU docs)
- [ ] SGLang validation — test NVFP4 models on SGLang, benchmark vs vLLM
- [ ] Makefile model scripts — update ollama-list.py, vllm-list.py for new models.yaml format
- [ ] devai.sh / ollama.sh — rewrite standalone scripts for production use
