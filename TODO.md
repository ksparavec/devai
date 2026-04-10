# TODO

## Local AI Agent Integration
- [x] Install Aider as local AI coding agent in JupyterLab
- [x] Configure Claude Code and Codex CLIs to use local models via router
- [x] Add JupyterLab launcher icons for Aider
- [x] Add interactive agent picker for shell-cpu/shell-gpu targets
- [x] Seed Codex config files via entrypoint

## Router Testing
- [x] Go unit tests for isVLLMModel, model routing, API translation (gpu-arbiter/main_test.go)
- [x] Fix API translation parameter forwarding (temperature, top_p, max_tokens, etc.)
- [x] Expand integration tests: empty model, concurrent requests, health staleness, parameter forwarding

## Open Items
- [ ] README.md — full rewrite of remaining legacy sections (appendices, detailed GPU docs)
- [ ] JupyterLab extension rebuild (run `make build-gpu` to rebuild with new launcher icons)
- [ ] devai.sh / ollama.sh — rewrite standalone scripts for production use
