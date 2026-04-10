#!/bin/bash
exec aider --model "ollama_chat/${OLLAMA_DEFAULT_MODEL:-qwen3.5:9b}"
