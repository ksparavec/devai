#!/bin/bash
exec interpreter --model "ollama/${OLLAMA_DEFAULT_MODEL:-llama3.2}"
