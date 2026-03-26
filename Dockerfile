# Layer 2: All packages + configuration (rebuild when packages change)
# CPU:  podman build -t devai-lab .
# GPU:  podman build --build-arg BASE_IMAGE=devai-base-gpu --build-arg GPU_BUILD=true -t devai-lab-gpu .

ARG BASE_IMAGE=devai-base
FROM ${BASE_IMAGE}

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG GPU_BUILD=false

ENV OLLAMA_HOST=http://devai-ollama:11434
ENV OLLAMA_URL=http://devai-ollama:11434
ENV OLLAMA_DEFAULT_MODEL=llama3.2
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV PATH="/home/devai/.local/bin:${PATH}"

# --- Binary installs (from local cache, populated by: make fetch) ---

# Install CLI binaries (pre-downloaded to /var/cache/bin/ via cache mount)
RUN cp /var/cache/bin/uv /var/cache/bin/uvx /var/cache/bin/claude /var/cache/bin/codex /var/cache/bin/ollama /usr/local/bin/ \
    && chmod +x /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/claude /usr/local/bin/codex /usr/local/bin/ollama

# Install code-server (pre-downloaded)
RUN cp -a /var/cache/bin/code-server /usr/local/lib/code-server \
    && ln -s /usr/local/lib/code-server/bin/code-server /usr/local/bin/code-server

# Install Gemini CLI (pre-installed by: make fetch)
RUN cp -a /var/cache/bin/gemini/lib/node_modules/@google /usr/local/lib/node_modules/ \
    && ln -sf ../lib/node_modules/@google/gemini-cli/bin/gemini.js /usr/local/bin/gemini

# --- Package installs (change more often, cached after binaries) ---

# Install PyTorch (CPU-only for CPU builds, full CUDA for GPU builds)
# Host cache dirs are bind-mounted via -v at build time (see Makefile)
RUN if [ "$GPU_BUILD" = "true" ]; then \
        uv pip install --system torch torchvision torchaudio; \
    else \
        uv pip install --system torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install Python packages (ML, data science, AI providers, Jupyter)
COPY .default-python-packages /tmp/.default-python-packages
RUN uv pip install --system -r /tmp/.default-python-packages

# Install optional project-specific dependencies
COPY requirements.txt* /tmp/
RUN if [ -f /tmp/requirements.txt ]; then \
        uv pip install --system -r /tmp/requirements.txt; \
    fi && rm -f /tmp/requirements.txt

# --- JupyterLab extension (depends on jupyterlab being installed above) ---

# Install JupyterLab AI launcher extension (pre-built)
COPY packages/jupyter-ai-launchers/jupyter_ai_launchers/labextension /usr/local/share/jupyter/labextensions/jupyter-ai-launchers

# --- Runtime setup ---

# Install ollama-chat wrapper for JupyterLab launcher
COPY scripts/ollama-chat.sh /usr/local/bin/ollama-chat
RUN chmod +x /usr/local/bin/ollama-chat

# Create user and directories (handle GID/UID conflicts in GPU base images)
RUN getent group 1000 >/dev/null || groupadd -g 1000 devai \
    && id -u 1000 >/dev/null 2>&1 || useradd -u 1000 -g 1000 -m -s /bin/bash devai \
    && mkdir -p /home/devai/work /home/devai/.local/bin \
    && ln -s /usr/local/bin/claude /home/devai/.local/bin/claude \
    && ln -s /usr/local/bin/codex /home/devai/.local/bin/codex \
    && ln -s /usr/local/bin/gemini /home/devai/.local/bin/gemini \
    && ln -s /usr/local/bin/ollama /home/devai/.local/bin/ollama \
    && ln -s /usr/local/bin/code-server /home/devai/.local/bin/code-server \
    && chown -R 1000:1000 /home/devai

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8888

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
