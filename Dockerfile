# Layer 2: All packages + configuration (rebuild when packages change)
# CPU:  podman build -t devai-lab .
# GPU:  podman build --build-arg BASE_IMAGE=devai-base-gpu --build-arg GPU_BUILD=true -t devai-lab-gpu .

ARG BASE_IMAGE=devai-base
FROM ${BASE_IMAGE}

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG GPU_BUILD=false

ENV OLLAMA_HOST=http://host.containers.internal:11434
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV PATH="/home/devai/.local/bin:${PATH}"

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

# Install Claude Code (binary from official distribution)
RUN ARCH=$(dpkg --print-architecture) \
    && case "$ARCH" in amd64) CC_PLATFORM=linux-x64;; arm64) CC_PLATFORM=linux-arm64;; *) echo "Unsupported arch: $ARCH" && exit 1;; esac \
    && CC_BUCKET="https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases" \
    && CC_VERSION=$(curl -fsSL "$CC_BUCKET/latest") \
    && curl -fsSL -o /usr/local/bin/claude "$CC_BUCKET/$CC_VERSION/$CC_PLATFORM/claude" \
    && chmod +x /usr/local/bin/claude

# Install OpenAI Codex (prebuilt binary from GitHub releases)
RUN ARCH=$(dpkg --print-architecture) \
    && case "$ARCH" in amd64) CODEX_ARCH=x86_64;; arm64) CODEX_ARCH=aarch64;; *) echo "Unsupported arch: $ARCH" && exit 1;; esac \
    && curl -fsSL "https://github.com/openai/codex/releases/latest/download/codex-${CODEX_ARCH}-unknown-linux-musl.tar.gz" \
       | tar -xz -C /usr/local/bin \
    && mv /usr/local/bin/codex-${CODEX_ARCH}-unknown-linux-musl /usr/local/bin/codex

# Install npm packages (Gemini CLI)
COPY .default-npm-packages /tmp/.default-npm-packages
RUN xargs npm install -g < /tmp/.default-npm-packages

# Install JupyterLab AI launcher extension (build JS directly, skip pip isolation)
COPY packages/jupyter-ai-launchers /tmp/jupyter-ai-launchers
RUN cd /tmp/jupyter-ai-launchers \
    && jlpm install \
    && jlpm run build:prod \
    && cp -r jupyter_ai_launchers/labextension /usr/local/share/jupyter/labextensions/jupyter-ai-launchers \
    && rm -rf /tmp/jupyter-ai-launchers

# Install optional project-specific dependencies
COPY requirements.txt* /tmp/
RUN if [ -f /tmp/requirements.txt ]; then \
        uv pip install --system -r /tmp/requirements.txt; \
    fi && rm -f /tmp/requirements.txt

# Create user and directories (handle GID/UID conflicts in GPU base images)
RUN getent group 1000 >/dev/null || groupadd -g 1000 devai \
    && id -u 1000 >/dev/null 2>&1 || useradd -u 1000 -g 1000 -m -s /bin/bash devai \
    && mkdir -p /home/devai/work /home/devai/.local/bin \
    && ln -s /usr/local/bin/claude /home/devai/.local/bin/claude \
    && ln -s /usr/local/bin/codex /home/devai/.local/bin/codex \
    && ln -s /usr/local/bin/gemini /home/devai/.local/bin/gemini \
    && chown -R 1000:1000 /home/devai

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8888

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
