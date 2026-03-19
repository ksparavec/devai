# Layer 2: All packages + configuration (rebuild when packages change)
# CPU:  podman build -t devai-lab .
# GPU:  podman build --build-arg BASE_IMAGE=devai-base-gpu --build-arg GPU_BUILD=true -t devai-lab-gpu .

ARG BASE_IMAGE=devai-base
FROM ${BASE_IMAGE}

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG GPU_BUILD=false

ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV OLLAMA_HOST=http://host.containers.internal:11434
ENV UV_BREAK_SYSTEM_PACKAGES=1

# Install PyTorch (CPU-only for CPU builds, full CUDA for GPU builds)
# Must come before other ML packages to prevent them pulling in NVIDIA deps
RUN if [ "$GPU_BUILD" = "true" ]; then \
        uv pip install --system --no-cache-dir torch torchvision torchaudio; \
    else \
        uv pip install --system --no-cache-dir torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install Python packages (ML, data science, AI providers, Jupyter)
COPY .default-python-packages /tmp/.default-python-packages
RUN uv pip install --system --no-cache-dir \
       -r /tmp/.default-python-packages

# Install npm packages (AI CLIs)
COPY .default-npm-packages /tmp/.default-npm-packages
RUN xargs npm install -g < /tmp/.default-npm-packages \
    && npm cache clean --force

# Install optional project-specific dependencies
COPY requirements.txt* /tmp/
RUN if [ -f /tmp/requirements.txt ]; then \
        uv pip install --system --no-cache-dir -r /tmp/requirements.txt; \
    fi && rm -f /tmp/requirements.txt

# Create user and directories
RUN groupadd -g 1000 devai \
    && useradd -u 1000 -g 1000 -m -s /bin/bash devai \
    && mkdir -p /home/devai/work \
    && chown -R devai:devai /home/devai

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8888

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
