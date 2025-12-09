# Unified Dockerfile for CPU and GPU builds
# CPU: docker build -t devai .
# GPU: docker build -t devai-gpu --build-arg BASE_IMAGE=docker.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 --build-arg GPU_BUILD=true .

ARG BASE_IMAGE=debian:trixie-slim
FROM ${BASE_IMAGE}

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG GPU_BUILD=false

ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV SHELL=/bin/bash
ENV OLLAMA_HOST=http://host.containers.internal:11434
ENV DEBIAN_FRONTEND=noninteractive

# mise configuration - system-wide installation
ENV MISE_DATA_DIR=/opt/mise
ENV MISE_CACHE_DIR=/opt/mise/cache
ENV MISE_INSTALL_PATH=/usr/local/bin/mise
ENV PATH="/opt/mise/shims:$PATH"

# Configure apt proxy if arguments are provided
RUN if [ -n "$HTTP_PROXY" ]; then echo "Acquire::http::Proxy \"$HTTP_PROXY\";" > /etc/apt/apt.conf.d/99proxy; fi && \
    if [ -n "$HTTPS_PROXY" ]; then echo "Acquire::https::Proxy \"$HTTPS_PROXY\";" >> /etc/apt/apt.conf.d/99proxy; fi

# Install only essential system packages and compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    gosu \
    git \
    build-essential \
    rustc \
    cargo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/apt/apt.conf.d/99proxy

# Install mise from mise.run
RUN curl https://mise.run | MISE_INSTALL_PATH=/usr/local/bin/mise sh \
    && mkdir -p /opt/mise/cache \
    && chmod -R 777 /opt/mise

# Copy mise configuration
COPY mise.toml /opt/mise/config.toml
COPY .default-npm-packages /root/.default-npm-packages
ENV MISE_CONFIG_FILE=/opt/mise/config.toml

# Copy python packages (GPU build appends gpu-extra)
COPY .default-python-packages /tmp/.default-python-packages
COPY .default-python-packages.gpu-extra* /tmp/
ARG GPU_BUILD
RUN if [ "$GPU_BUILD" = "true" ] && [ -f /tmp/.default-python-packages.gpu-extra ]; then \
        cat /tmp/.default-python-packages /tmp/.default-python-packages.gpu-extra > /root/.default-python-packages; \
    else \
        cp /tmp/.default-python-packages /root/.default-python-packages; \
    fi

# Install tools via mise (uv, node, python, and default packages)
RUN mise trust --all \
    && mise install \
    && npm cache clean --force \
    && rm -rf /root/.cache

# Install optional python dependencies
COPY requirements.txt* /tmp/
RUN if [ -f /tmp/requirements.txt ]; then uv pip install --system --no-cache-dir -r /tmp/requirements.txt; fi && \
    rm -f /tmp/requirements.txt

# Create user and directories
RUN groupadd -g 1000 devai \
    && useradd -u 1000 -g 1000 -m -s /bin/bash devai \
    && mkdir -p /home/devai/work \
    && chown -R devai:devai /home/devai

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose Jupyter port
EXPOSE 8888

# Entrypoint handles user creation and switching
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
