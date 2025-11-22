FROM python:3.11-slim

ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV SHELL=/bin/bash

# Configure apt proxy if arguments are provided
RUN if [ -n "$HTTP_PROXY" ]; then echo "Acquire::http::Proxy \"$HTTP_PROXY\";" > /etc/apt/apt.conf.d/99proxy; fi && \
    if [ -n "$HTTPS_PROXY" ]; then echo "Acquire::https::Proxy \"$HTTPS_PROXY\";" >> /etc/apt/apt.conf.d/99proxy; fi

# Install dependencies, uv, gosu, and dev tools
RUN apt-get update && apt-get install -y curl gnupg libterm-readline-gnu-perl gosu \
    git build-essential rustc cargo \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && uv pip install --system --no-cache-dir jupyterlab \
    && npm install -g @google/gemini-cli \
    && groupadd -g 1000 devai \
    && useradd -u 1000 -g 1000 -m -s /bin/bash devai \
    && mkdir -p /home/devai/work \
    && chown -R devai:devai /home/devai \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && npm cache clean --force \
    && rm -rf /root/.cache \
    && rm -f /etc/apt/apt.conf.d/99proxy

# Install optional python dependencies
COPY requirements.txt* /tmp/
RUN if [ -f /tmp/requirements.txt ]; then uv pip install --system --no-cache-dir -r /tmp/requirements.txt; fi && \
    rm -f /tmp/requirements.txt

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose Jupyter port
EXPOSE 8888

# Entrypoint handles user creation and switching
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
