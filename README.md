# Gemini Lab

A containerized environment designed for AI experimentation and development, featuring **JupyterLab** and the **Google Gemini CLI**. This setup provides a consistent, isolated workspace with essential tools pre-installed.

## Features

*   **Base Environment**: Python 3.11 (slim) on Debian.
*   **Interactive Computing**: **JupyterLab** pre-installed and configured.
*   **AI Tools**: Official **Google Gemini CLI** (`@google/gemini-cli`) available globally.
*   **Package Management**: Includes `uv` (fast Python installer) and `npm`.
*   **Development Tools**: `git`, `build-essential`, `rustc`, `cargo`.
*   **Installed Packages**:
    *   `python` (3.11)
    *   `jupyterlab`
    *   `@google/gemini-cli`
    *   `uv`
    *   `nodejs` (20.x)
    *   `rustc` / `cargo`
*   **Runtime**: Optimized for **Podman** (supports rootless mode) but fully compatible with Docker.

## Prerequisites

*   **Podman** (recommended) or Docker
*   **Make** (GNU Make)

## Quick Start

### 1. Configuration

#### Podman Storage Driver (Recommended)
Before building, ensure you are using the `overlay` storage driver for better performance and disk usage. The default `vfs` driver can be slow and space-consuming.

1.  **Check current driver**:
    ```bash
    podman info --format '{{.Store.GraphDriverName}}'
    ```

2.  **Update Configuration** (if the output is not `overlay`):
    Create or edit `~/.config/containers/storage.conf`:
    ```ini
    [storage]
    driver = "overlay"
    ```

3.  **Reset Storage** (Warning: This deletes all existing images/containers):
    ```bash
    podman system reset
    ```

#### Environment Setup
Copy the example configuration file to create your local environment settings:

```bash
cp .env.example .env
```

Open `.env` and you **must** adjust the following settings:

*   `HOST_HOME_DIR`: **Required.** Set this to your local user's home directory (e.g., `/home/username` or `/Users/username`).
    *   *Why?* Mounting your home directory allows the container to access your existing configuration files (like `.gitconfig`, `.ssh/`, or shell aliases), ensuring the container environment feels familiar and fully functional.
*   `CONTAINER_RUNTIME`: Defaults to `podman`. Change to `docker` if preferred.
*   `PORT`: Local port to access JupyterLab (default: `8888`).
*   **Proxy Settings**: Configure `HTTP_PROXY` and `HTTPS_PROXY` if you are behind a corporate firewall.

#### Optional: Add Python Packages
To install additional Python modules into the image:

1.  Copy the example requirements file:
    ```bash
    cp requirements.txt.example requirements.txt
    ```
2.  Edit `requirements.txt` and add your desired packages (one per line).
3.  Build the image (the build process will automatically detect and install these packages):
    ```bash
    make build
    ```

### 2. Build the Image

Use the `make` command to build the container image:

```bash
make build
```

This command passes proxy settings from your `.env` file (or environment variables) to the build process.

### 3. Run the Environment

Start the container with:

```bash
make run
```

*   The current directory (`.`) is mounted to `/home/devai/work` inside the container. Any files created in the `work/` folder inside JupyterLab will persist on your host machine.
*   The container handles user permissions automatically, mapping the internal user (`devai`) to your host user ID to avoid permission issues with mounted files.

### 4. Access JupyterLab

After running `make run`, the console will display access URLs. You will typically see links for both your **Host IP** and **localhost**:

```text
http://192.168.1.10:8888/lab?token=<long-token-string>
http://127.0.0.1:8888/lab?token=<long-token-string>
```

Copy and paste either URL into your browser to access the JupyterLab interface.

## Using the Gemini CLI

The Google Gemini CLI is installed globally. You can access it from a terminal within JupyterLab:

1.  Open a Terminal in JupyterLab (File -> New -> Terminal).
2.  **First Run & Authentication**:
    When you run a command for the first time (e.g., `gemini prompt "Hello"`), the CLI will prompt you to authenticate.
    *   It will likely provide a URL to visit in your browser to authorize the application.
    *   Follow the on-screen instructions to log in with your Google account.
    *   Once authenticated, a token will be saved locally (in the mounted home directory), so you won't need to log in again.

For more detailed usage instructions and API capabilities, refer to the [official Google Gemini documentation](https://ai.google.dev/).

### Cleaning Up
To remove the built image:

```bash
make clean
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
