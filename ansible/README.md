# Build Environment Setup for DevAI

Automated installation of container runtimes and cloud management CLIs for the DevAI build environment.

## Quick Start

From the repository root:

```bash
make setup-build-env       # Install runtimes + cloud tools (recommended)
make setup-cloud-tools     # Install cloud tools only (no runtimes)
```

Or run the bootstrap script directly:

```bash
cd ansible
./bootstrap.sh                         # Install everything
./bootstrap.sh --install-runtime none  # Cloud tools only
```

After installation, add the bin directory to your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Add this line to your shell profile (`~/.bashrc`, `~/.zshrc`) for persistence.

## What Gets Installed

### Container Runtimes

| Tool | Description | Default |
|------|-------------|---------|
| Podman | Rootless container runtime | Installed |
| Docker | Container runtime | Installed |

By default, both container runtimes are installed. Use `--install-runtime` to select specific runtimes.

### Cloud Management Tools

| Tool | Description | Installed To |
|------|-------------|--------------|
| AWS CLI v2 | Amazon Web Services CLI | `~/.local/bin/aws` |
| Azure CLI | Microsoft Azure CLI | `~/.local/bin/az` |
| Google Cloud CLI | GCP gcloud, gsutil, bq | `~/.local/bin/gcloud` |
| Terraform | Infrastructure as Code | `~/.local/bin/terraform` |
| kubectl | Kubernetes CLI | `~/.local/bin/kubectl` |
| kustomize | Kubernetes customization | `~/.local/bin/kustomize` |

## How It Works

The bootstrap script performs these steps:

1. **Install system prerequisites** (requires sudo)
   - curl, tar, unzip, make, git, ca-certificates
   - Uses apt-get, dnf, pacman, zypper, apk, or brew based on your distro

2. **Install uv** (Python package manager)
   - Downloaded to `~/.local/bin`

3. **Create Python virtual environment**
   - Python 3.12 by default
   - Located at `~/.local/devai-venv`

4. **Install Ansible** into the venv
   - Plus required Galaxy collections

5. **Run Ansible playbook** to install all tools
   - **Container runtimes** (Podman, Docker) - requires sudo via `become: true`
   - **Cloud tools** (AWS CLI, Azure CLI, etc.) - user-space, no root required
   - Respects `.env` file or `--install-runtime` flag for runtime selection

## Supported Platforms

| OS | Package Manager | Prerequisites |
|----|-----------------|---------------|
| Debian / Ubuntu | apt-get | sudo access |
| RHEL / Fedora / CentOS | dnf / yum | sudo access |
| Arch / Manjaro | pacman | sudo access |
| openSUSE / SLES | zypper | sudo access |
| Alpine | apk | sudo access |
| macOS | Homebrew | Homebrew pre-installed |

## Post-Installation: Configure Cloud Providers

```bash
# AWS
aws configure
# or for SSO
aws sso login

# Azure
az login

# Google Cloud
gcloud init
```

## Verify Installation

```bash
make check-cloud-tools
```

---

## Appendix A: Selective Installation

Install only specific components:

```bash
# Via Makefile - runtimes only
make setup-runtime                # Both runtimes (no cloud tools)
make setup-runtime-podman         # Podman only
make setup-runtime-docker         # Docker only

# Via Makefile - specific cloud tools only (no runtimes)
make setup-cloud-tools-aws        # AWS CLI only
make setup-cloud-tools-azure      # Azure CLI only
make setup-cloud-tools-gcp        # Google Cloud CLI only
make setup-cloud-tools-terraform  # Terraform only
make setup-cloud-tools-k8s        # kubectl + kustomize

# Via bootstrap script
./ansible/bootstrap.sh -- --tags aws,terraform
./ansible/bootstrap.sh -- --tags kubernetes
```

Skip specific tools:

```bash
./ansible/bootstrap.sh -- -e "install_gcloud=false"
./ansible/bootstrap.sh -- -e "install_azure_cli=false" -e "install_aws_cli=false"
```

## Appendix B: Custom Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVAI_PYTHON_VERSION` | `3.12` | Python version for venv |
| `DEVAI_VENV_DIR` | `~/.local/devai-venv` | Virtual environment path |
| `DEVAI_BIN_DIR` | `~/.local/bin` | Tool binary directory |
| `CONTAINER_RUNTIME` | `both` | Container runtime to install (in `.env`) |

### Bootstrap Options

```bash
# Custom Python version
./ansible/bootstrap.sh --python-version 3.11

# Custom directories
./ansible/bootstrap.sh --venv-dir ~/.venvs/devai --bin-dir ~/bin

# Container runtime selection
./ansible/bootstrap.sh --install-runtime podman    # Podman only
./ansible/bootstrap.sh --install-runtime docker    # Docker only
./ansible/bootstrap.sh --install-runtime both      # Both (default)
./ansible/bootstrap.sh --install-runtime none      # Skip runtime installation

# Dry run (check mode)
./ansible/bootstrap.sh -- --check

# Verbose output
./ansible/bootstrap.sh -- -v
```

Note: If `.env` exists with `CONTAINER_RUNTIME` set, that value is used as default. The `--install-runtime` flag overrides the `.env` setting.

### Custom Tool Versions

```bash
./ansible/bootstrap.sh -- -e "terraform_version=1.6.0"
./ansible/bootstrap.sh -- -e "kubectl_version=v1.28.0"
```

## Appendix C: Ansible Role Structure

```
ansible/
├── bootstrap.sh              # Entry point (installs prerequisites + runs Ansible)
├── ansible.cfg               # Ansible configuration (become=false)
├── requirements.yml          # Galaxy collection requirements
├── inventory/
│   └── localhost.yml
├── group_vars/
│   └── all.yml               # All configurable paths and versions
├── playbooks/
│   └── install-cloud-tools.yml
└── roles/
    ├── podman/               # Container runtime (requires sudo)
    ├── docker/               # Container runtime (requires sudo)
    ├── aws_cli/
    ├── azure_cli/
    ├── gcloud/
    ├── terraform/
    ├── kubectl/
    └── kustomize/
```

### Available Roles and Tags

| Role | Tags | Description |
|------|------|-------------|
| `podman` | `podman`, `runtime`, `runtimes` | Podman container runtime (requires sudo) |
| `docker` | `docker`, `runtime`, `runtimes` | Docker container runtime (requires sudo) |
| `aws_cli` | `aws`, `aws-cli`, `cloud-tools` | AWS CLI v2 |
| `azure_cli` | `azure`, `azure-cli`, `cloud-tools` | Azure CLI (via pip) |
| `gcloud` | `gcp`, `gcloud`, `cloud-tools` | Google Cloud CLI |
| `terraform` | `terraform`, `cloud-tools` | HashiCorp Terraform |
| `kubectl` | `kubectl`, `kubernetes`, `cloud-tools` | Kubernetes kubectl |
| `kustomize` | `kustomize`, `kubernetes`, `cloud-tools` | Kubernetes kustomize |

## Appendix D: Troubleshooting

### sudo password prompt

The script requires sudo for:
- Installing system prerequisites (curl, tar, unzip, make, git, ca-certificates)
- Installing container runtimes (Podman, Docker) via Ansible roles

If you're in a non-interactive environment, ensure sudo is configured for passwordless access or pre-install the prerequisites and runtimes manually:

```bash
# Debian/Ubuntu - prerequisites
sudo apt-get install -y curl tar unzip make git ca-certificates

# Debian/Ubuntu - runtimes (optional)
sudo apt-get install -y podman docker.io

# RHEL/Fedora - prerequisites
sudo dnf install -y curl tar unzip make git ca-certificates

# RHEL/Fedora - runtimes (optional)
sudo dnf install -y podman moby-engine
```

To skip runtime installation entirely:
```bash
./ansible/bootstrap.sh --install-runtime none
```

### Tool not found after installation

Ensure `~/.local/bin` is in your PATH:

```bash
echo $PATH | tr ':' '\n' | grep "$HOME/.local/bin" || echo "Not in PATH"
```

### uv installation fails

Manually install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Ansible Galaxy errors

Manually install collections:

```bash
~/.local/devai-venv/bin/ansible-galaxy collection install -r ansible/requirements.yml
```

### Re-run installation

The script is idempotent. Run it again to update or fix installations:

```bash
make setup-build-env
```
