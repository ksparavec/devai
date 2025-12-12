#!/usr/bin/env bash
#
# Bootstrap script for devai cloud tools installation
# Sets up Python venv with uv and installs Ansible, then runs the playbook
#
# Usage:
#   ./bootstrap.sh [options] [-- ansible-playbook options]
#
# Options:
#   --python-version VERSION   Python version to install (default: 3.12)
#   --venv-dir DIR             Directory for virtual environment (default: ~/.local/devai-venv)
#   --bin-dir DIR              Directory for tool binaries (default: ~/.local/bin)
#   --install-runtime RUNTIME  Container runtime to install: podman, docker, both, none (default: from .env or both)
#   --help                     Show this help message
#
# Examples:
#   ./bootstrap.sh
#   ./bootstrap.sh --python-version 3.11
#   ./bootstrap.sh --install-runtime podman
#   ./bootstrap.sh --install-runtime none -- --tags aws,terraform
#   ./bootstrap.sh -- -e "install_aws_cli=false"

set -euo pipefail

# Default configuration
PYTHON_VERSION="${DEVAI_PYTHON_VERSION:-3.12}"
VENV_DIR="${DEVAI_VENV_DIR:-${HOME}/.local/devai-venv}"
BIN_DIR="${DEVAI_BIN_DIR:-${HOME}/.local/bin}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Container runtime configuration
# Read from .env if exists, otherwise default to "both"
INSTALL_RUNTIME=""
if [[ -f "${REPO_DIR}/.env" ]]; then
    # Source .env to get CONTAINER_RUNTIME if set
    # shellcheck source=/dev/null
    source "${REPO_DIR}/.env" 2>/dev/null || true
    if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
        INSTALL_RUNTIME="${CONTAINER_RUNTIME}"
    fi
fi
INSTALL_RUNTIME="${INSTALL_RUNTIME:-both}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

show_help() {
    sed -n '2,/^$/p' "$0" | sed 's/^#//' | sed 's/^ //'
    exit 0
}

# Parse arguments
ANSIBLE_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --venv-dir)
            VENV_DIR="$2"
            shift 2
            ;;
        --bin-dir)
            BIN_DIR="$2"
            shift 2
            ;;
        --install-runtime)
            INSTALL_RUNTIME="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        --)
            shift
            ANSIBLE_ARGS=("$@")
            break
            ;;
        *)
            ANSIBLE_ARGS+=("$1")
            shift
            ;;
    esac
done

# Detect OS and architecture
detect_platform() {
    local os arch

    case "$(uname -s)" in
        Linux*)  os="linux" ;;
        Darwin*) os="macos" ;;
        *)       log_error "Unsupported OS: $(uname -s)"; exit 1 ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64)   arch="x86_64" ;;
        aarch64|arm64)  arch="aarch64" ;;
        *)              log_error "Unsupported architecture: $(uname -m)"; exit 1 ;;
    esac

    echo "${os}:${arch}"
}

# Detect Linux distribution
detect_distro() {
    if [[ -f /etc/os-release ]]; then
        # shellcheck source=/dev/null
        source /etc/os-release
        echo "${ID}"
    elif [[ -f /etc/debian_version ]]; then
        echo "debian"
    elif [[ -f /etc/redhat-release ]]; then
        echo "rhel"
    else
        echo "unknown"
    fi
}

# Install system prerequisites using appropriate package manager
install_prerequisites() {
    local os distro
    os="$(uname -s)"

    log_info "Checking and installing system prerequisites..."

    # Required packages
    local packages=(curl tar unzip make git ca-certificates)

    case "${os}" in
        Linux)
            distro="$(detect_distro)"
            log_info "Detected Linux distribution: ${distro}"

            case "${distro}" in
                debian|ubuntu|linuxmint|pop)
                    # Debian/Ubuntu family
                    local missing_packages=()
                    for pkg in "${packages[@]}"; do
                        if ! dpkg -s "${pkg}" &>/dev/null; then
                            missing_packages+=("${pkg}")
                        fi
                    done

                    if [[ ${#missing_packages[@]} -gt 0 ]]; then
                        log_info "Installing missing packages: ${missing_packages[*]}"
                        sudo apt-get update
                        sudo apt-get install -y "${missing_packages[@]}"
                    else
                        log_info "All prerequisites already installed"
                    fi
                    ;;

                fedora|rhel|centos|rocky|alma|ol)
                    # RHEL/Fedora family
                    local missing_packages=()
                    for pkg in "${packages[@]}"; do
                        if ! rpm -q "${pkg}" &>/dev/null; then
                            missing_packages+=("${pkg}")
                        fi
                    done

                    if [[ ${#missing_packages[@]} -gt 0 ]]; then
                        log_info "Installing missing packages: ${missing_packages[*]}"
                        if command -v dnf &>/dev/null; then
                            sudo dnf install -y "${missing_packages[@]}"
                        else
                            sudo yum install -y "${missing_packages[@]}"
                        fi
                    else
                        log_info "All prerequisites already installed"
                    fi
                    ;;

                arch|manjaro)
                    # Arch family
                    local missing_packages=()
                    for pkg in "${packages[@]}"; do
                        if ! pacman -Qi "${pkg}" &>/dev/null; then
                            missing_packages+=("${pkg}")
                        fi
                    done

                    if [[ ${#missing_packages[@]} -gt 0 ]]; then
                        log_info "Installing missing packages: ${missing_packages[*]}"
                        sudo pacman -Sy --noconfirm "${missing_packages[@]}"
                    else
                        log_info "All prerequisites already installed"
                    fi
                    ;;

                opensuse*|sles)
                    # SUSE family
                    local missing_packages=()
                    for pkg in "${packages[@]}"; do
                        if ! rpm -q "${pkg}" &>/dev/null; then
                            missing_packages+=("${pkg}")
                        fi
                    done

                    if [[ ${#missing_packages[@]} -gt 0 ]]; then
                        log_info "Installing missing packages: ${missing_packages[*]}"
                        sudo zypper install -y "${missing_packages[@]}"
                    else
                        log_info "All prerequisites already installed"
                    fi
                    ;;

                alpine)
                    # Alpine
                    local missing_packages=()
                    for pkg in "${packages[@]}"; do
                        if ! apk info -e "${pkg}" &>/dev/null; then
                            missing_packages+=("${pkg}")
                        fi
                    done

                    if [[ ${#missing_packages[@]} -gt 0 ]]; then
                        log_info "Installing missing packages: ${missing_packages[*]}"
                        sudo apk add --no-cache "${missing_packages[@]}"
                    else
                        log_info "All prerequisites already installed"
                    fi
                    ;;

                *)
                    log_warn "Unknown Linux distribution: ${distro}"
                    log_warn "Please ensure these packages are installed: ${packages[*]}"
                    ;;
            esac
            ;;

        Darwin)
            # macOS - check for Homebrew
            if ! command -v brew &>/dev/null; then
                log_error "Homebrew is required on macOS but not found."
                log_error "Install it from https://brew.sh:"
                log_error '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                exit 1
            fi

            # On macOS, most tools come with Xcode CLI tools, but check anyway
            local missing_packages=()
            for pkg in curl git make; do
                if ! command -v "${pkg}" &>/dev/null; then
                    missing_packages+=("${pkg}")
                fi
            done
            # Check for gnu-tar specifically
            if ! command -v tar &>/dev/null; then
                missing_packages+=("gnu-tar")
            fi
            if ! command -v unzip &>/dev/null; then
                missing_packages+=("unzip")
            fi

            if [[ ${#missing_packages[@]} -gt 0 ]]; then
                log_info "Installing missing packages via Homebrew: ${missing_packages[*]}"
                brew install "${missing_packages[@]}"
            else
                log_info "All prerequisites already installed"
            fi
            ;;

        *)
            log_error "Unsupported OS: ${os}"
            exit 1
            ;;
    esac

    # Verify critical tools are available
    for tool in curl tar unzip make git; do
        if ! command -v "${tool}" &>/dev/null; then
            log_error "Required tool '${tool}' is not available after installation"
            exit 1
        fi
    done

    log_info "All prerequisites verified"
}

# Install uv if not present
install_uv() {
    if command -v uv &>/dev/null; then
        log_info "uv is already installed: $(uv --version)"
        return 0
    fi

    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source the env to get uv in PATH
    if [[ -f "${HOME}/.local/bin/env" ]]; then
        # shellcheck source=/dev/null
        source "${HOME}/.local/bin/env"
    elif [[ -f "${HOME}/.cargo/env" ]]; then
        # shellcheck source=/dev/null
        source "${HOME}/.cargo/env"
    fi

    # Add to PATH for this session
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

    if ! command -v uv &>/dev/null; then
        log_error "Failed to install uv"
        exit 1
    fi

    log_info "uv installed: $(uv --version)"
}

# Create virtual environment with specified Python version
create_venv() {
    if [[ -d "${VENV_DIR}" ]] && [[ -f "${VENV_DIR}/bin/activate" ]]; then
        log_info "Virtual environment already exists at ${VENV_DIR}"
        return 0
    fi

    log_info "Creating virtual environment with Python ${PYTHON_VERSION} at ${VENV_DIR}..."
    mkdir -p "$(dirname "${VENV_DIR}")"

    uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"

    log_info "Virtual environment created successfully"
}

# Install Ansible into the virtual environment
install_ansible() {
    local venv_pip="${VENV_DIR}/bin/pip"
    local venv_ansible="${VENV_DIR}/bin/ansible-playbook"

    if [[ -f "${venv_ansible}" ]]; then
        log_info "Ansible is already installed in venv"
        return 0
    fi

    log_info "Installing Ansible into virtual environment..."

    # Use uv pip for faster installation
    uv pip install --python "${VENV_DIR}/bin/python" ansible

    if [[ ! -f "${venv_ansible}" ]]; then
        log_error "Failed to install Ansible"
        exit 1
    fi

    log_info "Ansible installed: $("${VENV_DIR}/bin/ansible" --version | head -1)"
}

# Install Ansible Galaxy collections
install_collections() {
    local requirements_file="${SCRIPT_DIR}/requirements.yml"
    local venv_galaxy="${VENV_DIR}/bin/ansible-galaxy"

    if [[ ! -f "${requirements_file}" ]]; then
        log_warn "No requirements.yml found, skipping collection installation"
        return 0
    fi

    log_info "Installing Ansible Galaxy collections..."
    "${venv_galaxy}" collection install -r "${requirements_file}"
}

# Ensure bin directory exists and is in PATH
setup_bin_dir() {
    mkdir -p "${BIN_DIR}"

    if [[ ":${PATH}:" != *":${BIN_DIR}:"* ]]; then
        log_warn "Adding ${BIN_DIR} to PATH for this session"
        export PATH="${BIN_DIR}:${PATH}"
    fi
}

# Run the Ansible playbook
run_playbook() {
    local playbook="${SCRIPT_DIR}/playbooks/install-cloud-tools.yml"
    local venv_ansible="${VENV_DIR}/bin/ansible-playbook"

    if [[ ! -f "${playbook}" ]]; then
        log_error "Playbook not found: ${playbook}"
        exit 1
    fi

    log_info "Running Ansible playbook..."

    # Determine which runtimes to install based on INSTALL_RUNTIME
    local install_podman="true"
    local install_docker="true"

    case "${INSTALL_RUNTIME}" in
        podman)
            install_docker="false"
            ;;
        docker)
            install_podman="false"
            ;;
        none)
            install_podman="false"
            install_docker="false"
            ;;
        both|*)
            # Install both (default)
            ;;
    esac

    # Pass configuration as extra vars
    "${venv_ansible}" "${playbook}" \
        -e "venv_dir=${VENV_DIR}" \
        -e "bin_dir=${BIN_DIR}" \
        -e "python_version=${PYTHON_VERSION}" \
        -e "install_podman=${install_podman}" \
        -e "install_docker=${install_docker}" \
        "${ANSIBLE_ARGS[@]}"
}

main() {
    local platform
    platform="$(detect_platform)"

    log_info "Platform detected: ${platform}"
    log_info "Configuration:"
    log_info "  Python version:      ${PYTHON_VERSION}"
    log_info "  Venv directory:      ${VENV_DIR}"
    log_info "  Bin directory:       ${BIN_DIR}"
    log_info "  Container runtime:   ${INSTALL_RUNTIME}"
    echo

    install_prerequisites
    install_uv
    create_venv
    install_ansible
    install_collections
    setup_bin_dir
    run_playbook

    echo
    log_info "Installation complete!"
    log_info "Ensure ${BIN_DIR} is in your PATH:"
    log_info "  export PATH=\"${BIN_DIR}:\${PATH}\""
}

main
