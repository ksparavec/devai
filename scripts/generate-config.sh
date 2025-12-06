#!/bin/bash
# Generate deployment-specific configuration files from common YAML sources
# Usage: ./scripts/generate-config.sh [compose|terraform|kubernetes] [profile]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"
DEPLOY_DIR="$PROJECT_ROOT/deploy"

# Default values
TARGET="${1:-all}"
PROFILE="${2:-dev}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check for yq (YAML processor)
check_dependencies() {
    if ! command -v yq &> /dev/null; then
        log_error "yq is required but not installed."
        log_info "Install with: pip install yq  OR  brew install yq  OR  snap install yq"
        exit 1
    fi
}

# Read value from YAML file
yaml_get() {
    local file="$1"
    local path="$2"
    local default="${3:-}"

    if [[ -f "$file" ]]; then
        result=$(yq -r "$path // empty" "$file" 2>/dev/null)
        if [[ -n "$result" && "$result" != "null" ]]; then
            echo "$result"
            return
        fi
    fi
    echo "$default"
}

# Generate Docker Compose .env file
generate_compose_env() {
    local profile="$1"
    local output="$DEPLOY_DIR/compose/.env"

    log_info "Generating Compose .env for profile: $profile"

    # Read from config files
    local app_name=$(yaml_get "$CONFIG_DIR/common/env.yaml" ".app.name" "devai-lab")
    local jupyter_port=$(yaml_get "$CONFIG_DIR/common/ports.yaml" ".services.jupyter.host" "8888")
    local ollama_port=$(yaml_get "$CONFIG_DIR/common/ports.yaml" ".services.ollama.host" "11434")
    local cpu=$(yaml_get "$CONFIG_DIR/profiles/${profile}.yaml" ".resources.cpu" "2")
    local memory=$(yaml_get "$CONFIG_DIR/profiles/${profile}.yaml" ".resources.memory" "4096")

    cat > "$output" << EOF
# Generated from config/ - do not edit directly
# Profile: $profile
# Generated: $(date -Iseconds)

# Image settings
IMAGE_NAME=$app_name
IMAGE_TAG=latest

# Port mappings
PORT=$jupyter_port
OLLAMA_PORT=$ollama_port

# Ollama host
OLLAMA_HOST=http://ollama:11434

# Working directory mount
HOST_WORK_DIR=./work

# User mapping
USER_ID=$(id -u)
GROUP_ID=$(id -g)
CONTAINER_USER=devai

# Resource hints (for documentation, compose uses deploy.resources)
# CPU=$cpu
# MEMORY=$memory

# API Keys (set these manually or via environment)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
EOF

    log_info "Generated: $output"
}

# Generate Terraform tfvars file
generate_terraform_tfvars() {
    local profile="$1"
    local cloud="$2"
    local output="$DEPLOY_DIR/terraform/${cloud}/terraform.tfvars"

    log_info "Generating Terraform tfvars for $cloud (profile: $profile)"

    # Read from config files
    local app_name=$(yaml_get "$CONFIG_DIR/common/env.yaml" ".app.name" "devai-lab")
    local cpu=$(yaml_get "$CONFIG_DIR/profiles/${profile}.yaml" ".resources.cpu" "2")
    local memory=$(yaml_get "$CONFIG_DIR/profiles/${profile}.yaml" ".resources.memory" "4096")
    local storage=$(yaml_get "$CONFIG_DIR/profiles/${profile}.yaml" ".resources.storage" "50")
    local replicas=$(yaml_get "$CONFIG_DIR/profiles/${profile}.yaml" ".replicas" "1")
    local https=$(yaml_get "$CONFIG_DIR/profiles/${profile}.yaml" ".features.https" "false")

    cat > "$output" << EOF
# Generated from config/ - do not edit directly
# Profile: $profile
# Generated: $(date -Iseconds)

project_name = "$app_name"
environment  = "$profile"

# Resources
cpu            = $cpu
memory         = $memory
storage_size_gb = $storage

# Scaling
replicas = $replicas

# Features
enable_https = $https
enable_gpu   = false

# Network (customize as needed)
# allowed_cidrs = ["0.0.0.0/0"]

# Region (required - set this manually)
# region = "us-east-1"  # AWS
# region = "eastus"     # Azure
# region = "us-central1" # GCP
EOF

    log_info "Generated: $output"
}

# Generate Kubernetes ConfigMap
generate_kubernetes_configmap() {
    local profile="$1"
    local output="$DEPLOY_DIR/kubernetes/base/configmap.yaml"

    log_info "Generating Kubernetes ConfigMap for profile: $profile"

    # Read from config files
    local app_name=$(yaml_get "$CONFIG_DIR/common/env.yaml" ".app.name" "devai-lab")
    local jupyter_port=$(yaml_get "$CONFIG_DIR/common/ports.yaml" ".services.jupyter.container" "8888")
    local ollama_host=$(yaml_get "$CONFIG_DIR/common/env.yaml" ".ollama.host" "http://ollama:11434")

    cat > "$output" << EOF
# Generated from config/ - do not edit directly
# Profile: $profile
# Generated: $(date -Iseconds)
apiVersion: v1
kind: ConfigMap
metadata:
  name: devai-config
  labels:
    app: $app_name
data:
  JUPYTER_PORT: "$jupyter_port"
  OLLAMA_HOST: "$ollama_host"
  CONTAINER_USER: "devai"
EOF

    log_info "Generated: $output"
}

# Main execution
main() {
    check_dependencies

    log_info "Generating configuration for target: $TARGET, profile: $PROFILE"

    case "$TARGET" in
        compose)
            generate_compose_env "$PROFILE"
            ;;
        terraform)
            for cloud in aws azure gcp; do
                if [[ -d "$DEPLOY_DIR/terraform/$cloud" ]]; then
                    generate_terraform_tfvars "$PROFILE" "$cloud"
                fi
            done
            ;;
        kubernetes)
            generate_kubernetes_configmap "$PROFILE"
            ;;
        all)
            generate_compose_env "$PROFILE"
            for cloud in aws azure gcp; do
                if [[ -d "$DEPLOY_DIR/terraform/$cloud" ]]; then
                    generate_terraform_tfvars "$PROFILE" "$cloud"
                fi
            done
            generate_kubernetes_configmap "$PROFILE"
            ;;
        *)
            log_error "Unknown target: $TARGET"
            echo "Usage: $0 [compose|terraform|kubernetes|all] [dev|staging|prod]"
            exit 1
            ;;
    esac

    log_info "Configuration generation complete!"
}

main "$@"
