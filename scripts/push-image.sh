#!/bin/bash
# Push container image to cloud registries
# Usage: ./scripts/push-image.sh [aws|azure|gcp] [--gpu]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
CLOUD="${1:-}"
GPU_FLAG=""
IMAGE_TAG="${IMAGE_TAG:-latest}"
PROJECT_NAME="${PROJECT_NAME:-devai-lab}"

# Parse arguments
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_FLAG="-gpu"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Get Terraform output
tf_output() {
    local cloud="$1"
    local output="$2"
    cd "$PROJECT_ROOT/deploy/terraform/$cloud"
    terraform output -raw "$output" 2>/dev/null || echo ""
}

# Build image
build_image() {
    log_info "Building image..."
    cd "$PROJECT_ROOT"

    if [[ -n "$GPU_FLAG" ]]; then
        docker build -f Dockerfile.gpu -t "${PROJECT_NAME}${GPU_FLAG}:${IMAGE_TAG}" .
    else
        docker build -t "${PROJECT_NAME}:${IMAGE_TAG}" .
    fi
}

# Push to AWS ECR
push_aws() {
    log_info "Pushing to AWS ECR..."

    local ecr_url
    ecr_url=$(tf_output aws ecr_repository_url)

    if [[ -z "$ecr_url" ]]; then
        log_error "ECR URL not found. Run 'make tf-apply-aws' first."
        exit 1
    fi

    local region
    region=$(echo "$ecr_url" | cut -d. -f4)

    log_info "Logging in to ECR..."
    aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$ecr_url"

    log_info "Tagging and pushing..."
    docker tag "${PROJECT_NAME}${GPU_FLAG}:${IMAGE_TAG}" "${ecr_url}:${IMAGE_TAG}"
    docker push "${ecr_url}:${IMAGE_TAG}"

    log_info "Pushed to: ${ecr_url}:${IMAGE_TAG}"
}

# Push to Azure ACR
push_azure() {
    log_info "Pushing to Azure ACR..."

    local acr_name
    acr_name=$(tf_output azure acr_name)

    if [[ -z "$acr_name" ]]; then
        log_error "ACR name not found. Run 'make tf-apply-azure' first."
        exit 1
    fi

    local acr_server
    acr_server=$(tf_output azure acr_login_server)

    log_info "Logging in to ACR..."
    az acr login --name "$acr_name"

    log_info "Tagging and pushing..."
    docker tag "${PROJECT_NAME}${GPU_FLAG}:${IMAGE_TAG}" "${acr_server}/${PROJECT_NAME}${GPU_FLAG}:${IMAGE_TAG}"
    docker push "${acr_server}/${PROJECT_NAME}${GPU_FLAG}:${IMAGE_TAG}"

    log_info "Pushed to: ${acr_server}/${PROJECT_NAME}${GPU_FLAG}:${IMAGE_TAG}"
}

# Push to GCP Artifact Registry
push_gcp() {
    log_info "Pushing to GCP Artifact Registry..."

    local ar_url
    ar_url=$(tf_output gcp artifact_registry_url)

    if [[ -z "$ar_url" ]]; then
        log_error "Artifact Registry URL not found. Run 'make tf-apply-gcp' first."
        exit 1
    fi

    local region
    region=$(echo "$ar_url" | cut -d- -f1-2)

    log_info "Configuring Docker for Artifact Registry..."
    gcloud auth configure-docker "${region}-docker.pkg.dev" --quiet

    log_info "Tagging and pushing..."
    docker tag "${PROJECT_NAME}${GPU_FLAG}:${IMAGE_TAG}" "${ar_url}/${PROJECT_NAME}${GPU_FLAG}:${IMAGE_TAG}"
    docker push "${ar_url}/${PROJECT_NAME}${GPU_FLAG}:${IMAGE_TAG}"

    log_info "Pushed to: ${ar_url}/${PROJECT_NAME}${GPU_FLAG}:${IMAGE_TAG}"
}

# Main
main() {
    if [[ -z "$CLOUD" ]]; then
        echo "Usage: $0 [aws|azure|gcp] [--gpu]"
        echo ""
        echo "Options:"
        echo "  aws     Push to AWS ECR"
        echo "  azure   Push to Azure ACR"
        echo "  gcp     Push to GCP Artifact Registry"
        echo "  --gpu   Build and push GPU image"
        echo ""
        echo "Environment variables:"
        echo "  IMAGE_TAG      Image tag (default: latest)"
        echo "  PROJECT_NAME   Project name (default: devai-lab)"
        exit 1
    fi

    build_image

    case "$CLOUD" in
        aws)
            push_aws
            ;;
        azure)
            push_azure
            ;;
        gcp)
            push_gcp
            ;;
        *)
            log_error "Unknown cloud: $CLOUD"
            exit 1
            ;;
    esac

    log_info "Image push complete!"
}

main "$@"
