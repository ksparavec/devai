#!/bin/bash
#
# CUDA Image Selector
# Queries Docker Hub for available NVIDIA CUDA images with cuDNN and helps user select one.
# Uses dialog for ncurses-based selection. Suggests the latest stable version from
# the previous major release for compatibility.
#

set -euo pipefail

# Determine script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DOCKER_HUB_API="https://registry.hub.docker.com/v2/repositories/nvidia/cuda/tags"
PAGE_SIZE=100
UBUNTU_VERSION="${UBUNTU_VERSION:-24.04}"
LIST_ONLY="${LIST_ONLY:-false}"
AUTO_SELECT="${AUTO_SELECT:-false}"
DOCKERFILE="${DOCKERFILE:-${PROJECT_ROOT}/Dockerfile.gpu}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --list)
            LIST_ONLY=true
            shift
            ;;
        --auto)
            AUTO_SELECT=true
            shift
            ;;
        --ubuntu)
            UBUNTU_VERSION="$2"
            shift 2
            ;;
        --dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        22.04|24.04)
            UBUNTU_VERSION="$1"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [UBUNTU_VERSION]"
            echo ""
            echo "Options:"
            echo "  --list              List available images without interactive selection"
            echo "  --auto              Automatically select the recommended version"
            echo "  --ubuntu VER        Specify Ubuntu version (default: 24.04)"
            echo "  --dockerfile FILE   Specify Dockerfile to update (default: Dockerfile.gpu)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                  # Interactive selection for Ubuntu 24.04"
            echo "  $0 22.04            # Interactive selection for Ubuntu 22.04"
            echo "  $0 --list           # List images only (non-interactive)"
            echo "  $0 --auto           # Auto-select recommended and update Dockerfile"
            exit 0
            ;;
        *)
            UBUNTU_VERSION="$1"
            shift
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BOLD}${BLUE}======================================${NC}"
    echo -e "${BOLD}${BLUE}    NVIDIA CUDA Image Selector${NC}"
    echo -e "${BOLD}${BLUE}======================================${NC}\n"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

print_info() {
    echo -e "${CYAN}$1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

# Check for required tools
check_dependencies() {
    local missing=()
    for cmd in curl python3; do
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$cmd")
        fi
    done

    if [ ${#missing[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing[*]}"
        exit 1
    fi

    # Check for dialog (optional, will fall back to basic selection)
    if ! command -v dialog &> /dev/null; then
        print_warning "Note: Install 'dialog' package for a nicer TUI experience"
        print_warning "      (apt install dialog / dnf install dialog)"
        echo ""
        HAS_DIALOG=false
    else
        HAS_DIALOG=true
    fi
}

# Fetch and parse CUDA image tags from Docker Hub (cuDNN only)
fetch_cuda_tags() {
    local ubuntu_ver="$1"
    local filter="cudnn-runtime-ubuntu${ubuntu_ver}"

    print_info "Fetching available CUDA+cuDNN images for Ubuntu ${ubuntu_ver}..."

    # Fetch tags from Docker Hub API
    local response
    response=$(curl -s "${DOCKER_HUB_API}?page_size=${PAGE_SIZE}&name=${filter}" 2>/dev/null)

    if [ -z "$response" ]; then
        print_error "Failed to fetch tags from Docker Hub"
        exit 1
    fi

    # Parse JSON and extract tag names, filtering for cudnn runtime images
    echo "$response" | python3 -c "
import sys
import json
import re

try:
    data = json.load(sys.stdin)
    results = data.get('results', [])

    tags = []
    for r in results:
        name = r.get('name', '')
        # Match pattern: X.Y.Z-cudnn-runtime-ubuntuXX.XX or X.Y.Z-cudnnX-runtime-ubuntuXX.XX
        if 'cudnn' in name and 'runtime-ubuntu' in name and 'devel' not in name:
            tags.append(name)

    for tag in sorted(tags, reverse=True):
        print(tag)
except Exception as e:
    print(f'Parse error: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# Group tags by major.minor version and get latest patch
group_by_version() {
    python3 -c "
import sys
import re
from collections import defaultdict

tags = [line.strip() for line in sys.stdin if line.strip()]
versions = defaultdict(list)

for tag in tags:
    # Extract version: e.g., 12.9.1 from 12.9.1-cudnn-runtime-ubuntu24.04
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)', tag)
    if match:
        major, minor, patch = match.groups()
        key = f'{major}.{minor}'
        versions[key].append({
            'tag': tag,
            'major': int(major),
            'minor': int(minor),
            'patch': int(patch),
            'full': f'{major}.{minor}.{patch}'
        })

# Sort and get latest patch for each major.minor
output = []
for key in sorted(versions.keys(), key=lambda x: (int(x.split('.')[0]), int(x.split('.')[1])), reverse=True):
    # Get latest patch version (highest patch number)
    latest = max(versions[key], key=lambda x: x['patch'])
    output.append({
        'major_minor': key,
        'latest_tag': latest['tag'],
        'full_version': latest['full'],
        'major': latest['major']
    })

# Print as TSV for easy parsing
for item in output:
    print(f\"{item['major_minor']}\t{item['full_version']}\t{item['latest_tag']}\t{item['major']}\")
"
}

# Find recommended version (latest from second-newest major version)
find_recommended() {
    local versions="$1"

    python3 -c "
import sys

lines = '''$versions'''.strip().split('\n')
if not lines or lines[0] == '':
    sys.exit(1)

# Parse versions
versions = []
for line in lines:
    parts = line.split('\t')
    if len(parts) >= 4:
        versions.append({
            'major_minor': parts[0],
            'full': parts[1],
            'tag': parts[2],
            'major': int(parts[3])
        })

# Find unique major versions
majors = sorted(set(v['major'] for v in versions), reverse=True)

if len(majors) >= 2:
    # Recommend latest from second-newest major version
    recommended_major = majors[1]
else:
    # Only one major version available
    recommended_major = majors[0]

# Find latest version from recommended major
for v in versions:
    if v['major'] == recommended_major:
        print(v['tag'])
        break
"
}

# Display version table (for --list mode)
display_table() {
    local versions="$1"
    local recommended="$2"
    local ubuntu_ver="$3"

    echo -e "\n${BOLD}Available CUDA+cuDNN Runtime Images (Ubuntu ${ubuntu_ver}):${NC}\n"

    printf "${BOLD}%-4s  %-10s  %-12s  %-55s${NC}\n" "#" "Branch" "Version" "Full Image Tag"
    printf "%-4s  %-10s  %-12s  %-55s\n" "---" "----------" "------------" "-------------------------------------------------------"

    local index=1
    while IFS=$'\t' read -r major_minor full_version tag major; do
        local marker=""
        local color=""

        if [ "$tag" = "$recommended" ]; then
            marker=" ${GREEN}[RECOMMENDED]${NC}"
            color="${GREEN}"
        fi

        printf "${color}%-4s${NC}  %-10s  %-12s  %-55s%b\n" \
            "$index" "$major_minor" "$full_version" "docker.io/nvidia/cuda:${tag}" "$marker"

        ((index++))
    done <<< "$versions"
}

# Dialog-based selection
select_with_dialog() {
    local versions="$1"
    local recommended="$2"
    local ubuntu_ver="$3"

    # Build dialog menu options
    local menu_items=()
    local index=1
    local recommended_index=1

    while IFS=$'\t' read -r major_minor full_version tag major; do
        local label="CUDA ${full_version}"
        if [ "$tag" = "$recommended" ]; then
            label="${label} [RECOMMENDED]"
            recommended_index=$index
        fi
        menu_items+=("$index" "$label")
        ((index++))
    done <<< "$versions"

    local count=$((index - 1))

    # Calculate dialog height
    local height=$((count + 10))
    [ $height -gt 20 ] && height=20

    # Show dialog
    local choice
    choice=$(dialog --clear --title "NVIDIA CUDA Image Selector" \
        --default-item "$recommended_index" \
        --menu "Select CUDA+cuDNN image for Ubuntu ${ubuntu_ver}:\n\nNote: CUDA 13.x is new. For best compatibility,\nthe latest 12.x version is recommended." \
        $height 70 $count \
        "${menu_items[@]}" \
        2>&1 >/dev/tty) || { clear; return 1; }

    clear

    # Get the selected tag
    echo "$versions" | sed -n "${choice}p" | cut -f3
}

# Basic selection (fallback when dialog not available)
select_basic() {
    local versions="$1"
    local recommended="$2"
    local count
    count=$(echo "$versions" | wc -l)

    echo ""
    print_warning "Note: CUDA 13.x is very new. For best compatibility with PyTorch/ChromaDB,"
    print_warning "      the latest 12.x version is recommended."
    echo ""

    while true; do
        echo -e -n "${BOLD}Enter selection (1-${count}) or 'q' to quit [recommended: press Enter]: ${NC}"
        read -r selection

        # Default to recommended
        if [ -z "$selection" ]; then
            echo "$recommended"
            return 0
        fi

        if [ "$selection" = "q" ] || [ "$selection" = "Q" ]; then
            echo ""
            return 1
        fi

        if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "$count" ]; then
            # Get the tag at this index
            echo "$versions" | sed -n "${selection}p" | cut -f3
            return 0
        fi

        print_error "Invalid selection. Please enter a number between 1 and ${count}."
    done
}

# Update Dockerfile with selected image
update_dockerfile() {
    local selected_tag="$1"
    local dockerfile="$2"

    if [ ! -f "$dockerfile" ]; then
        print_error "Dockerfile not found: $dockerfile"
        return 1
    fi

    local full_image="docker.io/nvidia/cuda:${selected_tag}"

    # Check current FROM line
    local current_from
    current_from=$(grep -E "^FROM.*nvidia/cuda" "$dockerfile" | head -1) || true

    if [ -z "$current_from" ]; then
        print_error "Could not find nvidia/cuda FROM line in $dockerfile"
        return 1
    fi

    echo ""
    print_info "Current: $current_from"
    print_info "New:     FROM ${full_image}"

    # Update the Dockerfile
    sed -i "s|^FROM.*nvidia/cuda:.*|FROM ${full_image}|" "$dockerfile"

    echo ""
    print_success "Updated ${dockerfile} with ${full_image}"
    return 0
}

# Main function
main() {
    print_header
    check_dependencies

    # Fetch tags (cuDNN only)
    local tags
    tags=$(fetch_cuda_tags "$UBUNTU_VERSION")

    if [ -z "$tags" ]; then
        print_error "No CUDA+cuDNN runtime images found for Ubuntu ${UBUNTU_VERSION}"
        exit 1
    fi

    # Group by version
    local versions
    versions=$(echo "$tags" | group_by_version)

    if [ -z "$versions" ]; then
        print_error "Failed to parse CUDA versions"
        exit 1
    fi

    # Find recommended version
    local recommended
    recommended=$(find_recommended "$versions")

    # In list-only mode, just print table and exit
    if [ "$LIST_ONLY" = "true" ]; then
        display_table "$versions" "$recommended" "$UBUNTU_VERSION"
        echo ""
        print_warning "Note: CUDA 13.x is very new. For best compatibility with PyTorch/ChromaDB,"
        print_warning "      the latest 12.x version is recommended."
        echo ""
        print_success "Recommended: docker.io/nvidia/cuda:${recommended}"
        exit 0
    fi

    # Auto-select mode: use recommended version without interaction
    if [ "$AUTO_SELECT" = "true" ]; then
        print_info "Auto-selecting recommended version..."
        echo ""
        print_success "Selected: docker.io/nvidia/cuda:${recommended}"
        if [ -f "$DOCKERFILE" ]; then
            update_dockerfile "$recommended" "$DOCKERFILE"
        else
            print_warning "Dockerfile not found: $DOCKERFILE"
            echo ""
            echo "To use this image, add to your Dockerfile:"
            echo "  FROM docker.io/nvidia/cuda:${recommended}"
        fi
        exit 0
    fi

    # Get user selection
    local selected
    if [ "$HAS_DIALOG" = "true" ]; then
        # Show table first for reference
        display_table "$versions" "$recommended" "$UBUNTU_VERSION"
        echo ""
        print_info "Opening selection dialog..."
        sleep 1

        if selected=$(select_with_dialog "$versions" "$recommended" "$UBUNTU_VERSION"); then
            :
        else
            print_info "Selection cancelled."
            exit 0
        fi
    else
        # Use basic selection
        display_table "$versions" "$recommended" "$UBUNTU_VERSION"
        if selected=$(select_basic "$versions" "$recommended"); then
            :
        else
            print_info "Selection cancelled."
            exit 0
        fi
    fi

    echo ""
    print_success "Selected: docker.io/nvidia/cuda:${selected}"

    # Update Dockerfile automatically
    if [ -f "$DOCKERFILE" ]; then
        update_dockerfile "$selected" "$DOCKERFILE"
    else
        print_warning "Dockerfile not found: $DOCKERFILE"
        echo ""
        echo "To use this image, add to your Dockerfile:"
        echo "  FROM docker.io/nvidia/cuda:${selected}"
    fi
}

# Run main function
main "$@"
