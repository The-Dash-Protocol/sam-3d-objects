#!/bin/bash
# SAM 3D Objects - Docker Build Script

set -e

# Configuration
IMAGE_NAME="${IMAGE_NAME:-sam3d-objects}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"  # e.g., "dockerhub_username" or "ghcr.io/username"
PUSH="${PUSH:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SAM 3D Objects - Docker Build${NC}"
echo -e "${GREEN}========================================${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${YELLOW}Project root: ${PROJECT_ROOT}${NC}"
echo -e "${YELLOW}Image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"

# Check for NVIDIA Docker runtime
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo -e "${YELLOW}Warning: NVIDIA Docker runtime not detected.${NC}"
    echo -e "${YELLOW}GPU compilation may not work correctly.${NC}"
fi

# Build the image
echo -e "${GREEN}Building Docker image...${NC}"
echo -e "${YELLOW}This may take 30-60 minutes on first build.${NC}"

cd "$PROJECT_ROOT"

docker build \
    --progress=plain \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f runpod/Dockerfile \
    .

echo -e "${GREEN}Build complete!${NC}"

# Tag and push if registry is specified
if [ -n "$REGISTRY" ]; then
    FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo -e "${GREEN}Tagging image as ${FULL_IMAGE}${NC}"
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_IMAGE}"

    if [ "$PUSH" = "true" ]; then
        echo -e "${GREEN}Pushing to registry...${NC}"
        docker push "${FULL_IMAGE}"
        echo -e "${GREEN}Push complete!${NC}"
    else
        echo -e "${YELLOW}To push the image, run:${NC}"
        echo -e "  docker push ${FULL_IMAGE}"
    fi
fi

# Print usage instructions
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To run locally:"
echo "  docker run --gpus all -p 8000:8000 \\"
echo "    -e HF_TOKEN=your_token \\"
echo "    -e RUN_MODE=api \\"
echo "    ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To run with Gradio UI:"
echo "  docker run --gpus all -p 7860:7860 \\"
echo "    -e HF_TOKEN=your_token \\"
echo "    -e RUN_MODE=gradio \\"
echo "    ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "For RunPod deployment, push to your registry:"
echo "  REGISTRY=your-registry PUSH=true ./runpod/build.sh"
