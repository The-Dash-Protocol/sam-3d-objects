#!/bin/bash
# SAM 3D Objects - Direct Setup for RunPod
# Run this script on a RunPod GPU pod (no Docker needed)
# Usage: bash setup_runpod.sh

set -e

echo "=========================================="
echo "SAM 3D Objects - RunPod Setup"
echo "=========================================="

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is required"
    echo "Usage: HF_TOKEN=hf_xxxxx bash setup_runpod.sh"
    exit 1
fi

# Set environment
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

echo "[1/6] Installing PyTorch (CUDA 12.1)..."
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[2/6] Installing core dependencies..."
pip install -e '.[dev]' || pip install -e '.'

echo "[3/6] Installing PyTorch3D (this may take a while)..."
pip install git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47

echo "[4/6] Installing inference dependencies..."
pip install kaolin==0.17.0
pip install git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7 || echo "gsplat install failed, continuing..."
pip install gradio==5.49.0 flask runpod seaborn==0.13.2

echo "[5/6] Downloading model checkpoint..."
pip install 'huggingface-hub[cli]<1.0'
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

mkdir -p checkpoints
huggingface-cli download \
    --repo-type model \
    --local-dir checkpoints/hf-download \
    facebook/sam-3d-objects

# Move to correct location
if [ -d "checkpoints/hf-download/checkpoints" ]; then
    mv checkpoints/hf-download/checkpoints checkpoints/hf
    rm -rf checkpoints/hf-download
else
    mv checkpoints/hf-download checkpoints/hf
fi

echo "[6/6] Applying Hydra patch..."
if [ -f "patching/hydra" ]; then
    chmod +x patching/hydra
    ./patching/hydra || echo "Hydra patch skipped"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start API server:"
echo "  python runpod/api_server.py"
echo ""
echo "To start Gradio UI:"
echo "  python runpod/gradio_app.py"
echo ""
echo "To run demo:"
echo "  python demo.py"
echo ""
