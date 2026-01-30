#!/bin/bash
# SAM 3D Objects - RunPod Startup Script

set -e

echo "=========================================="
echo "SAM 3D Objects - RunPod Startup"
echo "=========================================="

# Set environment variables
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LIDRA_SKIP_INIT=true
export PYTHONPATH=/app:/app/notebook:${PYTHONPATH}

# Function to download checkpoint from HuggingFace
download_checkpoint() {
    echo "Downloading model checkpoint from HuggingFace..."

    if [ -z "$HF_TOKEN" ]; then
        echo "ERROR: HF_TOKEN environment variable is required for checkpoint download."
        echo "Please set HF_TOKEN in your RunPod environment settings."
        exit 1
    fi

    # Login to HuggingFace
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

    # Create checkpoint directory
    mkdir -p /app/checkpoints

    # Download checkpoint
    TAG="hf"
    echo "Downloading facebook/sam-3d-objects..."
    huggingface-cli download \
        --repo-type model \
        --local-dir /app/checkpoints/${TAG}-download \
        facebook/sam-3d-objects

    # Move checkpoints to correct location
    if [ -d "/app/checkpoints/${TAG}-download/checkpoints" ]; then
        mv /app/checkpoints/${TAG}-download/checkpoints /app/checkpoints/${TAG}
        rm -rf /app/checkpoints/${TAG}-download
    else
        mv /app/checkpoints/${TAG}-download /app/checkpoints/${TAG}
    fi

    echo "Checkpoint downloaded successfully!"
}

# Check for existing checkpoint
CHECKPOINT_PATH="/app/checkpoints/hf/pipeline.yaml"
VOLUME_CHECKPOINT_PATH="/runpod-volume/checkpoints/hf/pipeline.yaml"

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Using local checkpoint: $CHECKPOINT_PATH"
    export CHECKPOINT_PATH="$CHECKPOINT_PATH"
elif [ -f "$VOLUME_CHECKPOINT_PATH" ]; then
    echo "Using volume checkpoint: $VOLUME_CHECKPOINT_PATH"
    export CHECKPOINT_PATH="$VOLUME_CHECKPOINT_PATH"
elif [ "$SKIP_DOWNLOAD" != "true" ]; then
    # Try to download if HF_TOKEN is available
    if [ -n "$HF_TOKEN" ]; then
        download_checkpoint
        export CHECKPOINT_PATH="$CHECKPOINT_PATH"
    else
        echo "WARNING: No checkpoint found and HF_TOKEN not set."
        echo "Please either:"
        echo "  1. Set HF_TOKEN environment variable to auto-download"
        echo "  2. Mount checkpoint at /app/checkpoints/hf/"
        echo "  3. Mount checkpoint at /runpod-volume/checkpoints/hf/"
    fi
fi

# Determine run mode
MODE=${RUN_MODE:-"serverless"}

case $MODE in
    "serverless")
        echo "Starting in Serverless mode..."
        export AUTO_LOAD_MODEL=${AUTO_LOAD_MODEL:-"true"}
        exec python /app/handler.py
        ;;

    "api")
        echo "Starting in API server mode..."
        exec python /app/runpod/api_server.py
        ;;

    "gradio")
        echo "Starting Gradio web interface..."
        exec python /app/runpod/gradio_app.py
        ;;

    "jupyter")
        echo "Starting Jupyter Lab..."
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
        ;;

    "shell")
        echo "Starting interactive shell..."
        exec /bin/bash
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: serverless, api, gradio, jupyter, shell"
        exit 1
        ;;
esac
