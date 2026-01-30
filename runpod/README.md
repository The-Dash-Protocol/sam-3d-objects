# SAM 3D Objects - RunPod Deployment Guide

This guide explains how to deploy the SAM 3D Objects model on RunPod.

## Table of Contents

1. [Requirements](#requirements)
2. [Deployment Options](#deployment-options)
3. [Building Docker Image](#building-docker-image)
4. [RunPod Serverless Deployment](#runpod-serverless-deployment)
5. [RunPod GPU Pod Deployment](#runpod-gpu-pod-deployment)
6. [API Usage](#api-usage)
7. [Troubleshooting](#troubleshooting)

---

## Requirements

### GPU Requirements
- **Minimum 32GB VRAM** (A100 40GB, A100 80GB, A6000 recommended)
- CUDA 12.1 compatible GPU

### Software
- Docker with NVIDIA Container Toolkit
- HuggingFace account and access token (for checkpoint download)

### HuggingFace Checkpoint Access
1. Visit [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects) model page
2. Click "Access repository" button to request access
3. After approval, generate HuggingFace token: https://huggingface.co/settings/tokens

---

## Deployment Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **Serverless** | Pay-per-request, auto-scaling | Cost efficient, no management | Cold start latency |
| **GPU Pod** | Always-on instance | Instant response, SSH access | Continuous cost |

---

## Building Docker Image

### Build Locally

```bash
# Run from project root
cd /path/to/sam-3d-objects

# Build Docker image (takes ~30-60 minutes)
docker build -t sam3d-objects:latest -f runpod/Dockerfile .

# Push to DockerHub or registry
docker tag sam3d-objects:latest your-registry/sam3d-objects:latest
docker push your-registry/sam3d-objects:latest
```

### Build Notes

- PyTorch3D and Flash Attention compilation takes time
- Must build in CUDA 12.1 environment
- Build machine needs sufficient RAM (16GB+)

---

## RunPod Serverless Deployment

### 1. Create Template

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Select "Custom Template"
3. Enter the following settings:

```
Container Image: your-registry/sam3d-objects:latest
Container Start Command: /app/start.sh
Container Disk: 50 GB (if including checkpoints)
```

### 2. Set Environment Variables

```
HF_TOKEN=hf_xxxxxxxxxxxxx  # HuggingFace token
RUN_MODE=serverless
AUTO_LOAD_MODEL=true
```

### 3. GPU Settings

```
GPU Type: A100 40GB or A100 80GB
Max Workers: 1-5 (as needed)
Idle Timeout: 30 (seconds)
```

### 4. Deploy

Click "Deploy" and note the endpoint URL

---

## RunPod GPU Pod Deployment

### 1. Create Pod

1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Click "Deploy"
3. Select GPU: A100 40GB/80GB or A6000
4. Select template: "RunPod Pytorch 2.1" or custom image

### 2. Setup via SSH

```bash
# Clone project
git clone https://github.com/facebookresearch/sam-3d-objects.git
cd sam-3d-objects

# Set environment
export HF_TOKEN="hf_xxxxxxxxxxxxx"
export CUDA_HOME=/usr/local/cuda

# Install without Docker
pip install -r requirements.txt
pip install -e '.[dev]'
pip install -e '.[p3d]'
pip install -e '.[inference]'

# Download checkpoint
huggingface-cli login --token $HF_TOKEN
huggingface-cli download --repo-type model --local-dir checkpoints/hf-download facebook/sam-3d-objects
mv checkpoints/hf-download/checkpoints checkpoints/hf

# Start API server
python runpod/api_server.py

# Or Gradio web interface
python runpod/gradio_app.py
```

### 3. Volume Mount (Recommended)

Save checkpoints to Network Volume to avoid re-downloading on Pod restart:

```bash
# Volume mount path: /runpod-volume
mkdir -p /runpod-volume/checkpoints
mv checkpoints/hf /runpod-volume/checkpoints/
```

---

## API Usage

### Serverless API Call

```python
import requests
import base64

# Encode image to base64
with open("image.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Mask (optional)
with open("mask.png", "rb") as f:
    mask_base64 = base64.b64encode(f.read()).decode()

# API call
response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={
        "Authorization": "Bearer YOUR_RUNPOD_API_KEY",
        "Content-Type": "application/json"
    },
    json={
        "input": {
            "image": image_base64,
            "mask": mask_base64,  # optional
            "seed": 42,
            "output_format": "ply"  # "ply", "glb", or "both"
        }
    }
)

result = response.json()

# Save PLY file
if "output" in result and "ply" in result["output"]:
    ply_data = base64.b64decode(result["output"]["ply"])
    with open("output.ply", "wb") as f:
        f.write(ply_data)
```

### Direct API Server Call (GPU Pod)

```python
import requests
import base64

# Encode image
with open("image.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# API call
response = requests.post(
    "http://YOUR_POD_IP:8000/predict",
    json={
        "image": image_base64,
        "seed": 42,
        "output_format": "ply"
    }
)

result = response.json()
print(result["status"])
```

### cURL Example

```bash
# Convert image to base64
IMAGE_BASE64=$(base64 -i image.png)

# Serverless call
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"image\": \"$IMAGE_BASE64\",
      \"seed\": 42
    }
  }"
```

---

## Response Format

```json
{
  "status": "success",
  "ply": "<base64 encoded PLY file>",
  "glb": "<base64 encoded GLB file (optional)>",
  "rotation": [[w, x, y, z]],
  "translation": [[x, y, z]],
  "scale": [[sx, sy, sz]]
}
```

---

## Troubleshooting

### 1. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
- Use larger GPU (A100 80GB recommended)
- Reduce input image size
- Process single images instead of batches

### 2. Checkpoint Download Failed

```
Error: Model checkpoint not found
```

**Solution:**
- Verify HF_TOKEN environment variable
- Check HuggingFace model access permissions
- Download manually and mount via Volume

### 3. PyTorch3D Compilation Error

**Solution:**
- Verify CUDA 12.1 environment
- Check GCC version compatibility (GCC 12 recommended)
- Use pre-built Docker image

### 4. Cold Start Delay

First request on Serverless has delay due to model loading

**Solution:**
- Set `AUTO_LOAD_MODEL=true` to load model at startup
- Increase Idle Timeout
- Use GPU Pod for always-on availability

---

## Cost Estimates

### RunPod Serverless
- A100 40GB: ~$0.0013/sec ($4.68/hour)
- A100 80GB: ~$0.0019/sec ($6.84/hour)
- Inference time: ~30-60 seconds/request

### RunPod GPU Pod
- A100 40GB: ~$1.69/hour (Community Cloud)
- A100 80GB: ~$2.29/hour (Community Cloud)
- A6000 48GB: ~$0.79/hour (Community Cloud)

---

## File Structure

```
runpod/
├── Dockerfile          # Docker image build configuration
├── handler.py          # RunPod Serverless handler
├── api_server.py       # Flask REST API server
├── gradio_app.py       # Gradio web interface
├── start.sh            # Container startup script
├── docker-compose.yml  # Local testing Compose configuration
└── README.md           # This document
```

---

## License

These deployment scripts are part of the SAM 3D Objects project and follow the project's license.
