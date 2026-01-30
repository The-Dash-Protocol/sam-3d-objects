# SAM 3D Objects - RunPod Deployment
# CUDA 12.1 + Python 3.11 base image

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd-dev \
    libglfw3 \
    libglfw3-dev \
    libosmesa6-dev \
    freeglut3-dev \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Set pip extra index URLs
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# Install PyTorch first (for CUDA 12.1)
RUN pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install NVIDIA packages
RUN pip install nvidia-cuda-nvcc-cu12==12.1.105 nvidia-pyindex cuda-python==12.1.0

# Install core dependencies (Step 1)
RUN pip install -e '.[dev]' --no-build-isolation || pip install -e '.'

# Install PyTorch3D (Step 2 - separate to handle dependency issues)
RUN pip install git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47

# Install flash-attn (may take a while to compile)
RUN pip install flash_attn==2.8.3 --no-build-isolation

# Install inference dependencies (kaolin, gsplat, gradio)
RUN pip install kaolin==0.17.0 && \
    pip install git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7 && \
    pip install gradio==5.49.0 seaborn==0.13.2

# Install RunPod SDK
RUN pip install runpod

# Install HuggingFace CLI for checkpoint download
RUN pip install 'huggingface-hub[cli]<1.0'

# Apply Hydra patch if exists
RUN if [ -f "/app/patching/hydra" ]; then chmod +x /app/patching/hydra && /app/patching/hydra; fi

# Create checkpoints directory
RUN mkdir -p /app/checkpoints

# Copy RunPod handler
COPY runpod/handler.py /app/handler.py
COPY runpod/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Set environment variables for inference
ENV LIDRA_SKIP_INIT=true
ENV PYTHONPATH=/app:/app/notebook

# Expose port for Gradio (optional web interface)
EXPOSE 7860

# Default command (can be overridden)
CMD ["/app/start.sh"]
