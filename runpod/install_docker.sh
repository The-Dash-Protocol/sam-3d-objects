#!/bin/bash
# Docker Installation Script for Ubuntu (run as root)
# Usage: bash install_docker.sh

set -e

echo "=========================================="
echo "Docker + NVIDIA Container Toolkit Install"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (or use sudo)"
    exit 1
fi

# Update packages
echo "[1/7] Updating packages..."
apt-get update

# Install prerequisites
echo "[2/7] Installing prerequisites..."
apt-get install -y ca-certificates curl gnupg

# Add Docker GPG key
echo "[3/7] Adding Docker GPG key..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo "[4/7] Adding Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
echo "[5/7] Installing Docker..."
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start Docker
echo "[6/7] Starting Docker service..."
systemctl start docker
systemctl enable docker

# Install NVIDIA Container Toolkit (for GPU support)
echo "[7/7] Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit

# Configure NVIDIA runtime
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""

# Verify installation
echo "Docker version:"
docker --version
echo ""

echo "Testing NVIDIA GPU access..."
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi || echo "GPU test failed (may need driver install)"

echo ""
echo "Done! Docker is ready to use."
