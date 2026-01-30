#!/usr/bin/env python3
"""
SAM 3D Objects - Standalone API Server

A Flask-based REST API server for deployment on RunPod GPU Pods
or any other GPU server environment.

Endpoints:
    POST /predict - Run 3D reconstruction
    GET /health - Health check
    GET /info - Model information
"""

import os
import sys
import base64
import tempfile
import traceback
from io import BytesIO
from typing import Optional

# Set environment before imports
os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME", "/usr/local/cuda")
os.environ["LIDRA_SKIP_INIT"] = "true"

sys.path.insert(0, "/app/notebook")
sys.path.insert(0, "/app")

from flask import Flask, request, jsonify, send_file
import numpy as np
from PIL import Image

app = Flask(__name__)

# Global model instance
_inference_model = None


def get_inference_model():
    """Lazy-load the inference model."""
    global _inference_model

    if _inference_model is None:
        from inference import Inference

        checkpoint_paths = [
            "/app/checkpoints/hf/pipeline.yaml",
            "/runpod-volume/checkpoints/hf/pipeline.yaml",
            os.environ.get("CHECKPOINT_PATH", ""),
        ]

        config_path = None
        for path in checkpoint_paths:
            if path and os.path.exists(path):
                config_path = path
                break

        if config_path is None:
            raise RuntimeError("Model checkpoint not found")

        print(f"Loading model from: {config_path}")
        _inference_model = Inference(config_path, compile=False)
        print("Model loaded successfully!")

    return _inference_model


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return np.array(image).astype(np.uint8)


def decode_base64_mask(base64_string: str) -> np.ndarray:
    """Decode base64 mask to boolean numpy array."""
    image = decode_base64_image(base64_string)
    mask = image > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    return mask


def encode_file_to_base64(file_path: str) -> str:
    """Encode file to base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        # Check if model can be loaded
        get_inference_model()
        return jsonify({
            "status": "healthy",
            "model_loaded": True
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@app.route("/info", methods=["GET"])
def info():
    """Model information endpoint."""
    return jsonify({
        "model": "SAM 3D Objects",
        "version": "1.0.0",
        "description": "Single-image 3D object reconstruction",
        "endpoints": {
            "/predict": "POST - Run 3D reconstruction",
            "/health": "GET - Health check",
            "/info": "GET - This endpoint"
        },
        "input_format": {
            "image": "base64 encoded image (required)",
            "mask": "base64 encoded mask (optional)",
            "seed": "random seed (optional)",
            "output_format": "ply | glb | both (default: ply)"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Run 3D reconstruction prediction."""
    try:
        # Get input data
        if request.is_json:
            data = request.get_json()
        else:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        # Validate required inputs
        if "image" not in data:
            return jsonify({"error": "Missing required 'image' field"}), 400

        # Decode image
        image = decode_base64_image(data["image"])

        # Decode mask (optional)
        if "mask" in data and data["mask"]:
            mask = decode_base64_mask(data["mask"])
        else:
            mask = np.ones(image.shape[:2], dtype=bool)

        # Get optional parameters
        seed = data.get("seed", None)
        output_format = data.get("output_format", "ply").lower()

        # Run inference
        inference = get_inference_model()
        print(f"Running inference with seed={seed}")
        output = inference(image, mask, seed=seed)

        # Prepare result
        result = {"status": "success"}

        with tempfile.TemporaryDirectory() as tmpdir:
            if output_format in ["ply", "both"]:
                ply_path = os.path.join(tmpdir, "output.ply")
                output["gs"].save_ply(ply_path)
                result["ply"] = encode_file_to_base64(ply_path)

            if output_format in ["glb", "both"]:
                try:
                    glb_path = os.path.join(tmpdir, "output.glb")
                    if "mesh" in output and output["mesh"] is not None:
                        output["mesh"].export(glb_path)
                        result["glb"] = encode_file_to_base64(glb_path)
                    else:
                        result["glb_error"] = "Mesh output not available"
                except Exception as e:
                    result["glb_error"] = str(e)

        # Include metadata
        if "rotation" in output:
            result["rotation"] = output["rotation"].cpu().numpy().tolist()
        if "translation" in output:
            result["translation"] = output["translation"].cpu().numpy().tolist()
        if "scale" in output:
            result["scale"] = output["scale"].cpu().numpy().tolist()

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"Starting API server on {host}:{port}")

    # Pre-load model if requested
    if os.environ.get("AUTO_LOAD_MODEL", "true").lower() == "true":
        print("Pre-loading model...")
        try:
            get_inference_model()
        except Exception as e:
            print(f"Warning: Could not pre-load model: {e}")

    # Run Flask server
    app.run(host=host, port=port, threaded=True)
