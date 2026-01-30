#!/usr/bin/env python3
"""
SAM 3D Objects - RunPod Serverless Handler

This handler provides a REST API for 3D object reconstruction from single images.

Input format:
{
    "input": {
        "image": "<base64 encoded image>",
        "mask": "<base64 encoded mask (optional)>",
        "seed": 42,  # optional, default: random
        "output_format": "ply"  # "ply" | "glb" | "both"
    }
}

Output format:
{
    "output": {
        "ply": "<base64 encoded PLY file>",
        "glb": "<base64 encoded GLB file (if requested)>",
        "status": "success"
    }
}
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

# Add notebook to path for inference module
sys.path.insert(0, "/app/notebook")
sys.path.insert(0, "/app")

import runpod
import numpy as np
from PIL import Image

# Global inference model (loaded once)
_inference_model = None


def get_inference_model():
    """Lazy-load the inference model to avoid loading on cold start."""
    global _inference_model

    if _inference_model is None:
        from inference import Inference

        # Check for checkpoint
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
            raise RuntimeError(
                "Model checkpoint not found. Please download the checkpoint first. "
                "See README for instructions."
            )

        print(f"Loading model from: {config_path}")
        _inference_model = Inference(config_path, compile=False)
        print("Model loaded successfully!")

    return _inference_model


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    # Remove data URL prefix if present
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
        mask = mask[..., -1]  # Use alpha channel or last channel
    return mask


def encode_file_to_base64(file_path: str) -> str:
    """Encode file to base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job: dict) -> dict:
    """
    RunPod handler function for SAM 3D Objects inference.

    Args:
        job: RunPod job dictionary containing input data

    Returns:
        Dictionary containing output data or error message
    """
    try:
        job_input = job.get("input", {})

        # Validate required inputs
        if "image" not in job_input:
            return {"error": "Missing required 'image' field in input"}

        # Decode image
        image = decode_base64_image(job_input["image"])

        # Decode mask (optional)
        mask = None
        if "mask" in job_input and job_input["mask"]:
            mask = decode_base64_mask(job_input["mask"])
        else:
            # Create full mask if not provided (entire image)
            mask = np.ones(image.shape[:2], dtype=bool)

        # Get optional parameters
        seed = job_input.get("seed", None)
        output_format = job_input.get("output_format", "ply").lower()

        # Load model
        inference = get_inference_model()

        # Run inference
        print(f"Running inference with seed={seed}")
        output = inference(image, mask, seed=seed)

        # Prepare output
        result = {"status": "success"}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Export PLY
            if output_format in ["ply", "both"]:
                ply_path = os.path.join(tmpdir, "output.ply")
                output["gs"].save_ply(ply_path)
                result["ply"] = encode_file_to_base64(ply_path)

            # Export GLB (mesh) if requested
            if output_format in ["glb", "both"]:
                try:
                    glb_path = os.path.join(tmpdir, "output.glb")
                    if "mesh" in output and output["mesh"] is not None:
                        # Export mesh to GLB
                        output["mesh"].export(glb_path)
                        result["glb"] = encode_file_to_base64(glb_path)
                    else:
                        result["glb_error"] = "Mesh output not available"
                except Exception as e:
                    result["glb_error"] = str(e)

        # Include additional metadata
        if "rotation" in output:
            result["rotation"] = output["rotation"].cpu().numpy().tolist()
        if "translation" in output:
            result["translation"] = output["translation"].cpu().numpy().tolist()
        if "scale" in output:
            result["scale"] = output["scale"].cpu().numpy().tolist()

        return result

    except Exception as e:
        traceback.print_exc()
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Health check endpoint for RunPod
def health_check():
    """Check if the model is loaded and ready."""
    try:
        get_inference_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    print("Starting SAM 3D Objects RunPod Handler...")

    # Pre-load model if AUTO_LOAD environment variable is set
    if os.environ.get("AUTO_LOAD_MODEL", "false").lower() == "true":
        print("Pre-loading model...")
        try:
            get_inference_model()
        except Exception as e:
            print(f"Warning: Could not pre-load model: {e}")

    # Start RunPod serverless handler
    runpod.serverless.start({"handler": handler})
