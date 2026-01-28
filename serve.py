# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
SAM 3D Objects API Server using LitServe

Usage:
    python serve.py

API Endpoint:
    POST /predict

Request Body (JSON):
    {
        "image": "<base64 encoded image>",
        "masks": ["<base64 encoded mask 1>", "<base64 encoded mask 2>", ...],
        "seed": 42  // optional
    }

Response (JSON):
    {
        "objects": [
            {
                "index": 0,
                "rotation": [x, y, z, w],  // quaternion
                "translation": [x, y, z],
                "scale": [sx, sy, sz],
                "mesh_glb": "<base64 encoded GLB>"  // optional, if mesh available
            },
            ...
        ],
        "success": true
    }
"""

import sys
import os
import base64
import io
from typing import Optional

import numpy as np
from PIL import Image
import litserve as ls

# Add notebook to path for inference imports
sys.path.append(os.path.join(os.path.dirname(__file__), "notebook"))
from inference import Inference


class SAM3DObjectsAPI(ls.LitAPI):
    """LitServe API for SAM 3D Objects inference."""

    def setup(self, device: str):
        """Initialize the inference pipeline."""
        # Load model
        config_path = os.path.join(
            os.path.dirname(__file__), "checkpoints/hf/pipeline.yaml"
        )
        self.inference = Inference(config_path, compile=False)
        print(f"Model loaded on {device}")

    def decode_request(self, request: dict) -> dict:
        """Decode incoming request with base64 image and masks."""
        # Decode image
        image_b64 = request.get("image")
        if not image_b64:
            raise ValueError("Missing 'image' field in request")

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image).astype(np.uint8)

        # Decode masks
        masks_b64 = request.get("masks", [])
        if not masks_b64:
            raise ValueError("Missing 'masks' field in request")

        masks = []
        for mask_b64 in masks_b64:
            mask_bytes = base64.b64decode(mask_b64)
            mask_img = Image.open(io.BytesIO(mask_bytes))
            mask = np.array(mask_img)
            # Convert to boolean mask
            mask = mask > 0
            if mask.ndim == 3:
                mask = mask[..., -1]
            masks.append(mask)

        seed = request.get("seed", 42)

        return {
            "image": image,
            "masks": masks,
            "seed": seed,
        }

    def predict(self, inputs: dict) -> dict:
        """Run inference on image with multiple masks."""
        image = inputs["image"]
        masks = inputs["masks"]
        seed = inputs["seed"]

        results = []

        for idx, mask in enumerate(masks):
            try:
                # Run inference for each mask
                output = self.inference(image, mask, seed=seed)

                result = {
                    "index": idx,
                    "rotation": output["rotation"].cpu().squeeze().tolist(),
                    "translation": output["translation"].cpu().squeeze().tolist(),
                    "scale": output["scale"].cpu().squeeze().tolist(),
                }

                # Export mesh to GLB if available
                if output.get("glb") is not None:
                    glb_buffer = io.BytesIO()
                    output["glb"].export(glb_buffer, file_type="glb")
                    glb_buffer.seek(0)
                    result["mesh_glb"] = base64.b64encode(glb_buffer.read()).decode()
                elif output.get("mesh") is not None and len(output["mesh"]) > 0:
                    # Export raw mesh data if GLB not available
                    mesh = output["mesh"][0]
                    result["mesh"] = {
                        "vertices": mesh.vertices.cpu().tolist(),
                        "faces": mesh.faces.cpu().tolist(),
                    }
                    if mesh.vertex_attrs is not None:
                        result["mesh"]["vertex_colors"] = mesh.vertex_attrs.cpu().tolist()

                results.append(result)

            except Exception as e:
                results.append({
                    "index": idx,
                    "error": str(e),
                })

        return {"objects": results, "success": True}

    def encode_response(self, output: dict) -> dict:
        """Encode response (already JSON-serializable)."""
        return output


def main():
    """Run the API server."""
    api = SAM3DObjectsAPI()
    server = ls.LitServer(
        api,
        accelerator="auto",
        max_batch_size=1,  # Process one request at a time (GPU memory)
        timeout=300,  # 5 minute timeout for inference
    )
    server.run(port=8000, generate_client_file=False)


if __name__ == "__main__":
    main()
