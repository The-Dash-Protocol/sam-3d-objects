# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
SAM 3D Objects API Server using LitServe

Usage:
    cd /path/to/sam-3d-objects
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

# Get project root directory (where this script is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add notebook to path for inference imports
sys.path.append(os.path.join(PROJECT_ROOT, "notebook"))
from inference import Inference


class SAM3DObjectsAPI(ls.LitAPI):
    """LitServe API for SAM 3D Objects inference."""

    def __init__(self):
        """Initialize with max_batch_size for new LitServe API."""
        super().__init__()
        self.max_batch_size = 1  # Process one request at a time (GPU memory)

    def setup(self, device: str):
        """Initialize the inference pipeline."""
        # Load model - config path relative to project root
        config_path = os.path.join(PROJECT_ROOT, "checkpoints/hf/pipeline.yaml")
        self.inference = Inference(config_path, compile=False)
        print(f"Model loaded on {device}")

    def _decode_base64_image(self, b64_str: str) -> bytes:
        """Decode base64 string, handling data URL prefix if present."""
        # Handle data URL format: "data:image/png;base64,iVBOR..."
        if b64_str.startswith("data:"):
            # Extract base64 part after the comma
            b64_str = b64_str.split(",", 1)[1]
        return base64.b64decode(b64_str)

    def _decode_gemini_mask(self, mask_data: dict, image_size: tuple) -> np.ndarray:
        """Decode Gemini-style mask with bounding box.

        Args:
            mask_data: dict with "mask" (base64 PNG) and "box_2d" ([y0, x0, y1, x1] in 0-1000)
            image_size: (height, width) of the original image
        """
        img_h, img_w = image_size

        # Decode mask PNG
        mask_bytes = self._decode_base64_image(mask_data["mask"])
        mask_img = Image.open(io.BytesIO(mask_bytes))
        mask_crop = np.array(mask_img)

        # Get bounding box (Gemini format: [y0, x0, y1, x1] normalized to 0-1000)
        box = mask_data["box_2d"]
        y0 = int(box[0] / 1000 * img_h)
        x0 = int(box[1] / 1000 * img_w)
        y1 = int(box[2] / 1000 * img_h)
        x1 = int(box[3] / 1000 * img_w)

        # Resize mask to bounding box size
        box_h, box_w = y1 - y0, x1 - x0
        mask_resized = np.array(Image.fromarray(mask_crop).resize((box_w, box_h)))

        # Create full-size mask and place the crop
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = mask_resized if mask_resized.ndim == 2 else mask_resized[..., 0]

        return full_mask > 127  # Binarize at midpoint

    def decode_request(self, request: dict) -> dict:
        """Decode incoming request with base64 image and masks.

        Supports two mask formats:
        1. Simple base64 PNG: ["base64...", "base64..."]
        2. Gemini format: [{"mask": "data:image/png;base64,...", "box_2d": [y0,x0,y1,x1]}, ...]
        """
        # Decode image
        image_b64 = request.get("image")
        if not image_b64:
            raise ValueError("Missing 'image' field in request")

        image_bytes = self._decode_base64_image(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image).astype(np.uint8)
        img_h, img_w = image.shape[:2]

        # Decode masks
        masks_input = request.get("masks", [])
        if not masks_input:
            raise ValueError("Missing 'masks' field in request")

        masks = []
        for mask_item in masks_input:
            if isinstance(mask_item, dict):
                # Gemini format: {"mask": "...", "box_2d": [...]}
                mask = self._decode_gemini_mask(mask_item, (img_h, img_w))
            else:
                # Simple base64 PNG string
                mask_bytes = self._decode_base64_image(mask_item)
                mask_img = Image.open(io.BytesIO(mask_bytes))
                mask = np.array(mask_img)
                # Convert to boolean mask
                mask = mask > 127
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
        timeout=300,  # 5 minute timeout for inference
    )
    server.run(port=8000, generate_client_file=False)


if __name__ == "__main__":
    main()
