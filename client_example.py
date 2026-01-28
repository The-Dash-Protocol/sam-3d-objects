# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Example client for SAM 3D Objects API Server

Usage:
    # Start server first: python serve.py
    # Then run client:
    python client_example.py
"""

import base64
import json
import requests
from pathlib import Path


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def encode_masks(mask_folder: str, indices: list[int] = None) -> list[str]:
    """Encode mask files to base64 strings."""
    mask_folder = Path(mask_folder)
    masks_b64 = []

    if indices is None:
        # Auto-detect masks (0.png, 1.png, ...)
        idx = 0
        while (mask_folder / f"{idx}.png").exists():
            indices = indices or []
            indices.append(idx)
            idx += 1

    for idx in indices:
        mask_path = mask_folder / f"{idx}.png"
        if mask_path.exists():
            masks_b64.append(encode_image(str(mask_path)))

    return masks_b64


def call_api(
    image_path: str,
    mask_folder: str,
    mask_indices: list[int] = None,
    seed: int = 42,
    server_url: str = "http://localhost:8000/predict",
) -> dict:
    """Call SAM 3D Objects API."""
    # Prepare request
    payload = {
        "image": encode_image(image_path),
        "masks": encode_masks(mask_folder, mask_indices),
        "seed": seed,
    }

    # Send request
    response = requests.post(
        server_url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=300,
    )
    response.raise_for_status()

    return response.json()


def save_mesh_from_response(response: dict, output_dir: str = "outputs"):
    """Save mesh files from API response."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for obj in response.get("objects", []):
        idx = obj.get("index", 0)

        # Save GLB if available
        if "mesh_glb" in obj:
            glb_path = output_dir / f"object_{idx}.glb"
            glb_bytes = base64.b64decode(obj["mesh_glb"])
            with open(glb_path, "wb") as f:
                f.write(glb_bytes)
            print(f"Saved: {glb_path}")

        # Save pose info as JSON
        pose_path = output_dir / f"object_{idx}_pose.json"
        pose_data = {
            "rotation": obj.get("rotation"),
            "translation": obj.get("translation"),
            "scale": obj.get("scale"),
        }
        with open(pose_path, "w") as f:
            json.dump(pose_data, f, indent=2)
        print(f"Saved: {pose_path}")


def main():
    """Example usage."""
    # Example paths (adjust to your data)
    image_path = "notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"
    mask_folder = "notebook/images/shutterstock_stylish_kidsroom_1640806567"

    # Call API with specific mask indices
    print("Calling SAM 3D Objects API...")
    response = call_api(
        image_path=image_path,
        mask_folder=mask_folder,
        mask_indices=[14, 15],  # Example: process masks 14 and 15
        seed=42,
    )

    # Print results
    print("\nAPI Response:")
    for obj in response.get("objects", []):
        idx = obj.get("index")
        if "error" in obj:
            print(f"  Object {idx}: Error - {obj['error']}")
        else:
            print(f"  Object {idx}:")
            print(f"    Rotation (quaternion): {obj.get('rotation')}")
            print(f"    Translation: {obj.get('translation')}")
            print(f"    Scale: {obj.get('scale')}")
            print(f"    Has mesh: {'mesh_glb' in obj or 'mesh' in obj}")

    # Save outputs
    save_mesh_from_response(response, output_dir="outputs")


if __name__ == "__main__":
    main()
