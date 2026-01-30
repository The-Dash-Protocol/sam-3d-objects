#!/usr/bin/env python3
"""
SAM 3D Objects - RunPod Client Example

This script demonstrates how to call the SAM 3D Objects API
deployed on RunPod (both Serverless and GPU Pod).
"""

import os
import sys
import base64
import argparse
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_base64_to_file(base64_string: str, output_path: str) -> None:
    """Decode base64 string and save to file."""
    data = base64.b64decode(base64_string)
    with open(output_path, "wb") as f:
        f.write(data)


def call_serverless_api(
    endpoint_id: str,
    api_key: str,
    image_path: str,
    mask_path: str = None,
    seed: int = None,
    output_format: str = "ply",
    timeout: int = 300,
) -> dict:
    """
    Call RunPod Serverless API.

    Args:
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        image_path: Path to input image
        mask_path: Path to mask image (optional)
        seed: Random seed (optional)
        output_format: Output format (ply, glb, or both)
        timeout: Request timeout in seconds

    Returns:
        API response dictionary
    """
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Prepare input
    payload = {
        "input": {
            "image": encode_image_to_base64(image_path),
            "output_format": output_format,
        }
    }

    if mask_path:
        payload["input"]["mask"] = encode_image_to_base64(mask_path)

    if seed is not None:
        payload["input"]["seed"] = seed

    print(f"Calling RunPod Serverless API...")
    print(f"Endpoint: {endpoint_id}")

    start_time = time.time()
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    elapsed = time.time() - start_time

    print(f"Response received in {elapsed:.2f} seconds")

    if response.status_code != 200:
        raise RuntimeError(f"API error: {response.status_code} - {response.text}")

    return response.json()


def call_direct_api(
    api_url: str,
    image_path: str,
    mask_path: str = None,
    seed: int = None,
    output_format: str = "ply",
    timeout: int = 300,
) -> dict:
    """
    Call direct API server (GPU Pod deployment).

    Args:
        api_url: API server URL (e.g., http://pod-ip:8000)
        image_path: Path to input image
        mask_path: Path to mask image (optional)
        seed: Random seed (optional)
        output_format: Output format (ply, glb, or both)
        timeout: Request timeout in seconds

    Returns:
        API response dictionary
    """
    url = f"{api_url.rstrip('/')}/predict"

    payload = {
        "image": encode_image_to_base64(image_path),
        "output_format": output_format,
    }

    if mask_path:
        payload["mask"] = encode_image_to_base64(mask_path)

    if seed is not None:
        payload["seed"] = seed

    print(f"Calling API at {url}...")

    start_time = time.time()
    response = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - start_time

    print(f"Response received in {elapsed:.2f} seconds")

    if response.status_code != 200:
        raise RuntimeError(f"API error: {response.status_code} - {response.text}")

    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3D Objects - RunPod API Client"
    )
    parser.add_argument(
        "image",
        help="Path to input image"
    )
    parser.add_argument(
        "-m", "--mask",
        help="Path to mask image (optional)"
    )
    parser.add_argument(
        "-o", "--output",
        default="output.ply",
        help="Output file path (default: output.ply)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        help="Random seed"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["ply", "glb", "both"],
        default="ply",
        help="Output format (default: ply)"
    )

    # API connection options
    api_group = parser.add_mutually_exclusive_group(required=True)
    api_group.add_argument(
        "--serverless",
        metavar="ENDPOINT_ID",
        help="RunPod Serverless endpoint ID"
    )
    api_group.add_argument(
        "--api-url",
        help="Direct API server URL (e.g., http://pod-ip:8000)"
    )

    parser.add_argument(
        "--api-key",
        help="RunPod API key (or set RUNPOD_API_KEY env var)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    if args.mask and not os.path.exists(args.mask):
        print(f"Error: Mask file not found: {args.mask}")
        sys.exit(1)

    # Get API key if using serverless
    api_key = args.api_key or os.environ.get("RUNPOD_API_KEY")
    if args.serverless and not api_key:
        print("Error: RunPod API key required. Set --api-key or RUNPOD_API_KEY env var")
        sys.exit(1)

    try:
        # Call API
        if args.serverless:
            result = call_serverless_api(
                endpoint_id=args.serverless,
                api_key=api_key,
                image_path=args.image,
                mask_path=args.mask,
                seed=args.seed,
                output_format=args.format,
                timeout=args.timeout,
            )
            # Handle serverless response format
            if "output" in result:
                result = result["output"]
        else:
            result = call_direct_api(
                api_url=args.api_url,
                image_path=args.image,
                mask_path=args.mask,
                seed=args.seed,
                output_format=args.format,
                timeout=args.timeout,
            )

        # Check for errors
        if "error" in result:
            print(f"Error from API: {result['error']}")
            sys.exit(1)

        # Save outputs
        output_path = Path(args.output)
        output_dir = output_path.parent

        if "ply" in result:
            ply_path = output_dir / f"{output_path.stem}.ply"
            decode_base64_to_file(result["ply"], str(ply_path))
            print(f"PLY saved to: {ply_path}")

        if "glb" in result:
            glb_path = output_dir / f"{output_path.stem}.glb"
            decode_base64_to_file(result["glb"], str(glb_path))
            print(f"GLB saved to: {glb_path}")

        if "glb_error" in result:
            print(f"GLB export note: {result['glb_error']}")

        # Print metadata
        print("\nReconstruction metadata:")
        if "rotation" in result:
            print(f"  Rotation: {result['rotation']}")
        if "translation" in result:
            print(f"  Translation: {result['translation']}")
        if "scale" in result:
            print(f"  Scale: {result['scale']}")

        print("\nDone!")

    except requests.exceptions.Timeout:
        print(f"Error: Request timed out after {args.timeout} seconds")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
