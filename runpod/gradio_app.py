#!/usr/bin/env python3
"""
SAM 3D Objects - Gradio Web Interface

An interactive web interface for 3D object reconstruction.
"""

import os
import sys
import tempfile
import traceback

# Set environment before imports
os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME", "/usr/local/cuda")
os.environ["LIDRA_SKIP_INIT"] = "true"

sys.path.insert(0, "/app/notebook")
sys.path.insert(0, "/app")

import gradio as gr
import numpy as np
from PIL import Image

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


def process_image(image, mask_image, seed):
    """
    Process image and mask to generate 3D reconstruction.

    Args:
        image: Input image (numpy array)
        mask_image: Mask image (numpy array, optional)
        seed: Random seed

    Returns:
        Path to PLY file, status message
    """
    try:
        if image is None:
            return None, "Please upload an image first."

        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Handle mask
        if mask_image is not None:
            if isinstance(mask_image, Image.Image):
                mask_image = np.array(mask_image)
            mask = mask_image > 0
            if mask.ndim == 3:
                mask = mask[..., -1]
        else:
            # Full image mask
            mask = np.ones(image.shape[:2], dtype=bool)

        # Parse seed
        if seed is None or seed == "" or seed == 0:
            seed = None
        else:
            seed = int(seed)

        # Get model
        inference = get_inference_model()

        # Run inference
        print(f"Running inference with seed={seed}")
        output = inference(image, mask, seed=seed)

        # Save PLY file
        output_dir = tempfile.mkdtemp()
        ply_path = os.path.join(output_dir, "reconstruction.ply")
        output["gs"].save_ply(ply_path)

        return ply_path, f"Reconstruction complete! Seed: {seed if seed else 'random'}"

    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {str(e)}"


def create_interface():
    """Create Gradio interface."""
    with gr.Blocks(
        title="SAM 3D Objects",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # SAM 3D Objects - 3D Reconstruction from Single Image

        Upload an image and optionally a mask to reconstruct the 3D object.
        The mask should be white (255) for the object area and black (0) for background.

        **Note:** This model requires significant GPU memory (~32GB VRAM).
        """)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Input Image",
                    type="numpy",
                    height=400
                )
                mask_input = gr.Image(
                    label="Mask (Optional - white=object, black=background)",
                    type="numpy",
                    height=400
                )
                seed_input = gr.Number(
                    label="Seed (0 or empty for random)",
                    value=42,
                    precision=0
                )
                submit_btn = gr.Button("Generate 3D", variant="primary")

            with gr.Column():
                output_file = gr.File(
                    label="Download PLY File"
                )
                output_viewer = gr.Model3D(
                    label="3D Preview",
                    height=500
                )
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )

        # Connect events
        submit_btn.click(
            fn=process_image,
            inputs=[image_input, mask_input, seed_input],
            outputs=[output_viewer, status_output]
        ).then(
            fn=lambda x: x,
            inputs=[output_viewer],
            outputs=[output_file]
        )

        # Example images
        gr.Markdown("""
        ## Tips
        - For best results, use images with clear objects on neutral backgrounds
        - The mask helps the model focus on specific objects in complex scenes
        - Processing may take 30-60 seconds depending on GPU
        """)

    return demo


if __name__ == "__main__":
    # Pre-load model
    print("Pre-loading model...")
    try:
        get_inference_model()
        print("Model ready!")
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")

    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    )
