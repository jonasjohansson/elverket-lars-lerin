"""Generate depth maps from images using Depth Anything V2."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
from transformers import pipeline

def generate_depth_map(image_path: str, output_path: str = None):
    """Generate a depth map from an image and save as grayscale PNG."""
    image_path = Path(image_path)
    if output_path is None:
        output_path = image_path.with_stem(image_path.stem + "_depth").with_suffix(".png")
    else:
        output_path = Path(output_path)

    print(f"Loading model (first run downloads ~400MB)...")
    pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device="mps",  # Apple Silicon GPU
    )

    print(f"Processing {image_path}...")
    image = Image.open(image_path).convert("RGB")
    result = pipe(image)

    depth = result["depth"]  # PIL Image
    # Normalize to full 0-255 range
    depth_arr = np.array(depth, dtype=np.float32)
    depth_arr = (depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-8) * 255
    depth_img = Image.fromarray(depth_arr.astype(np.uint8))
    depth_img = depth_img.resize(image.size, Image.LANCZOS)

    depth_img.save(output_path)
    print(f"Depth map saved to {output_path} ({depth_img.size[0]}x{depth_img.size[1]})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_depth.py <image_path> [output_path]")
        sys.exit(1)

    img = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    generate_depth_map(img, out)
