from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rich.console import Console

from htr.utils.image_io import load_image_as_tensor
from htr.utils.repro import seed_everything

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a single handwritten image using a saved Keras model.")
    parser.add_argument("--model-path", required=True, help="Path to .keras model file.")
    parser.add_argument("--image", required=True, help="Path to image (png/jpg).")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    seed_everything()

    import tensorflow as tf  # noqa: WPS433

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    x = load_image_as_tensor(args.image)
    probs = model.predict(x, verbose=0)[0]
    topk = int(args.topk)

    idx = np.argsort(probs)[::-1][:topk]
    console.print(f"Image: {args.image}")
    console.print(f"Model: {args.model_path}")
    for rank, i in enumerate(idx, start=1):
        console.print(f"#{rank}: class={int(i)} prob={float(probs[i]):.4f}")


if __name__ == "__main__":
    main()
