from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImagePreprocessConfig:
    size: tuple[int, int] = (28, 28)
    invert: bool = True
    normalize: bool = True


def load_image_as_tensor(path: str | Path, cfg: ImagePreprocessConfig = ImagePreprocessConfig()) -> np.ndarray:
    """Load an image file and convert to a (1, H, W, 1) float tensor for Keras models."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    img = Image.open(p).convert("L")
    img = ImageOps.fit(img, cfg.size, method=Image.BILINEAR)

    if cfg.invert:
        img = ImageOps.invert(img)

    arr = np.array(img, dtype=np.float32)

    if cfg.normalize:
        arr /= 255.0

    # Shape: (1, H, W, 1)
    return arr[None, :, :, None]
