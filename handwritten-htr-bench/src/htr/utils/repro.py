from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 42
    deterministic_tf: bool = False


def seed_everything(cfg: SeedConfig = SeedConfig()) -> None:
    """Best-effort seeding for reproducibility."""
    os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # TensorFlow is optional at import-time for sklearn-only usage.
    try:
        import tensorflow as tf  # noqa: WPS433

        tf.random.set_seed(cfg.seed)
        if cfg.deterministic_tf:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except Exception:
        # Keep sklearn scripts working even if TF isn't installed.
        pass
