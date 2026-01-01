from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class EmnistSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    label_names: list[str]


EmnistSplitName = Literal["balanced", "byclass"]


def _load_tfds(name: str):
    import tensorflow_datasets as tfds  # noqa: WPS433
    return tfds.load(name, as_supervised=True, with_info=True)


def load_emnist(split: EmnistSplitName = "balanced", max_train: int | None = None, max_test: int | None = None) -> EmnistSplit:
    """EMNIST from TFDS.

    `balanced` is a good default. `byclass` is larger and slower.
    Returns float32 images normalized to [0,1] with shape (n, 28, 28, 1).
    """
    ds_name = f"emnist/{split}"
    (ds_train, ds_test), info = _load_tfds(ds_name)
    label_names = list(getattr(info.features["label"], "names", []))

    def to_numpy(ds, max_items: int | None):
        xs, ys = [], []
        n = 0
        for x, y in ds:
            xs.append(x.numpy())
            ys.append(y.numpy())
            n += 1
            if max_items is not None and n >= max_items:
                break
        x_arr = np.stack(xs).astype("float32") / 255.0
        y_arr = np.array(ys).astype("int64")
        # TFDS images are (28,28,1) already for EMNIST
        if x_arr.ndim == 3:
            x_arr = x_arr[..., None]
        return x_arr, y_arr

    x_train, y_train = to_numpy(ds_train, max_train)
    x_test, y_test = to_numpy(ds_test, max_test)

    return EmnistSplit(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, label_names=label_names)
