from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MnistSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_mnist() -> MnistSplit:
    """MNIST via tf.keras.datasets. Returns float32 images normalized to [0,1]."""
    import tensorflow as tf  # noqa: WPS433

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dim -> (n, 28, 28, 1)
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    return MnistSplit(x_train=x_train, x_test=x_test, y_train=y_train.astype("int64"), y_test=y_test.astype("int64"))
