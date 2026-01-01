from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DigitsSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_digits_split(test_size: float = 0.2, random_state: int = 42) -> DigitsSplit:
    data = load_digits()
    x = data.data.astype(np.float32)  # (n, 64)
    y = data.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return DigitsSplit(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
