from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SkModelName = Literal["logreg", "svm", "dtree", "rf"]


@dataclass(frozen=True)
class SkModelConfig:
    name: SkModelName
    random_state: int = 42


def build_sklearn_model(cfg: SkModelConfig):
    if cfg.name == "logreg":
        return LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            n_jobs=None,
        )
    if cfg.name == "svm":
        return SVC(C=1.1, kernel="rbf")  # matches report-style tuning
    if cfg.name == "dtree":
        return DecisionTreeClassifier(random_state=cfg.random_state)
    if cfg.name == "rf":
        return RandomForestClassifier(
            n_estimators=35,
            criterion="entropy",
            random_state=cfg.random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown sklearn model: {cfg.name}")
