from __future__ import annotations

import argparse
from pathlib import Path
import pickle

from rich.console import Console

from htr.data.digits import load_digits_split
from htr.models.sklearn_models import SkModelConfig, build_sklearn_model
from htr.utils.metrics import accuracy
from htr.utils.repro import seed_everything

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sklearn models on sklearn Digits dataset.")
    parser.add_argument("--models", nargs="+", default=["logreg", "svm", "dtree", "rf"], help="Subset of models to run.")
    parser.add_argument("--save-dir", default="artifacts/digits", help="Where to store trained models.")
    args = parser.parse_args()

    seed_everything()
    split = load_digits_split()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        cfg = SkModelConfig(name=name)  # type: ignore[arg-type]
        model = build_sklearn_model(cfg)
        model.fit(split.x_train, split.y_train)
        preds = model.predict(split.x_test)
        acc = accuracy(split.y_test, preds)
        console.print(f"[bold]{name}[/bold] accuracy: {acc:.4f} (n={len(split.y_test)})")

        with open(save_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    console.print(f"Saved models to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
