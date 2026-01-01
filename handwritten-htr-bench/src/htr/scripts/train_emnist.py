from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from htr.data.emnist import load_emnist
from htr.models.keras_models import KerasModelConfig, build_keras_model
from htr.utils.repro import seed_everything

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Keras models on EMNIST via TFDS.")
    parser.add_argument("--split", choices=["balanced", "byclass"], default="balanced")
    parser.add_argument("--model", choices=["mlp", "cnn", "rnn"], default="cnn")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train", type=int, default=20000, help="Limit training examples for CPU runs.")
    parser.add_argument("--max-test", type=int, default=5000, help="Limit test examples for CPU runs.")
    parser.add_argument("--save-dir", default="artifacts/emnist")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    seed_everything()
    data = load_emnist(split=args.split, max_train=args.max_train, max_test=args.max_test)

    num_classes = int(data.y_train.max() + 1)
    console.print(f"Loaded EMNIST/{args.split}: train={len(data.y_train)} test={len(data.y_test)} classes={num_classes}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = KerasModelConfig(name=args.model, num_classes=num_classes, learning_rate=args.lr)
    model = build_keras_model(cfg)

    model.fit(
        data.x_train,
        data.y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    loss, acc = model.evaluate(data.x_test, data.y_test, verbose=0)
    console.print(f"[bold]{args.model}[/bold] test accuracy: {acc:.4f} | loss: {loss:.4f}")

    out_path = save_dir / f"{args.split}_{args.model}.keras"
    model.save(out_path)
    console.print(f"Saved model to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
