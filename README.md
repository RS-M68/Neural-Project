# Handwritten Recognition Bench (Digits + MNIST + EMNIST)

A clean, reproducible codebase that benchmarks classic ML (scikit-learn) and deep learning (Keras/TensorFlow)
models on standard handwritten datasets:
- **sklearn Digits** (8×8)
- **MNIST** (28×28)
- **EMNIST (Balanced / ByClass)** via TensorFlow Datasets

This repo is a cleaned implementation based on the project report you shared:
- Models: Multinomial Logistic Regression, Support Vector Machine, Decision Tree, Random Forest, Neural Networks, CNN, RNN/LSTM
- Datasets: sklearn Digits, Keras MNIST, “extra” EMNIST, plus a sample handwritten-words dataset (report scope)

## What this repo actually does (honest scope)
Your report title says “Handwritten Text to Digital Text Conversion”.
Most of the described experiments are **character-level classification** (digits/letters), not full-page text recognition with CTC.
So this repo focuses on what the report actually specifies: **classification benchmarks**, plus a small **image prediction CLI**.

If you want true word/line HTR (CNN+BiLSTM+CTC on IAM), that’s a different repo.

---

## Quickstart

### 1) Create venv + install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Run benchmarks
#### sklearn Digits (LogReg, SVM, DecisionTree, RandomForest)
```bash
python -m htr.scripts.train_digits --save-dir artifacts/digits
```

#### MNIST (MLP, CNN, RNN/LSTM)
```bash
python -m htr.scripts.train_mnist --model cnn --epochs 3 --save-dir artifacts/mnist
python -m htr.scripts.train_mnist --model mlp --epochs 3 --save-dir artifacts/mnist
python -m htr.scripts.train_mnist --model rnn --epochs 3 --save-dir artifacts/mnist
```

#### EMNIST (MLP, CNN, RNN/LSTM)
```bash
python -m htr.scripts.train_emnist --split balanced --model cnn --epochs 2 --save-dir artifacts/emnist
```

### 3) Predict a single image (PNG/JPG)
```bash
python -m htr.scripts.predict_image --model-path artifacts/mnist/cnn.keras --image path/to/image.png
```

---

## Project structure
```
handwritten-htr-bench/
  src/htr/
    data/              # dataset loading helpers
    models/            # sklearn + keras models
    scripts/           # CLI entrypoints
    utils/             # preprocessing, io, metrics
  artifacts/           # saved models (gitignored)
```

---

## Notes on reproducibility
- Seeds are set for Python/NumPy/TensorFlow where practical.
- Training defaults are kept small so the code runs on CPU.
- If you want accuracies closer to report numbers, increase epochs and tune params.

---

## License
MIT
