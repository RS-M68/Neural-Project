# Handwritten Text → Digital Text (Various Models)
Name : RAHUL SAGAR MYAKALA ID : 700735146 CRN : 13993

This repository contains code to reproduce the experiments described in the project *Handwritten Text to Digital Text Conversion using Various Deep Learning Models*. The project implements classical ML classifiers (Logistic Regression, SVM, Decision Tree, Random Forest) and Keras models (Dense NN, CNN, RNN/LSTM) on digit/character datasets (sklearn digits, Keras MNIST, EMNIST).


## Requirements
- Python 3.9+
- numpy>=1.22
-scikit-learn>=1.0
-tensorflow>=2.10
-tensorflow-datasets>=4.8
-matplotlib>=3.5
-pandas>=1.4


## Quick start
1. Create virtual environment and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   python experiments/train_sklearn_digits.py
   python experiments/train_keras_emnist.py

