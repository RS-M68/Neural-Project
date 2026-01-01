from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import tensorflow as tf


KerasModelName = Literal["mlp", "cnn", "rnn"]


@dataclass(frozen=True)
class KerasModelConfig:
    name: KerasModelName
    num_classes: int
    input_shape: tuple[int, int, int] = (28, 28, 1)
    learning_rate: float = 1e-3


def build_keras_model(cfg: KerasModelConfig) -> tf.keras.Model:
    if cfg.name == "mlp":
        return _mlp(cfg)
    if cfg.name == "cnn":
        return _cnn(cfg)
    if cfg.name == "rnn":
        return _rnn(cfg)
    raise ValueError(f"Unknown Keras model: {cfg.name}")


def compile_model(model: tf.keras.Model, lr: float = 1e-3) -> tf.keras.Model:
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _mlp(cfg: KerasModelConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=cfg.input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(cfg.num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="mlp")
    return compile_model(model, cfg.learning_rate)


def _cnn(cfg: KerasModelConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=cfg.input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(cfg.num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="cnn")
    return compile_model(model, cfg.learning_rate)


def _rnn(cfg: KerasModelConfig) -> tf.keras.Model:
    # Treat each row as a timestep -> (28 timesteps, 28 features)
    inputs = tf.keras.Input(shape=cfg.input_shape)
    x = tf.keras.layers.Reshape((cfg.input_shape[0], cfg.input_shape[1]))(inputs)
    x = tf.keras.layers.LSTM(128)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(cfg.num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="rnn_lstm")
    return compile_model(model, cfg.learning_rate)
