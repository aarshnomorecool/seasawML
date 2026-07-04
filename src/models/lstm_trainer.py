"""
SeaSaw LSTM Trainer
====================
Phase 5 (LSTM half). Trains one direct-output LSTM per horizon on the flat
tabular datasets produced by Phase 4 (see CLAUDE.md "LSTM Strategy" — direct
single-output, no recursive prediction, no seq2seq).

Sequence construction
---------------------
Phase 4 saves flat (n_samples, n_features) arrays per split. The LSTM needs
(n_sequences, sequence_length, n_features) windows, so `build_sequences()`
turns each split into overlapping windows of the last `sequence_length`
rows, paired with the *last* row's already-horizon-shifted target. This
means the first (sequence_length - 1) rows of each split can't form a full
window and are dropped — a one-time edge effect at each split boundary,
not a leak (train/val/test remain from Phase 4's chronological split).

Architecture: LSTM(128) -> Dropout(0.2) -> LSTM(64) -> Dense(32) -> Dense(1)
Loss: MSE on the (already log-transformed) target.

Usage
-----
    from src.models.lstm_trainer import LSTMTrainer

    trainer = LSTMTrainer(horizon="A", sequence_length=288)
    trainer.train(X_train, y_train, X_val, y_val)
    trainer.save("models/lstm_horizon_A.h5")
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


def build_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn flat (n, features) X and (n,) y into sliding windows of length
    sequence_length: X_seq[i] = X[i : i+sequence_length], y_seq[i] = y at the
    last row of that window (y is already the horizon-shifted target from
    Phase 4, so no further shifting happens here).
    """
    if len(X) < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} rows to build one sequence, got {len(X)}"
        )

    windows = sliding_window_view(X, sequence_length, axis=0)  # (n-L+1, features, L)
    windows = np.moveaxis(windows, -1, 1)  # -> (n-L+1, L, features)
    y_seq = y[sequence_length - 1:]
    return windows.astype(np.float32), y_seq.astype(np.float32)


class LSTMTrainer:
    """
    Parameters
    ----------
    horizon         : label used only for logging (e.g. "A", "B", "C")
    sequence_length : number of past timesteps per input window (default 288 = 24h at 5-min res)
    """

    def __init__(self, horizon: str, sequence_length: int = 288):
        self.horizon = horizon
        self.sequence_length = sequence_length
        self.model: keras.Model = None
        self.history = None

    def _build_model(self, n_features: int) -> keras.Model:
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, n_features)),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(64),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        patience: int = 5,
    ) -> keras.callbacks.History:
        X_train_seq, y_train_seq = build_sequences(X_train, y_train, self.sequence_length)
        X_val_seq, y_val_seq = build_sequences(X_val, y_val, self.sequence_length)

        logger.info(
            f"[{self.horizon}] sequences: train={X_train_seq.shape} val={X_val_seq.shape}"
        )

        self.model = self._build_model(n_features=X_train.shape[1])

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )

        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=2,
        )
        return self.history

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"Saved LSTM model -> {path}")

    @staticmethod
    def load(path: Union[str, Path], recompile: bool = False) -> keras.Model:
        # compile=False avoids a Keras 3 legacy-H5 deserialization bug where
        # the saved "mse" loss/metric config fails to reconstruct on load
        # (ValueError: Could not deserialize 'keras.metrics.mse' ...).
        # predict() works fine uncompiled; pass recompile=True if you need
        # .evaluate() or further training.
        model = keras.models.load_model(path, compile=False)
        if recompile:
            model.compile(optimizer="adam", loss="mse")
        return model
