"""
lstm_model.py
-------------
Sequence-to-point LSTM forecaster.
  Input:  sliding window of `seq_len` days of features
  Output: predicted new_cases_7d for the next day

Usage
-----
    from src.models.lstm_model import LSTMForecaster
    model = LSTMForecaster(seq_len=30)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    model.save('models/saved/lstm.keras')
"""

import numpy as np
import os


def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Convert a flat array into overlapping windows."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


class LSTMForecaster:
    def __init__(self, seq_len: int = 30, units: int = 64,
                 dropout: float = 0.2, epochs: int = 40,
                 batch_size: int = 64, patience: int = 8):
        self.seq_len = seq_len
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.model = None
        self._x_mean = None
        self._x_std = None

    # ── normalisation ────────────────────────────────────────────────────────

    def _fit_scaler(self, X: np.ndarray):
        self._x_mean = X.mean(axis=0)
        self._x_std  = X.std(axis=0) + 1e-8

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self._x_mean) / self._x_std

    # ── build ────────────────────────────────────────────────────────────────

    def _build(self, n_features: int):
        # Lazy import so TF only loads when needed
        import tensorflow as tf
        from tensorflow.keras import layers, models, callbacks  # noqa

        inp = layers.Input(shape=(self.seq_len, n_features))
        x   = layers.LSTM(self.units, return_sequences=True,
                           dropout=self.dropout)(inp)
        x   = layers.LSTM(self.units // 2, dropout=self.dropout)(x)
        x   = layers.Dense(32, activation='relu')(x)
        out = layers.Dense(1)(x)
        m   = models.Model(inp, out)
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='huber')
        return m

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_split: float = 0.1):
        import tensorflow as tf
        from tensorflow.keras import callbacks  # noqa

        self._fit_scaler(X)
        Xn = self._scale(X)
        yn = np.log1p(y)            # log-scale targets

        Xs, ys = _build_sequences(Xn, yn, self.seq_len)
        self.model = self._build(X.shape[1])

        cb = [
            callbacks.EarlyStopping(patience=self.patience,
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6),
        ]
        self.model.fit(Xs, ys,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       validation_split=validation_split,
                       callbacks=cb,
                       verbose=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = self._scale(X)
        Xs, _ = _build_sequences(Xn, np.zeros(len(Xn)), self.seq_len)
        raw = self.model.predict(Xs, verbose=0).flatten()
        return np.expm1(raw)         # inverse log1p

    def save(self, path: str):
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        meta = {'x_mean': self._x_mean, 'x_std': self._x_std,
                'seq_len': self.seq_len}
        joblib.dump(meta, path.replace('.keras', '_meta.pkl'))
        print(f"  LSTM saved → {path}")

    @classmethod
    def load(cls, path: str):
        import joblib, tensorflow as tf
        meta = joblib.load(path.replace('.keras', '_meta.pkl'))
        obj = cls(seq_len=meta['seq_len'])
        obj._x_mean = meta['x_mean']
        obj._x_std  = meta['x_std']
        obj.model   = tf.keras.models.load_model(path)
        return obj
