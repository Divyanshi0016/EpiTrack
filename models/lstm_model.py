"""
lstm_model.py
-------------
Sequence-to-point LSTM forecaster.
  Input:  sliding window of `seq_len` days of engineered features
  Output: predicted new_cases_7d for the next day

Usage
-----
    from src.models.lstm_model import LSTMForecaster

    model = LSTMForecaster(seq_len=14, units=64, epochs=30)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    model.save('models/saved/lstm_US.keras')

    # Reload later
    model = LSTMForecaster.load('models/saved/lstm_US.keras')

Fixes applied
-------------
  1. Safety check: raises clear error if not enough samples to build sequences
  2. Auto-reduce seq_len: if seq_len >= len(data), reduce automatically
  3. validation_split guard: sets to 0.0 if too few sequences for a split
  4. batch_size guard: clips batch_size to len(sequences) so fit() never crashes
  5. Empty-array guard in predict()
  6. Robust save/load using .keras format
"""

import numpy as np
import os


# ── Sequence builder ──────────────────────────────────────────────────────────

def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Convert flat arrays into overlapping sliding-window sequences.
    Returns empty arrays (not an error) if data is shorter than seq_len+1.
    """
    if len(X) <= seq_len:
        n_feat = X.shape[1] if X.ndim == 2 else 1
        return np.empty((0, seq_len, n_feat)), np.empty((0,))

    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


# ── Main class ────────────────────────────────────────────────────────────────

class LSTMForecaster:
    """
    Stacked LSTM with:
      - Z-score normalisation (fit on train, applied on test)
      - log1p target scaling to handle large case counts
      - Huber loss (robust to outliers)
      - EarlyStopping + ReduceLROnPlateau callbacks
      - All edge-case guards for small datasets
    """

    def __init__(
        self,
        seq_len:    int   = 14,    # reduced default from 30→14 for safety
        units:      int   = 64,
        dropout:    float = 0.2,
        epochs:     int   = 40,
        batch_size: int   = 32,
        patience:   int   = 8,
    ):
        self.seq_len    = seq_len
        self.units      = units
        self.dropout    = dropout
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.model      = None
        self._x_mean    = None
        self._x_std     = None
        self._fitted_seq_len = seq_len   # may be auto-reduced

    # ── Normalisation ─────────────────────────────────────────────────────────

    def _fit_scaler(self, X: np.ndarray):
        self._x_mean = X.mean(axis=0)
        self._x_std  = X.std(axis=0) + 1e-8   # avoid divide-by-zero

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self._x_mean is None:
            raise RuntimeError("Scaler not fitted — call fit() first.")
        return (X - self._x_mean) / self._x_std

    # ── Model builder ─────────────────────────────────────────────────────────

    def _build(self, seq_len: int, n_features: int):
        import tensorflow as tf
        from tensorflow.keras import layers, models

        inp = layers.Input(shape=(seq_len, n_features))
        x   = layers.LSTM(self.units, return_sequences=True,
                          dropout=self.dropout)(inp)
        x   = layers.LSTM(self.units // 2, dropout=self.dropout)(x)
        x   = layers.Dense(32, activation='relu')(x)
        out = layers.Dense(1)(x)

        m = models.Model(inp, out)
        m.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='huber'
        )
        return m

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_split: float = 0.1) -> None:
        """
        Train the LSTM on (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        validation_split : float
            Fraction of sequences held out for validation.
            Automatically set to 0.0 if there are too few sequences.
        """
        from tensorflow.keras import callbacks

        # ── Guard 1: auto-reduce seq_len if data is too short ────────────────
        effective_seq_len = self.seq_len
        if len(X) <= self.seq_len:
            effective_seq_len = max(3, len(X) // 2)
            print(
                f"  ⚠️  seq_len={self.seq_len} >= n_samples={len(X)}. "
                f"Auto-reducing seq_len to {effective_seq_len}."
            )
        self._fitted_seq_len = effective_seq_len

        # ── Normalise ─────────────────────────────────────────────────────────
        self._fit_scaler(X)
        Xn = self._scale(X)
        yn = np.log1p(np.clip(y, 0, None))   # log1p + clip negatives to 0

        # ── Build sequences ───────────────────────────────────────────────────
        Xs, ys = _build_sequences(Xn, yn, effective_seq_len)

        # ── Guard 2: check we have enough sequences at all ───────────────────
        if len(Xs) == 0:
            raise ValueError(
                f"\n"
                f"  ✗  Not enough data to build any training sequences.\n"
                f"  ✗  Rows available : {len(X)}\n"
                f"  ✗  seq_len used   : {effective_seq_len}\n"
                f"  ✗  Minimum needed : {effective_seq_len + 1}\n\n"
                f"  Fixes:\n"
                f"    1. Use a country with more data  →  COUNTRY = 'US'\n"
                f"    2. Reduce seq_len                →  LSTMForecaster(seq_len=7)\n"
            )

        # ── Guard 3: auto-disable validation_split if too few sequences ───────
        # Need at least 1 sample in each split
        min_for_split = max(10, int(np.ceil(1.0 / (1.0 - validation_split)) + 1))
        if len(Xs) < min_for_split:
            print(
                f"  ⚠️  Only {len(Xs)} sequences — "
                f"disabling validation_split (was {validation_split})."
            )
            validation_split = 0.0

        # ── Guard 4: clip batch_size to number of sequences ───────────────────
        effective_batch = min(self.batch_size, len(Xs))

        print(
            f"  Training LSTM  |  sequences={len(Xs)}  "
            f"seq_len={effective_seq_len}  "
            f"features={Xs.shape[2]}  "
            f"batch={effective_batch}  "
            f"val_split={validation_split}"
        )

        # ── Build model ───────────────────────────────────────────────────────
        self.model = self._build(effective_seq_len, Xs.shape[2])

        # ── Callbacks ─────────────────────────────────────────────────────────
        cb = [
            callbacks.EarlyStopping(
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=0
            ),
        ]

        # ── Train ─────────────────────────────────────────────────────────────
        self.model.fit(
            Xs, ys,
            epochs=self.epochs,
            batch_size=effective_batch,
            validation_split=validation_split,
            callbacks=cb,
            verbose=1,
        )

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for X.
        Returns array of length (len(X) - seq_len).
        Returns empty array if X is too short.
        """
        if self.model is None:
            raise RuntimeError("Model not trained — call fit() first.")
        if self._x_mean is None:
            raise RuntimeError("Scaler not fitted — call fit() first.")

        Xn = self._scale(X)
        Xs, _ = _build_sequences(Xn, np.zeros(len(Xn)), self._fitted_seq_len)

        # Guard: nothing to predict
        if len(Xs) == 0:
            print(
                f"  ⚠️  predict(): X has {len(X)} rows but "
                f"seq_len={self._fitted_seq_len}. Returning empty array."
            )
            return np.array([])

        raw = self.model.predict(Xs, verbose=0).flatten()
        return np.expm1(np.clip(raw, -10, 20))   # inverse log1p, clipped

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights (.keras) and scaler metadata (_meta.pkl)."""
        import joblib

        if self.model is None:
            raise RuntimeError("Nothing to save — model not trained yet.")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Save Keras model
        self.model.save(path)

        # Save scaler + config
        meta = {
            'x_mean':         self._x_mean,
            'x_std':          self._x_std,
            'seq_len':        self.seq_len,
            'fitted_seq_len': self._fitted_seq_len,
            'units':          self.units,
            'dropout':        self.dropout,
        }
        meta_path = path.replace('.keras', '_meta.pkl')
        joblib.dump(meta, meta_path)
        print(f"  ✅  LSTM saved  →  {path}")
        print(f"  ✅  Meta saved  →  {meta_path}")

    @classmethod
    def load(cls, path: str) -> 'LSTMForecaster':
        """Load a previously saved LSTMForecaster."""
        import joblib
        import tensorflow as tf

        meta_path = path.replace('.keras', '_meta.pkl')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        meta = joblib.load(meta_path)
        obj  = cls(
            seq_len  = meta['seq_len'],
            units    = meta.get('units', 64),
            dropout  = meta.get('dropout', 0.2),
        )
        obj._x_mean         = meta['x_mean']
        obj._x_std          = meta['x_std']
        obj._fitted_seq_len = meta.get('fitted_seq_len', meta['seq_len'])
        obj.model           = tf.keras.models.load_model(path)

        print(f"  ✅  LSTM loaded  ←  {path}")
        return obj


# ── Convenience function (matches usage in notebook 03) ──────────────────────

def train_lstm(series: np.ndarray,
               feature_matrix: np.ndarray,
               test_days: int = 30,
               seq_len: int = 14,
               epochs: int = 30,
               country: str = 'global') -> tuple:
    """
    High-level helper used directly in notebook 03.

    Parameters
    ----------
    series         : 1-D smoothed case counts (used as target y)
    feature_matrix : 2-D array of engineered features (used as X)
    test_days      : number of days held out for testing
    seq_len        : LSTM input window length
    epochs         : max training epochs

    Returns
    -------
    model, preds, y_test, dates_test
    """
    import pandas as pd

    n = len(feature_matrix)
    split = n - test_days

    X_train = feature_matrix[:split]
    y_train = series[:split]
    X_test  = feature_matrix[split:]
    y_test  = series[split:]

    model = LSTMForecaster(seq_len=seq_len, units=64, epochs=epochs)
    model.fit(X_train, y_train, validation_split=0.1)

    preds = model.predict(X_test)

    # Save
    save_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'models', 'saved'
    )
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f'lstm_{country}.keras'))

    return model, preds, y_test, X_test
