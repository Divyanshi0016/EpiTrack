"""
evaluation.py
-------------
Unified evaluation utilities for all three models.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'plots')


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """Mean Absolute Percentage Error — adds epsilon to avoid div/0."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-9))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str = '') -> dict:
    """Return a dict of all metrics."""
    metrics = {
        'model': name,
        'RMSE':  rmse(y_true, y_pred),
        'MAE':   mae(y_true, y_pred),
        'MAPE':  mape(y_true, y_pred),
        'R2':    r2(y_true, y_pred),
    }
    print(f"  [{name}]  RMSE={metrics['RMSE']:.1f}  "
          f"MAE={metrics['MAE']:.1f}  MAPE={metrics['MAPE']:.1f}%  "
          f"R²={metrics['R2']:.4f}")
    return metrics


def plot_forecast(dates, y_true, y_pred,
                  country: str, model_name: str,
                  conf_lower=None, conf_upper=None):
    """Save actual vs predicted line chart to outputs/plots/."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#050a0f')
    ax.set_facecolor('#0a1520')

    ax.plot(dates, y_true, color='#7a9e90', linewidth=1.2, label='Actual', alpha=0.8)
    ax.plot(dates, y_pred, color='#00c896', linewidth=1.8, label=f'{model_name} forecast')

    if conf_lower is not None and conf_upper is not None:
        ax.fill_between(dates, conf_lower, conf_upper,
                        color='#00c896', alpha=0.12, label='90% CI')

    ax.set_title(f'{country} — {model_name}', color='#d0e8e0', fontsize=13, pad=10)
    ax.set_xlabel('Date', color='#3a5e50')
    ax.set_ylabel('Daily New Cases (7d avg)', color='#3a5e50')
    ax.tick_params(colors='#3a5e50')
    for spine in ax.spines.values():
        spine.set_edgecolor('#0f1e2e')
    ax.legend(facecolor='#0d1b2a', edgecolor='#1a3040', labelcolor='#7a9e90')
    ax.grid(True, color='#0f1e2e', linewidth=0.5)

    fname = f"{country.replace(' ','_')}_{model_name}.png"
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120,
                bbox_inches='tight', facecolor='#050a0f')
    plt.close(fig)
    print(f"  Saved plot → outputs/plots/{fname}")


def compare_models(results: list) -> pd.DataFrame:
    """
    results: list of dicts from evaluate().
    Returns a sorted DataFrame and prints a summary table.
    """
    df = pd.DataFrame(results).set_index('model')
    df = df.sort_values('RMSE')
    print("\n── Model Comparison ──────────────────────────────")
    print(df.to_string(float_format='{:.2f}'.format))
    print("──────────────────────────────────────────────────")
    return df
