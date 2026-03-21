"""
xgboost_model.py
----------------
XGBoost regressor for epidemic forecasting with SHAP explainability.

Usage
-----
    from src.models.xgboost_model import XGBForecaster
    model = XGBForecaster()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    model.plot_shap(X_test, feature_names)
"""

import numpy as np
import pandas as pd
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'plots')


class XGBForecaster:
    def __init__(self, n_estimators: int = 500,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 early_stopping_rounds: int = 30):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method='hist',
            objective='reg:squarederror',
            random_state=42,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.feature_names = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: list = None,
            eval_set=None):
        import xgboost as xgb

        self.feature_names = feature_names
        y_log = np.log1p(np.clip(y, 0, None))

        self.model = xgb.XGBRegressor(
            **self.params,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric='rmse',
        )

        fit_kwargs = dict(verbose=50)
        if eval_set is not None:
            Xv, yv = eval_set
            fit_kwargs['eval_set'] = [(Xv, np.log1p(np.clip(yv, 0, None)))]

        self.model.fit(X, y_log, **fit_kwargs)
        print(f"  XGB best iteration: {self.model.best_iteration}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.expm1(self.model.predict(X))

    def get_feature_importance(self) -> pd.Series:
        imp = self.model.feature_importances_
        names = self.feature_names or [f'f{i}' for i in range(len(imp))]
        return pd.Series(imp, index=names).sort_values(ascending=False)

    def plot_feature_importance(self, top_n: int = 15):
        os.makedirs(PLOT_DIR, exist_ok=True)
        imp = self.get_feature_importance().head(top_n)

        fig, ax = plt.subplots(figsize=(8, top_n * 0.45))
        fig.patch.set_facecolor('#050a0f')
        ax.set_facecolor('#0a1520')

        bars = ax.barh(imp.index[::-1], imp.values[::-1], color='#00c896', height=0.6)
        ax.set_title('XGBoost Feature Importance', color='#d0e8e0', fontsize=13)
        ax.tick_params(colors='#7a9e90', labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor('#0f1e2e')
        ax.set_facecolor('#0a1520')
        ax.grid(axis='x', color='#0f1e2e', linewidth=0.5)

        out = os.path.join(PLOT_DIR, 'xgb_feature_importance.png')
        fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='#050a0f')
        plt.close(fig)
        print(f"  Saved → {out}")

    def plot_shap(self, X: np.ndarray, feature_names: list = None,
                  max_display: int = 15):
        """SHAP beeswarm plot — shows direction of each feature's effect."""
        import shap
        os.makedirs(PLOT_DIR, exist_ok=True)

        names = feature_names or self.feature_names or [f'f{i}' for i in range(X.shape[1])]
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X,
                          feature_names=names,
                          max_display=max_display,
                          show=False)
        out = os.path.join(PLOT_DIR, 'shap_summary.png')
        plt.savefig(out, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  SHAP plot saved → {out}")
        return shap_values

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'model': self.model,
                     'feature_names': self.feature_names}, path)
        print(f"  XGB saved → {path}")

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls()
        obj.model = data['model']
        obj.feature_names = data['feature_names']
        return obj
