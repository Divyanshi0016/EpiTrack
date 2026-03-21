"""
sir_model.py
------------
Compartmental SIR / SEIR epidemic models using scipy ODE integration.
Fits beta (transmission rate) and gamma (recovery rate) to real data.

Usage
-----
    from src.models.sir_model import SIRModel
    model = SIRModel()
    model.fit(I_observed, N=population)
    preds = model.predict(days=60)
    print(f"R0 = {model.r0:.2f}")
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'plots')


# ── ODE systems ───────────────────────────────────────────────────────────────

def _sir_ode(y, t, beta, gamma, N):
    S, I, R = y
    dS = -beta * S * I / N
    dI =  beta * S * I / N - gamma * I
    dR =  gamma * I
    return [dS, dI, dR]


def _seir_ode(y, t, beta, gamma, sigma, N):
    S, E, I, R = y
    dS = -beta * S * I / N
    dE =  beta * S * I / N - sigma * E
    dI =  sigma * E - gamma * I
    dR =  gamma * I
    return [dS, dE, dI, dR]


# ── SIR ───────────────────────────────────────────────────────────────────────

class SIRModel:
    """
    Fit a SIR model to observed active-case time series.

    Parameters
    ----------
    beta_init, gamma_init : initial guesses for optimisation
    """
    def __init__(self, beta_init: float = 0.3, gamma_init: float = 0.1):
        self.beta_init  = beta_init
        self.gamma_init = gamma_init
        self.beta  = None
        self.gamma = None
        self.N     = None
        self._result = None

    @property
    def r0(self) -> float:
        if self.beta and self.gamma:
            return self.beta / self.gamma
        return None

    def _simulate(self, params, N, I0, days):
        beta, gamma = params
        S0 = N - I0
        R0 = 0
        t  = np.arange(days)
        sol = odeint(_sir_ode, [S0, I0, R0], t,
                     args=(beta, gamma, N))
        return sol[:, 1]   # I compartment

    def _loss(self, params, N, I_obs):
        I_pred = self._simulate(params, N, I_obs[0], len(I_obs))
        return np.sum((I_obs - I_pred) ** 2)

    def fit(self, I_observed: np.ndarray, N: int):
        """
        I_observed : daily active / new case counts (1-D array)
        N          : total population
        """
        self.N = N
        I = np.clip(I_observed.astype(float), 0, None)
        res = minimize(
            self._loss,
            x0=[self.beta_init, self.gamma_init],
            args=(N, I),
            method='L-BFGS-B',
            bounds=[(0.01, 5.0), (0.01, 2.0)],
            options={'maxiter': 2000},
        )
        self.beta, self.gamma = res.x
        self._result = res
        print(f"  SIR fit  β={self.beta:.4f}  γ={self.gamma:.4f}  R0={self.r0:.2f}")
        return self

    def predict(self, days: int) -> np.ndarray:
        assert self.beta is not None, "Call fit() first."
        I0  = 1   # placeholder; caller should set via I_observed[-1]
        return self._simulate([self.beta, self.gamma], self.N, I0, days)

    def predict_from(self, I_last: float, days: int) -> np.ndarray:
        return self._simulate([self.beta, self.gamma], self.N, I_last, days)

    def plot(self, I_observed: np.ndarray, country: str = ''):
        os.makedirs(PLOT_DIR, exist_ok=True)
        fitted = self._simulate([self.beta, self.gamma],
                                self.N, I_observed[0], len(I_observed))
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#050a0f')
        ax.set_facecolor('#0a1520')

        ax.plot(I_observed, color='#7a9e90', label='Observed', linewidth=1.4)
        ax.plot(fitted,     color='#00c896', label=f'SIR fit  R₀={self.r0:.2f}',
                linewidth=1.8, linestyle='--')
        ax.set_title(f'SIR Model — {country}', color='#d0e8e0', fontsize=13)
        ax.legend(facecolor='#0d1b2a', edgecolor='#1a3040', labelcolor='#7a9e90')
        ax.tick_params(colors='#3a5e50')
        ax.grid(True, color='#0f1e2e', linewidth=0.5)

        fname = f"SIR_{country.replace(' ','_')}.png"
        fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120,
                    bbox_inches='tight', facecolor='#050a0f')
        plt.close(fig)
        print(f"  SIR plot → outputs/plots/{fname}")


# ── SEIR ──────────────────────────────────────────────────────────────────────

class SEIRModel:
    """
    SEIR model: adds an Exposed (latent) compartment.
    sigma = 1 / incubation_period  (default: 1/5 for COVID-like)
    """
    def __init__(self, beta_init: float = 0.3, gamma_init: float = 0.1,
                 sigma: float = 1/5):
        self.beta_init  = beta_init
        self.gamma_init = gamma_init
        self.sigma      = sigma
        self.beta  = None
        self.gamma = None
        self.N     = None

    @property
    def r0(self):
        if self.beta and self.gamma:
            return self.beta / self.gamma
        return None

    def _simulate(self, params, N, I0, days):
        beta, gamma = params
        E0 = I0 * 2
        S0 = N - E0 - I0
        t  = np.arange(days)
        sol = odeint(_seir_ode, [S0, E0, I0, 0], t,
                     args=(beta, gamma, self.sigma, N))
        return sol[:, 2]   # I compartment

    def _loss(self, params, N, I_obs):
        I_pred = self._simulate(params, N, I_obs[0], len(I_obs))
        return np.sum((I_obs - I_pred) ** 2)

    def fit(self, I_observed: np.ndarray, N: int):
        self.N = N
        I = np.clip(I_observed.astype(float), 0, None)
        res = minimize(self._loss, [self.beta_init, self.gamma_init],
                       args=(N, I), method='L-BFGS-B',
                       bounds=[(0.01, 5.0), (0.01, 2.0)],
                       options={'maxiter': 2000})
        self.beta, self.gamma = res.x
        print(f"  SEIR fit  β={self.beta:.4f}  γ={self.gamma:.4f}  "
              f"σ={self.sigma:.4f}  R0={self.r0:.2f}")
        return self

    def predict_from(self, I_last: float, days: int) -> np.ndarray:
        return self._simulate([self.beta, self.gamma], self.N, I_last, days)
