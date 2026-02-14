from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cognitive_oddballs.models.model import Model

Number = int | float | np.number


def exp_clip(x: Number, clip: float = 60.0) -> float:
    """Exponentiation with clipping to avoid overflow."""
    return float(np.exp(np.clip(float(x), -clip, clip)))


@dataclass
class HGF2Config:
    # Core HGF parameters
    # world state
    mu1_0: float
    sig1_0: float

    # volatility of the world
    mu2_0: float
    sig2_0: float

    # model parameters
    eta: float  # volatility of random walk variance
    s: float  # observation noise variance

    # Numerical stability
    min_var: float = 1e-8
    exp_clip_value: float = 30.0

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.eta <= 0:
            raise ValueError("eta must be > 0")
        if self.s <= 0:
            raise ValueError("s (observation variance) must be > 0")
        if self.sig1_0 < self.min_var:
            raise ValueError(f"sig1_0 must be >= {self.min_var}")
        if self.sig2_0 < self.min_var:
            raise ValueError(f"sig2_0 must be >= {self.min_var}")


class HGFPaper2Gaussian(Model):
    """
    2-level Gaussian HGF as in Markovic & Kiebel (2016):
    x2_t = x2_{t-1} + sqrt(eta) * n
    x1_t = x1_{t-1} + exp(x2_t/2) * n   -> Var = exp(x2_t)
    o_t  = x1_t + sqrt(s) * n
    """

    def __init__(
        self,
        eta: float = 0.005,
        s: float = 15.0**2,
        mu1_init: float = 0.0,
        sig1_init: float = 10.0,
        mu2_init: float = -4.0,
        sig2_init: float = 1.0,
        min_var: float = 1e-8,
        exp_clip_value: float = 30.0,
    ):
        self.cfg = HGF2Config(
            mu1_0=float(mu1_init),
            sig1_0=float(sig1_init),
            mu2_0=float(mu2_init),
            sig2_0=float(sig2_init),
            eta=float(eta),
            s=float(s),
            min_var=float(min_var),
            exp_clip_value=float(exp_clip_value),
        )

        # current posteriors -- basically a reset_state function
        self.mu1 = self.cfg.mu1_0
        self.sig1 = max(self.cfg.sig1_0, self.cfg.min_var)
        self.mu2 = self.cfg.mu2_0
        self.sig2 = max(self.cfg.sig2_0, self.cfg.min_var)

        # Initialize history as DataFrame
        self.history: pd.DataFrame = pd.DataFrame(
            columns=[
                "o",
                "mu1_hat",  # prediction before update -> belief about x1_t before seeing o_t
                "sig1_hat",  # uncertainty about x1_t before seeing o_t
                "mu1",  # posterior mean level 1 after update -> belief about x1_t
                "sig1",  # posterior variance level 1 after update -> uncertainty about x1_t
                "mu2_hat",  # prediction before update -> belief about x2_t before seeing o_t
                "sig2_hat",  # uncertainty about x2_t before seeing o_t
                "mu2",  # posterior mean level 2 after update -> belief about x2_t
                "sig2",  # posterior variance level 2 after update -> uncertainty about x2_t
                "omega",  # exp(mu2_hat) - predicted volatility
                "delta1",  # prediction error level 1
                "alpha1",  # learning rate level 1
                "delta2",  # prediction error level 2
                "k","r",    # helpers for PE
                "vfe",  # variational free energy: calculated after update
            ]
        )
    
    def update(self, o: Number) -> None:
        o = float(o)
        minvar = self.cfg.min_var

        # ----------- Prediction -----------------
        # Level 2
        mu2_prev = self.mu2
        sig2_prev_post = self.sig2
        sig2_hat = max(sig2_prev_post + self.cfg.eta, minvar)

        # Level 1
        omega = exp_clip(mu2_prev, self.cfg.exp_clip_value)
        mu1_prev = self.mu1
        sig1_prev = self.sig1
        den1 = max(sig1_prev + omega, minvar)

        # ----------- Update -------------------
        # ----------- Level 1 ------------------
        # prediction error with observation
        delta1 = o - mu1_prev
        # sigma^{(1)}_t via precision sum (paper)
        # 1/sig1_t = 1/s + 1/(sig1_{t-1} + omega)
        sig1_new = 1.0 / max((1.0 / self.cfg.s) + (1.0 / den1), minvar)
        # learning rate
        alpha1 = sig1_new / max(self.cfg.s, minvar)
        # posterior update
        eps1 = alpha1 * delta1
        mu1_new = mu1_prev + eps1

        # ------------ Level 2 ---------------
        # prediction error on volatility
        delta2 = (sig1_new + eps1 * eps1) / den1 - 1.0

        # helper coefficients
        k = omega / den1
        r = (omega - sig1_prev) / den1

        # precision
        pi2 = (1.0 / sig2_hat) + 0.5 * k * (k + r * delta2) # eta already added

        # ---- clamps (NUMERICAL STABILITY) ----
        min_pi2 = 1e-6  # floor on precision (avoids huge sig2)
        max_sig2 = 1e2  # cap on variance (avoids huge updates)

        pi2 = max(pi2, min_pi2)  # clamp on precision
        sig2_new = 1.0 / pi2
        sig2_new = min(sig2_new, max_sig2)  # clamp on variance
        mu2_new = mu2_prev + 0.5 * sig2_new * k * delta2

        # state update of the model
        self.mu1, self.mu2 = mu1_new, mu2_new
        self.sig1, self.sig2 = sig1_new, sig2_new

        variational_free_energy = self.calc_variational_free_energy(
            observation=o,
            mu1_prev=mu1_prev,
            sigma1_prev=sig1_prev,
            mu2_prev=mu2_prev,
            sigma2_prev=sig2_prev_post,
        )

        # Log - append row to DataFrame
        row_data = {
            "o": o,
            "mu1_hat": mu1_prev,
            "sig1_hat": sig1_prev,
            "mu1": self.mu1,
            "sig1": self.sig1,
            "mu2_hat": mu2_prev,
            "sig2_hat": sig2_hat,
            "mu2": self.mu2,
            "sig2": self.sig2,
            "omega": omega,
            "delta1": delta1,
            "alpha1": alpha1,
            "delta2": delta2,
            "k": k,
            "r": r,
            "vfe": variational_free_energy,
        }

        self.history = pd.concat([self.history, pd.DataFrame([row_data])], ignore_index=True)
    
    def calc_variational_free_energy(
        self,
        observation: float,
        mu1_prev: float,
        sigma1_prev: float,
        mu2_prev: float,
        sigma2_prev: float,
    ) -> float:
        
        minvar = self.cfg.min_var
        s = max(self.cfg.s, minvar)
        sig1 = max(self.sig1, minvar)
        sig2 = max(self.sig2, minvar)

        v1 = max(sigma1_prev + exp_clip(self.mu2, self.cfg.exp_clip_value), minvar)
        v2 = max(sigma2_prev + self.cfg.eta, minvar)

        term_a = np.log(s)
        term_a *= -0.5

        term_b = (self.sig1 + ((observation - self.mu1) ** 2)) / s
        term_b *= -0.5

        term_c = np.log(v1)
        term_c *= -0.5

        term_d = (sig1 + (self.mu1 - mu1_prev) ** 2) / v1
        term_d *= -0.5

        term_e = np.log(v2)
        term_e *= -0.5

        term_f = (self.sig2 + (self.mu2 - mu2_prev) ** 2) / v2
        term_f *= -0.5

        term_g = np.log(2 * np.pi)
        term_g *= -0.5

        term_h = 1

        term_i = np.log(max(sig1, minvar)) + np.log(max(sig2, minvar))
        term_i *= 0.5

        free_energy = term_a + term_b + term_c + term_d + term_e + term_f + term_g + term_h + term_i

        return free_energy

    
    # ----------- Model Interface -----------
    def run(self, observations: np.ndarray) -> pd.DataFrame:
        # reset state & history
        self.mu1 = self.cfg.mu1_0
        self.sig1 = max(self.cfg.sig1_0, self.cfg.min_var)
        self.mu2 = self.cfg.mu2_0
        self.sig2 = max(self.cfg.sig2_0, self.cfg.min_var)
        self.history = pd.DataFrame(columns=self.history.columns)

        initial_data = {
            "o": 0.0,
            "mu1_hat": self.cfg.mu1_0,
            "sig1_hat": self.cfg.sig1_0 + self.cfg.eta,  # initial prediction includes process noise
            "mu1": self.cfg.mu1_0,
            "sig1": self.cfg.sig1_0,
            "mu2_hat": self.cfg.mu2_0,
            "sig2_hat": self.cfg.sig2_0 + self.cfg.eta,
            "mu2": self.cfg.mu2_0,
            "sig2": self.cfg.sig2_0,
            "omega": exp_clip(self.cfg.mu2_0, self.cfg.exp_clip_value),
            "delta1": 0.0,
            "alpha1": 0.0,
            "delta2": 0.0,
            "k": 0.0,
            "r": 0.0,
            "vfe": 0.0,
        }

        self.history = pd.DataFrame([initial_data])

        for x in observations:
            self.update(x)

        output = self.history[["mu1", "delta1", "alpha1", "vfe"]].copy()

        output["updates"] = self.history["mu1"].diff().shift(-1)

        output = output.rename(
            columns={
                "mu1": "beliefs",
                "delta1": "prediction_errors",
                "alpha1": "learning_rates",
                "vfe": "variational_free_energy",
            }
        )

        return output
    
    def set_parameters_cma(self, theta: np.ndarray) -> None:
        """
        CMA-ES parameterisation: 
            theta = [mu1_0, log_sig1_0, mu2_0, log_sig2_0, log_eta, log_s]
        where sig1_0, sig2_0, eta, and s are all variances to ensure positiviy.
        """
        mu1_0, log_sig1_0, mu2_0, log_sig2_0, log_eta, log_s = map(float, theta)

        self.cfg.mu1_0 = mu1_0
        self.cfg.sig1_0 = max(np.exp(log_sig1_0), self.cfg.min_var)
        self.cfg.mu2_0 = mu2_0
        self.cfg.sig2_0 = max(np.exp(log_sig2_0), self.cfg.min_var)
        self.cfg.eta = max(np.exp(log_eta), self.cfg.min_var)
        self.cfg.s = max(np.exp(log_s), self.cfg.min_var)

        # re-validate / clamp parameters after setting
        if self.cfg.sig1_0 < self.cfg.min_var:
            self.cfg.sig1_0 = self.cfg.min_var
        if self.cfg.sig2_0 < self.cfg.min_var:
            self.cfg.sig2_0 = self.cfg.min_var
        
        # reset state and clear history
        self.mu1 = self.cfg.mu1_0
        self.sig1 = max(self.cfg.sig1_0, self.cfg.min_var)
        self.mu2 = self.cfg.mu2_0
        self.sig2 = max(self.cfg.sig2_0, self.cfg.min_var)

        self.history = pd.DataFrame(columns=self.history.columns)

    def objective_cma(self, observations: np.ndarray) -> float:
        """
        Objective function for CMA-ES optimization: negative log-likelihood of the observations
        given the current model parameters. We can use the variational free energy as a proxy for
        the negative log-likelihood, since it approximates the evidence lower bound.
        """
        output = self.run(observations)
        total_vfe = output["variational_free_energy"].to_numpy()
        if not np.isfinite(total_vfe).all():
            raise FloatingPointError("Non-finite variational free energy values in HGF: {F}.")
        return -float(total_vfe.sum())
    
    @staticmethod
    def decode_cma_theta(theta: np.ndarray) -> dict:
        """
        Map CMA parameter vector back to named parameters for interpretability.

        theta = [mu1_0, log_sig1_0, mu2_0, log_sig2_0, log_eta, log_s]
        """
        mu1_0, log_sig1_0, mu2_0, log_sig2_0, log_eta, log_s = map(float, theta)

        return {
            "mu1_0": mu1_0,
            "sig1_0": max(np.exp(log_sig1_0), 1e-8),
            "mu2_0": mu2_0,
            "sig2_0": max(np.exp(log_sig2_0), 1e-8),
            "eta": max(np.exp(log_eta), 1e-8),
            "s": max(np.exp(log_s), 1e-8),
            "log_sig1_0": log_sig1_0,
            "log_sig2_0": log_sig2_0,
            "log_eta": log_eta,
            "log_s": log_s,
        }

    # ---------- Plotting (adapted from plot_results in hgf/hgf3.py) ----------
    def plot_results(
        self,
        true_x1: Iterable[Number] | None = None,
        true_x2: Iterable[Number] | None = None,
    ):
        trials = np.arange(len(self.history))
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))

        # 1) Observations vs inferred state (mu1)
        ax = axes[0]
        ax.plot(trials, self.history["o"].values, label="o (observation)", linewidth=1)
        ax.plot(trials, self.history["mu1"].values, label="mu1 (prediction)", linewidth=2)

        if true_x1 is not None:
            ax.plot(trials, list(true_x1), label="true x1", linewidth=1)

        ax.set_ylabel("Observation / state")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2) State uncertainty (sig1)
        ax = axes[1]
        mu1 = self.history["mu1"].values
        sig1 = np.maximum(self.history["sig1"].values, 0.0)
        ax.plot(trials, mu1, label="mu1", linewidth=2)
        ax.fill_between(trials, mu1 - np.sqrt(sig1), mu1 + np.sqrt(sig1), alpha=0.2)
        ax.set_ylabel("Level 1: x1 belief")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3) Volatility belief (mu2) + omega = exp(mu2_prev)
        ax = axes[2]
        mu2 = self.history["mu2"].values
        sig2 = np.maximum(self.history["sig2"].values, 0.0)
        ax.plot(trials, mu2, label="mu2 (volatility belief)", linewidth=2)
        ax.fill_between(trials, mu2 - np.sqrt(sig2), mu2 + np.sqrt(sig2), alpha=0.2)

        if true_x2 is not None:
            ax.plot(trials, list(true_x2), label="true x2", linewidth=1)

        ax.set_ylabel("Level 2: x2 belief")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4) Learning rate alpha1 + delta2 (diagnostics)
        ax = axes[3]
        ax.plot(trials, self.history["alpha1"].values, label="alpha1 (learning rate)", linewidth=2)
        ax.plot(trials, self.history["delta2"].values, label="delta2 (volatility PE)", linewidth=1)
        ax.set_ylabel("Diagnostics")
        ax.set_xlabel("Trial")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# ============================================================
# Sanity test: simulate data from the paper generative model
# ============================================================


def simulate_paper_environment(
    duration: int = 320,
    eta_true: float = 0.05,
    s_true: float = 15.0**2,
    x2_baseline: float = -4.0,
    burst_every: int = 100,
    burst_len: int = 8.0,
    burst_add: float = 0.7,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    x2 = np.zeros(duration, dtype=float)
    x1 = np.zeros(duration, dtype=float)

    x2[0] = x2_baseline
    x1[0] = 0.0

    for t in range(1, duration):
        x2[t] = x2[t - 1] + np.sqrt(eta_true) * rng.standard_normal()

        if burst_every > 0 and (t % burst_every) < burst_len:
            x2[t] += burst_add

        # step std = exp(x2/2)
        x1[t] = x1[t - 1] + np.exp(x2[t] / 2.0) * rng.standard_normal()

    o = x1 + np.sqrt(s_true) * rng.standard_normal(duration)
    return o, x1, x2


def _demo_paper_hgf2():
    o, x1_true, x2_true = simulate_paper_environment()

    model = HGFPaper2Gaussian(
        eta=0.02,
        s=225.0,
        mu1_init=0.0,
        sig1_init=25.0,
        mu2_init=-4.0,
        sig2_init=1.5,
        min_var=1e-8,
        exp_clip_value=30.0,
    )
    model.run(o)
    model.plot_results(true_x1=x1_true, true_x2=x2_true)
    plt.show()


if __name__ == "__main__":
    _demo_paper_hgf2()