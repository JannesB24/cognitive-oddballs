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


def sigmoid_stable(x: Number) -> float:
    """Numerically stable sigmoid."""
    x = float(x)
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    else:
        z = np.exp(x)
        return float(z / (1.0 + z))


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


class HGFPaper2Gaussian(Model):
    """
    2-level Gaussian HGF as in Markovic & Kiebel (2016):
    x2_t = x2_{t-1} + sqrt(eta) * n
    x1_t = x1_{t-1} + exp(x2_t/2) * n   -> Var = exp(x2_t)
    o_t  = x1_t + sqrt(s) * n
    """

    def __init__(
        self,
        eta: float,
        s: float,
        mu1_init: float = 0.0,
        sig1_init: float = 1.0,
        mu2_init: float = 0.0,
        sig2_init: float = 1.0,
        min_var: float = 1e-8,
        exp_clip_value: float = 60.0,
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

        if self.cfg.eta <= 0:
            raise ValueError("eta must be > 0")
        if self.cfg.s <= 0:
            raise ValueError("s (observation variance) must > 0")

        # current posteriors
        self.mu1 = self.cfg.mu1_0
        self.sig1 = max(self.cfg.sig1_0, self.cfg.min_var)
        self.mu2 = self.cfg.mu2_0
        self.sig2 = max(self.cfg.sig2_0, self.cfg.min_var)

        # trial counter
        self.trial = 0

        self.history: dict[str, list[float]] = {
            "F_t": [],
            "o": [],
            "mu1_hat": [],
            "sig1_hat": [],
            "mu1": [],
            "sig1": [],
            "mu2_hat": [],
            "sig2_hat": [],
            "mu2": [],
            "sig2": [],
            "omega": [],  # exp(mu2_hat)
            "delta1": [],
            "alpha1": [],
            "delta2": [],
            "k": [],
            "r": [],
        }

    def update(self, o: Number) -> None:
        o = float(o)
        minvar = self.cfg.min_var

        # ----------- Prediction -----------------
        # Level 2
        mu2_hat = self.mu2
        sig2_hat = max(self.sig2 + self.cfg.eta, minvar)
        # Level 1
        # omega = exp(mu2_hat)
        omega = exp_clip(mu2_hat, self.cfg.exp_clip_value)

        mu1_prev = self.mu1
        sig1_prev = self.sig1
        mu2_prev = self.mu2
        sig2_prev = self.sig2
        # denominator helper
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
        pi2 = (1.0 / sig2_hat) + 0.5 * k * (k + r * delta2)

        # posterior update
        sig2_new = 1.0 / max(pi2, minvar)
        mu2_new = mu2_hat + 0.5 * max(sig2_new, minvar) * k * delta2

        # state update of the model
        self.mu1, self.mu2 = mu1_new, mu2_new
        self.sig1, self.sig2 = sig1_new, sig2_new

        # ------------ Free energy (eq. 29) ---------------
        # TODO: LLM-generated -- verify correctness
        s = self.cfg.s

        den1 = max(sig1_prev + omega, minvar)
        den2 = max(sig2_prev + self.cfg.eta, minvar)

        term1 = -0.5 * np.log(s)
        term2 = -0.5 * den1 / s
        term3 = -0.5 * np.log(den1)
        term4 = 0.5 * (sig1_new + (mu1_new - mu1_prev) ** 2) / den1
        term5 = 0.5 * np.log(den2)
        term6 = 0.5 * (sig2_new + (mu2_new - mu2_prev) ** 2) / den2
        term7 = -0.5 * np.log(2.0 * np.pi)
        term8 = 0.5 * np.log(max(sig1_new, minvar))
        term9 = 0.5 * np.log(max(sig2_new, minvar))

        vfe = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
        self.history["F_t"].append(float(vfe))

        # Log
        self.history["o"].append(o)
        self.history["mu1_hat"].append(mu1_prev)
        self.history["sig1_hat"].append(den1)
        self.history["mu1"].append(self.mu1)
        self.history["sig1"].append(self.sig1)
        self.history["mu2_hat"].append(mu2_hat)
        self.history["sig2_hat"].append(sig2_hat)
        self.history["mu2"].append(self.mu2)
        self.history["sig2"].append(self.sig2)
        self.history["omega"].append(omega)
        self.history["delta1"].append(delta1)
        self.history["alpha1"].append(alpha1)
        self.history["delta2"].append(delta2)
        self.history["k"].append(k)
        self.history["r"].append(r)

    # ---------- Model interface implementation ----------

    def run(self, observations: np.ndarray) -> pd.DataFrame:
        for x in observations:
            self.update(x)

        output = self.history[["x_0_expected_mean"]]

        rename_dict = {"x_0_expected_mean": "raw_responses"}

        return output.rename(columns=rename_dict)
    
    # TODO: LLM-generated -- verify correctness
    # ALSO this does not include all parameters yet
    def set_parameters_cma(self, theta: np.ndarray) -> None:
        """
        If all:
        theta = [log_eta, log_s, mu1_0, log_sig1_0, mu2_0, log_sig2_0]
        Currently:
        theta = [log_eta, log_s, mu2_0, log_sig2_0]
        """
        log_eta, log_s, mu2_0, log_sig2_0 = map(float, theta)

        self.cfg.eta = np.exp(log_eta)
        self.cfg.s = np.exp(log_s)
        self.cfg.mu2_0 = mu2_0
        self.cfg.sig2_0 = np.exp(log_sig2_0)

        # reset state to the priors for this parameter setting
        self.mu1 = self.cfg.mu1_0
        self.sig1 = max(self.cfg.sig1_0, self.cfg.min_var)
        self.mu2 = self.cfg.mu2_0
        self.sig2 = max(self.cfg.sig2_0, self.cfg.min_var)

        # reset history
        for k in self.history:
            self.history[k] = []

    # TODO: LLM-generated -- verify correctness
    def objective_cma(self, observations: np.ndarray) -> float:
        """
        Return negative free energy (≈ negative log-likelihood) over this sequence.
        """
        F_sum = 0.0
        for o in observations:
            self.update(o)
            F_sum += self.history["F_t"][-1]

        # CMA-ES minimizes, so return negative F
        return -float(F_sum)
    
    # TODO: LLM-generated -- verify correctness
    # does not include all parameters yet
    @staticmethod
    def decode_cma_theta(theta: np.ndarray) -> dict:
        """
        Map CMA parameter vector back to named, interpretable parameters.
        """
        log_eta, log_s, mu2_0, log_sig2_0 = map(float, theta)

        return {
            "eta": float(np.exp(log_eta)),
            "s": float(np.exp(log_s)),
            "mu2_0": mu2_0,
            "sig2_0": float(np.exp(log_sig2_0)),
            # possibly: expose log-params as well for diagnostics
            "log_eta": log_eta,
            "log_s": log_s,
            "log_sig2_0": log_sig2_0,
        }

    # ---------- Plotting (adapted from plot_results in hgf/hgf3.py) ----------
    def plot_results(
        self,
        true_x1: Iterable[Number] | None = None,
        true_x2: Iterable[Number] | None = None,
    ):
        trials = np.arange(len(self.history["o"]))
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))

        # 1) Observations vs inferred state (mu1)
        ax = axes[0]
        ax.plot(trials, self.history["o"], label="o (observation)", linewidth=1)
        ax.plot(trials, self.history["mu1_hat"], label="mu1_hat (prediction)", linewidth=2)
        ax.plot(trials, self.history["mu1"], label="mu1 (posterior)", linewidth=2)

        if true_x1 is not None:
            ax.plot(trials, list(true_x1), label="true x1", linewidth=1)

        ax.set_ylabel("Observation / state")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2) State uncertainty (sig1)
        ax = axes[1]
        mu1 = np.array(self.history["mu1"])
        sig1 = np.maximum(np.array(self.history["sig1"]), 0.0)
        ax.plot(trials, mu1, label="mu1", linewidth=2)
        ax.fill_between(trials, mu1 - np.sqrt(sig1), mu1 + np.sqrt(sig1), alpha=0.2)
        ax.set_ylabel("Level 1: x1 belief")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3) Volatility belief (mu2) + omega = exp(mu2_hat)
        ax = axes[2]
        mu2 = np.array(self.history["mu2"])
        sig2 = np.maximum(np.array(self.history["sig2"]), 0.0)
        ax.plot(trials, mu2, label="mu2 (volatility belief)", linewidth=2)
        ax.fill_between(trials, mu2 - np.sqrt(sig2), mu2 + np.sqrt(sig2), alpha=0.2)

        if true_x2 is not None:
            ax.plot(trials, list(true_x2), label="true x2", linewidth=1)

        ax.set_ylabel("Level 2: x2 belief")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4) Learning rate alpha1 + delta2 (diagnostics)
        ax = axes[3]
        ax.plot(trials, self.history["alpha1"], label="alpha1 (learning rate)", linewidth=2)
        ax.plot(trials, self.history["delta2"], label="delta2 (volatility PE)", linewidth=1)
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
    burst_add: float = 0.9,
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
        eta=0.005,
        s=15.0**2,
        mu1_init=0.0,
        sig1_init=10.0,
        mu2_init=-4.0,  # vicino a x2_baseline
        sig2_init=1.0,
    )
    model.run(o)
    model.plot_results(true_x1=x1_true, true_x2=x2_true)
    plt.show()


if __name__ == "__main__":
    _demo_paper_hgf2()
