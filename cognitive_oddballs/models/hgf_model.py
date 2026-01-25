from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

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
class HGFConfig:
    # Core HGF parameters (Mathys et al. style)
    kappa: float
    omega: float
    theta: float

    # Observation model
    mode: str = "bernoulli"  # "bernoulli" or "gaussian"

    # For gaussian mode:
    # Default is tuned for "scale A" tasks
    # (e.g., outcomes in ~0..300 with observation SD ~5/15/25/35).
    # See Nassar et al. 2010 for typical SD magnitudes.  :contentReference[oaicite:2]{index=2}
    sigma_obs: float = 15.0  # observation noise standard deviation (same units as u)

    # Numerical stability
    min_var: float = 1e-8
    exp_clip_value: float = 100.0


class HGF:
    """
    Hierarchical Gaussian Filter (Mathys et al. 2011/2014 style updates).

    Supports:
      - Bernoulli-HGF: binary observations (u in {0,1}), observation model is sigmoid(m2)
      - Gaussian-HGF: continuous observations (u in R), observation model is identity with
        fixed sigma_obs
    """

    def __init__(
        self,
        kappa: float,
        omega: float,
        theta: float,
        mode: str = "bernoulli",
        sigma_obs: float = 15.0,
        # Optional initial states
        m2_init: float = 0.0,
        m3_init: float = 0.0,
        s2_init: float = 1.0,
        s3_init: float = 1.0,
        # Numerical stability
        min_var: float = 1e-8,
        exp_clip_value: float = 60.0,
    ):
        self.cfg = HGFConfig(
            kappa=float(kappa),
            omega=float(omega),
            theta=float(theta),
            mode=str(mode).lower(),
            sigma_obs=float(sigma_obs),
            min_var=float(min_var),
            exp_clip_value=float(exp_clip_value),
        )

        if self.cfg.mode not in ("bernoulli", "gaussian"):
            raise ValueError("mode must be 'bernoulli' or 'gaussian'")

        if self.cfg.mode == "gaussian" and self.cfg.sigma_obs <= 0:
            raise ValueError("sigma_obs must be > 0 for gaussian mode")

        # Posterior means/variances for levels 2 and 3
        self.m2 = float(m2_init)
        self.m3 = float(m3_init)
        self.s2 = float(s2_init)
        self.s3 = float(s3_init)

        self.counter = 0
        self.history: dict[str, list[float]] = {
            "u": [],
            "m1_hat": [],  # predicted mean at observation level
            "s1_hat": [],  # predicted variance at observation level
            "m2": [],
            "m3": [],
            "s2": [],
            "s3": [],
            "d1": [],  # prediction error level 1
            "d2": [],  # prediction error level 2
        }

    # ---------- Level 1 (observation model) ----------
    def _level1_predict(self) -> tuple[float, float]:
        """
        Returns:
          m1_hat: predicted mean of observation
          s1_hat: predicted variance at observation level
        """
        if self.cfg.mode == "bernoulli":
            # Bernoulli-HGF: m1_hat = sigmoid(m2), var = p(1-p)
            m1_hat = sigmoid_stable(self.m2)
            s1_hat = m1_hat * (1.0 - m1_hat)
            s1_hat = max(s1_hat, self.cfg.min_var)
            return m1_hat, s1_hat

        # Gaussian-HGF: identity observation model, fixed observation noise variance
        m1_hat = float(self.m2)
        s1_hat = max(self.cfg.sigma_obs**2, self.cfg.min_var)
        return m1_hat, s1_hat

    # ---------- Predictions (priors) ----------
    def _predictions(self) -> tuple[float, float, float, float, float]:
        """
        Using (k-1) values to predict trial k.

        Returns:
          pi1_hat, pi2_hat, pi3_hat, m1_hat, s1_hat
        """
        m1_hat, s1_hat = self._level1_predict()
        pi1_hat = 1.0 / max(s1_hat, self.cfg.min_var)

        e3 = exp_clip(self.cfg.kappa * self.m3 + self.cfg.omega, self.cfg.exp_clip_value)
        s2_hat = max(self.s2 + e3, self.cfg.min_var)
        pi2_hat = 1.0 / s2_hat

        pi3_hat = 1.0 / max(self.s3 + self.cfg.theta, self.cfg.min_var)

        return pi1_hat, pi2_hat, pi3_hat, m1_hat, s1_hat

    # ---------- Main update ----------
    def update(self, u: Number) -> None:
        """
        Update beliefs given observation u at time k.

        - bernoulli: u should be 0/1 (will be cast to float)
        - gaussian:  u is continuous float (scale A ~0..300 expected by default sigma_obs=15)
        """
        u = float(u)

        # Step 1: predictions
        pi1_hat, pi2_hat, pi3_hat, m1_hat, s1_hat = self._predictions()

        # Step 2–3: level 1 prediction error
        d1 = u - m1_hat

        # Step 4: update level 2 (m2 and s2)
        s2_prev = self.s2
        m2_prev = self.m2

        # Keep the same structure as your original script:
        # pi2 = pi2_hat + (1/pi1_hat) = pi2_hat + s1_hat
        pi2 = pi2_hat + (1.0 / max(pi1_hat, self.cfg.min_var))
        self.s2 = 1.0 / max(pi2, self.cfg.min_var)

        self.m2 = self.m2 + self.s2 * d1

        # Step 5: level 2 PE and helper terms
        e3 = exp_clip(self.cfg.kappa * self.m3 + self.cfg.omega, self.cfg.exp_clip_value)
        denom = max(e3 + s2_prev, self.cfg.min_var)

        w2 = e3 / denom
        r2 = (e3 - s2_prev) / denom

        d2 = (self.s2 + (self.m2 - m2_prev) ** 2) / max(s2_prev + e3, self.cfg.min_var) - 1.0

        # Step 6: update level 3 (m3 and s3)
        pi3 = pi3_hat + (self.cfg.kappa**2 / 2.0) * w2 * (w2 + r2 * d2)
        self.s3 = 1.0 / max(pi3, self.cfg.min_var)

        self.m3 = self.m3 + self.s3 * (self.cfg.kappa / 2.0) * w2 * d2

        # Log history
        self.history["u"].append(u)
        self.history["m1_hat"].append(m1_hat)
        self.history["s1_hat"].append(s1_hat)
        self.history["m2"].append(self.m2)
        self.history["m3"].append(self.m3)
        self.history["s2"].append(self.s2)
        self.history["s3"].append(self.s3)
        self.history["d1"].append(d1)
        self.history["d2"].append(d2)

        self.counter += 1

    def run(self, inputs: Iterable[Number]) -> dict[str, list[float]]:
        for u in inputs:
            self.update(u)
        return self.history

    # ---------- Plotting ----------
    def plot_results(
        self,
        true_latent: Iterable[Number] | None = None,
        debug_level1_raw_m2: bool = False,
    ):
        trials = np.arange(len(self.history["u"]))
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Level 1: observations and prediction
        ax = axes[0]
        ax.plot(trials, self.history["u"], label="u (observation)", linewidth=1)

        if debug_level1_raw_m2:
            ax.plot(trials, self.history["m2"], label="raw m2", linewidth=2)
            ax.set_ylabel("Level 1 (debug): raw m2")
        else:
            ax.plot(trials, self.history["m1_hat"], label="m1_hat (prediction)", linewidth=2)
            ax.set_ylabel("Level 1: prediction")

        if true_latent is not None:
            ax.plot(trials, list(true_latent), label="true latent", linewidth=1)

        ax.legend()
        ax.grid(True, alpha=0.3)

        # Level 2
        ax = axes[1]
        m2 = np.array(self.history["m2"])
        s2 = np.maximum(np.array(self.history["s2"]), 0.0)
        ax.plot(trials, m2, label="m2", linewidth=2)
        ax.fill_between(trials, m2 - np.sqrt(s2), m2 + np.sqrt(s2), alpha=0.2)
        ax.set_ylabel("Level 2: x2")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Level 3
        ax = axes[2]
        m3 = np.array(self.history["m3"])
        s3 = np.maximum(np.array(self.history["s3"]), 0.0)
        ax.plot(trials, m3, label="m3", linewidth=2)
        ax.fill_between(trials, m3 - np.sqrt(s3), m3 + np.sqrt(s3), alpha=0.2)
        ax.set_ylabel("Level 3: x3")
        ax.set_xlabel("Trial")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# -----------------------------
# Minimal self-tests (optional)
# -----------------------------
def _demo_bernoulli():
    np.random.seed(0)
    p = np.concatenate([np.full(80, 0.2), np.full(80, 0.8)])
    u = (np.random.rand(len(p)) < p).astype(int)

    hgf = HGF(kappa=1.8, omega=-3.0, theta=0.5, mode="bernoulli")
    hgf.run(u)
    hgf.plot_results(true_latent=p, debug_level1_raw_m2=False)


def generate_reference_scenario_bernoulli(seed: int = 42):
    """
    Reference scenario (Bernoulli):
      - Stage 1: 100 trials with p=0.5
      - Stage 2: 120 trials alternating p=0.9 and p=0.1 in 20-trial blocks
      - Stage 3: 100 trials with p=0.5
    Returns:
      inputs (np.ndarray of int 0/1),
      true_prob (np.ndarray of float)
    """
    rng = np.random.default_rng(seed)

    # Stage 1: 100 trials with p=0.5
    p1 = 0.5
    stage1 = (rng.random(100) < p1).astype(int)
    true_prob1 = np.full(100, p1, dtype=float)

    # Stage 2: 120 trials alternating high/low probability in blocks of 20
    stage2 = []
    true_prob2 = []
    for i in range(6):  # 6 blocks * 20 = 120
        p = 0.9 if i % 2 == 0 else 0.1
        stage2_block = (rng.random(20) < p).astype(int)
        stage2.append(stage2_block)
        true_prob2.append(np.full(20, p, dtype=float))

    stage2 = np.concatenate(stage2)
    true_prob2 = np.concatenate(true_prob2)

    # Stage 3: 100 trials with p=0.5
    stage3 = (rng.random(100) < p1).astype(int)
    true_prob3 = np.full(100, p1, dtype=float)

    inputs = np.concatenate([stage1, stage2, stage3]).astype(int)
    true_prob = np.concatenate([true_prob1, true_prob2, true_prob3]).astype(float)

    return inputs, true_prob


def _demo_reference_scenario_bernoulli():
    inputs, true_prob = generate_reference_scenario_bernoulli(seed=42)

    hgf = HGF(kappa=1.8, omega=-3.0, theta=0.5, mode="bernoulli")
    hgf.run(inputs)

    # Plotta prediction (m1_hat) vs true probability
    hgf.plot_results(true_latent=true_prob, debug_level1_raw_m2=False)


def _demo_gaussian_scale_a():
    np.random.seed(0)
    mu = np.concatenate([np.full(80, 100.0), np.full(80, 200.0)])
    # observation noise consistent with scale A
    u = mu + np.random.normal(0, 15.0, size=len(mu))

    hgf = HGF(kappa=1.8, omega=-3.0, theta=0.5, mode="gaussian", sigma_obs=15.0)
    hgf.run(u)
    hgf.plot_results(true_latent=mu, debug_level1_raw_m2=False)


if __name__ == "__main__":
    _demo_bernoulli()
    _demo_reference_scenario_bernoulli()
    _demo_gaussian_scale_a()
    plt.show()
