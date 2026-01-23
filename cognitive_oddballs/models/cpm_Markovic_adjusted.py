import numpy as np
import pandas as pd
from scipy import stats


class ChangePointModel_VarInference:
    """
    Change Point Model (CPM) following Marković & Kiebel (2016).

    Perceptual free parameters:
        μ₀¹, σ₀¹, s, w₁, w₂, h
    """

    def __init__(
        self,
        x,
        mu0,
        sigma0,
        obs_noise,
        w1,
        w2,
        h
    ):
        # Observations
        self.x = np.asarray(x)
        self.n_trials = len(x)

        # ===== Perceptual free parameters =====
        self.mu0 = mu0                  # μ₀¹
        self.sigma0 = sigma0            # σ₀¹
        self.obs_noise = obs_noise      # s
        self.w1 = w1                    # w₁
        self.w2 = w2                    # w₂
        self.hazard_rate = h            # h

        # ===== Latent states =====
        self.mu = mu0
        self.var = sigma0 ** 2

        self.history = {
            "beliefs": [],
            "prediction_errors": [],
            "learning_rates": [],
            "uncertainties": [],
            "change_point_probs": [],
        }

    # --------------------------------------------------

    def _change_point_probability(self, delta):
        """
        Bayesian change-point probability Ω_t
        """

        # Likelihood under no change
        var_no_cp = self.var + self.obs_noise ** 2
        like_no_cp = stats.norm.pdf(delta, 0.0, np.sqrt(var_no_cp))

        # Likelihood under change (uniform prior)
        like_cp = 1.0 / 300.0

        num = self.hazard_rate * like_cp
        den = num + (1 - self.hazard_rate) * like_no_cp

        return np.clip(num / den, 1e-6, 1 - 1e-6)

    # --------------------------------------------------

    def update(self, t):
        """
        Single-trial variational update
        """

        # Prediction error
        delta = self.x[t] - self.mu

        # Change-point probability
        omega = self._change_point_probability(delta)

        # Relative uncertainty τ_t
        tau = self.var / (self.var + self.obs_noise ** 2)

        # Learning rate (Table 1 form from Marković & Kiebel, 2016)
        alpha = self.w1 * omega + self.w2 * tau
        alpha = np.clip(alpha, 0.0, 1.0)

        # Posterior mean update
        self.mu = self.mu + alpha * delta
        self.mu = np.clip(self.mu, 0, 500)

        # Posterior variance update
        self.var = (
            omega * self.sigma0 ** 2
            + (1 - omega) * (1 - alpha) * self.var
        )

        # Store
        self.history["beliefs"].append(self.mu)
        self.history["prediction_errors"].append(delta)
        self.history["learning_rates"].append(alpha)
        self.history["uncertainties"].append(tau)
        self.history["change_point_probs"].append(omega)

    # --------------------------------------------------

    def run(self, mu_true=None):
        """
        Run the CPM on the full sequence
        """

        # Reset states
        self.mu = self.mu0
        self.var = self.sigma0 ** 2

        self.history = {
            "beliefs": [self.mu],
            "prediction_errors": [0.0],
            "learning_rates": [0.0],
            "uncertainties": [self.var / (self.var + self.obs_noise ** 2)],
            "change_point_probs": [0.0],
        }

        for t in range(1, self.n_trials):
            self.update(t)

        df = pd.DataFrame(
            {
                "Trial": np.arange(1, self.n_trials + 1),
                "BagDrop": self.x,
                "Belief": self.history["beliefs"],
                "CPP": self.history["change_point_probs"],
                "RelUncertainty": self.history["uncertainties"],
                "LearningRate": self.history["learning_rates"],
                "PredictionError": self.history["prediction_errors"],
            }
        )

        if mu_true is not None:
            df.insert(1, "TruePosition", mu_true)

        return df
