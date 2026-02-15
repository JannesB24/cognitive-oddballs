# LLM GENERATED

"""
Change Point Model with Variational Inference (Marković & Kiebel, 2016)

This module implements the Change Point Model (CPM) reformulated using
variational Bayesian inference as described in Marković & Kiebel (2016).

The reformulation makes the CPM directly comparable to Hierarchical Gaussian
Filter (HGF) models by:
1. Using the same variational inference framework
2. Adding explicit second-level hierarchical variables
3. Providing comparable learning signals at each level

Reference:
    Marković, D., & Kiebel, S. J. (2016). Comparative Analysis of Behavioral
    Models for Adaptive Learning in Changing Environments. Frontiers in
    Computational Neuroscience, 10, 33.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from cognitive_oddballs.environments.change_point_oddball import generate_change_point_environment
from cognitive_oddballs.environments.random_walk_oddball import generate_random_walk_environment
from cognitive_oddballs.models.model import Model

logger = logging.getLogger(__name__)


class ChangePointModelVariational(Model):
    """
    Change Point Model using Variational Bayesian Inference.

    Implements the switching-state-space formulation with variational
    Bayesian inference from Marković & Kiebel (2016), Section 4.1.2.

    The model assumes the environment switches between:
    - Stability (1 - h): x_t = x_{t-1} + √w₁ · noise
    - Change-point (h): x_t = √w₂ · noise

    Parameters
    ----------
    mu0 : float
        Initial belief about hidden state POSITION (μ₀¹)
        0 <= mu0 <= 500 (ideally the centre as the starting position)
    sigma0 : float
        Initial uncertainty (σ₀¹)
        0 < sigma0 <= 100
    obs_noise : float
        Observation noise STANDARD DEVIATION (s)
        obs_noise > 0  (matched to the environment noise)
    w1_std : float
        Stability drift STANDARD DEVIATION (√w₁)
        0 ≤ w1_std ≤ 10 for typical continuous drift.
        For change-point environments: 0 or very small (ie. 0.1) for minimal drift.
        Note: Squared internally to get variance w₁
    w2 : float
        Change-point jump STANDARD DEVIATION (√w₂)
        130 <= w2 <= 125000 for an environment with an observation range of 0-500.
        Note: Squared internally to get variance w₂
    h : float
        Hazard rate - prior probability of change-point (h)
        Typical value: 0.1 (10% chance per trial)
    add_second_level : bool, optional
        Whether to track second-level hierarchical variables for
        comparison with HGF models (default: True)

    Attributes
    ----------
    mu : float
        Current posterior expectation (μ^(1))
    sigma : float
        Current posterior uncertainty (σ^(1))
    mu2 : float
        Second-level belief (log-odds of change-point probability)
    sigma2 : float
        Second-level uncertainty
    history : dict
        Trial-by-trial tracking of all internal variables

    Examples
    --------
    >>> import numpy as np
    >>> observations = np.random.normal(250, 10, 100)
    >>> model = ChangePointModelVariational(
    ...     mu0=250,
    ...     sigma0=10,
    ...     obs_noise=10,
    ...     w1=0.01,
    ...     w2=900,
    ...     h=0.1
    ... )
    >>> results = model.run()
    >>> print(results[['Trial', 'Belief', 'LearningRate']].head())
    """

    def __init__(
        self,
        mu0: float = 250.0,  # just needed to put any default parameters here for eval to work. they'll be overwritten anyway - R.
        sigma0: float = 10.0,
        obs_noise_std: float = 10.0,
        w1_std: float = 0.01,
        w2_std: float = 100.0,
        h: float = 0.1,
        add_second_level: bool = True,
    ) -> None:
        # ===== Perceptual free parameters =====
        self.mu0 = float(mu0)  # μ₀¹ - Initial belief
        self.sigma0 = float(sigma0)  # σ₀¹ - Initial uncertainty
        self.obs_noise = float(obs_noise_std)  # s - Observation noise (std dev)
        self.hazard_rate = float(h)  # h - Hazard rate

        # STORE STANDARD DEVITAION VALUES
        self.obs_noise_std = float(obs_noise_std)  # s - Observation noise (std dev)
        self.w1_std = float(w1_std)  # w₁ - Stability diffusion (std dev)
        self.w2_std = float(w2_std)  # w₂ - Change-point variance (std dev)

        # VARIANCE CONVERSION FOR INTERNAL USE
        self.obs_noise_var = obs_noise_std**2
        self.w1_var = w1_std**2
        self.w2_var = w2_std**2

        # First-level latent states
        self.mu = mu0  # μ^(1) - Posterior expectation
        self.sigma = sigma0  # σ^(1) - Posterior STANDARD DEVIATION

        # Second-level states (for HGF comparability)
        self.add_second_level = add_second_level
        if add_second_level:
            self.mu2 = 0.0  # μ^(2) - Volatility (log-odds of Ω)
            self.sigma2 = 1.0  # σ^(2) - Second-level uncertainty

        # History tracking
        columns = [
            "beliefs",  # μ^(1)
            "prediction_errors",  # δ_t
            "learning_rates",  # α^(1)
            "uncertainties",  # σ^(1)
            "change_point_probs",  # Ω_t
            "variational_free_energy",  # Variational free energy
        ]

        if add_second_level:
            columns.extend(
                [
                    "mu2",  # μ^(2)
                    "epsilon2",  # ε^(2) - Second-level prediction error
                    "alpha2",  # α^(2) - Second-level learning rate
                ]
            )

        self.history = pd.DataFrame(columns=columns)

    # helper fct
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    # --------------------------------------------------
    # Second-level transformations (for HGF comparability)
    # --------------------------------------------------

    def _omega_to_mu2(self, omega, scaling=1.0) -> float:
        """
        Convert change-point probability to second-level representation.

        Implements the log-odds transformation from Marković & Kiebel
        Equation (5):
            μ^(2) = (1/a) * ln(Ω / (1-Ω))

        This makes the CPM's change-point probability directly comparable
        to the HGF's volatility representation.

        Parameters
        ----------
        omega : float
            Change-point probability [0, 1]
        scaling : float, optional
            Scaling constant (a in paper), default=1.0

        Returns
        -------
        float
            Second-level belief μ^(2)
        """
        omega = np.clip(omega, 1e-6, 1 - 1e-6)
        return (1 / scaling) * np.log(omega / (1 - omega))

    def _mu2_to_omega(self, mu2: float, scaling: float = 1.0) -> float:
        """
        Convert second-level representation back to change-point probability.

        Inverse of _omega_to_mu2:
            Ω = 1 / (1 + e^(-a*μ^(2)))

        Parameters
        ----------
        mu2 : float
            Second-level belief
        scaling : float, optional
            Scaling constant (a in paper)

        Returns
        -------
        float
            Change-point probability Ω
        """
        return 1 / (1 + np.exp(-scaling * mu2))

    # --------------------------------------------------
    # Core inference functions
    # --------------------------------------------------

    def _change_point_probability(self, delta: float) -> float:
        """
        Compute change-point probability Ω_t using Bayes rule.

        From Marković & Kiebel Equation (4):

        Ω_t = [N(o_t; 0, s+w₂) · h] /
              [N(o_t; μ_{t-1}, σ²_{t-1}+w₁+s) · (1-h) + N(o_t; 0, s+w₂) · h]

        The change-point probability reflects how likely it is that the
        hidden state just underwent a discontinuous jump based on:
        - The prediction error magnitude
        - Prior expectations about change frequency (hazard rate)

        Parameters
        ----------
        delta : float
            Prediction error δ_t = o_t - μ_{t-1}

        Returns
        -------
        float
            Change-point probability Ω_t ∈ [0, 1]
        """
        # Likelihood under stability (no change-point)
        # x_t follows x_{t-1} with small drift w1
        var_stability = self.sigma**2 + self.w1_var + self.obs_noise_var
        like_stability = stats.norm.pdf(delta, loc=0.0, scale=np.sqrt(var_stability))

        # Likelihood under change-point
        # x_t is drawn from a wide distribution (large w2)
        var_change = self.obs_noise_var + self.w2_var
        like_change = stats.norm.pdf(delta, loc=0.0, scale=np.sqrt(var_change))

        # Bayes rule with hazard rate as prior
        numerator = self.hazard_rate * like_change
        denominator = numerator + (1 - self.hazard_rate) * like_stability

        # NaN PREVENTION (in case of zero division)
        denominator = np.maximum(denominator, 1e-10)

        omega = numerator / denominator
        return np.clip(omega, 1e-6, 1 - 1e-6)

    # --------------------------------------------------

    def update(self, observation: float):
        """
        Single-trial variational update following Marković & Kiebel (2016).

        Implements the update equations from Equation (4):

        1. μ_t^(1) = μ_{t-1}^(1) + ε_t^(1)
        2. ε_t^(1) = α_t^(1) [o_t - μ_{t-1}^(1)]
        3. α_t^(1) = σ_t^(1) / s
        4. 1/σ_t^(1) = (1-Ω_t)/(σ²_{t-1}^(1) + w₁) + 1/s

        The key insight is that the learning rate α is derived from
        posterior uncertainty σ, which itself depends on change-point
        probability Ω.

        Parameters
        ----------
        t : int
            Trial index
        """
        mu1_prev = self.mu
        sigma1_prev = self.sigma

        # 1. Prediction error
        delta = observation - mu1_prev

        # 2. Change-point probability (Bayes rule)
        omega = self._change_point_probability(delta)

        # 3. Update posterior uncertainty (inverse variance form)
        # This is the key equation that links change-points to learning
        inv_sigma_squared = (1 - omega) / (sigma1_prev**2 + self.w1_var) + 1 / self.obs_noise_var
        # NaN PREVENTION (in case of zero division)
        inv_sigma_squared = np.maximum(inv_sigma_squared, 1e-10)
        sigma_new = 1 / np.sqrt(inv_sigma_squared)

        # 4. Learning rate (ratio of posterior to observation uncertainty)
        alpha = sigma_new / self.obs_noise_std
        alpha = float(np.clip(alpha, 0.0, 1.0))  # Ensure valid range

        # 5. Update posterior expectation (delta rule)
        mu_new = mu1_prev + alpha * delta
        mu_new = float(np.clip(mu_new, 0, 500))  # Clip to valid screen range

        # 6. Second-level update (if enabled)
        epsilon2 = 0.0
        alpha2 = 0.0

        if self.add_second_level:
            # Get previous change-point probability
            omega_prev = (
                self.history["change_point_probs"].iloc[-1] if len(self.history) > 0 else omega
            )

            # Convert to log-odds space
            mu2_new = self._omega_to_mu2(omega)
            mu2_prev = self._omega_to_mu2(omega_prev)

            # Second-level prediction error
            # (how much did volatility change?)
            epsilon2 = mu2_new - mu2_prev

            # Second-level learning rate
            # (simplified version; could be made more sophisticated)
            alpha2 = self.sigma2**2 / (self.sigma2**2 + 1.0)

            # Update second-level belief
            self.mu2 = mu2_prev + alpha2 * epsilon2

        # 7. Update states
        self.mu = mu_new
        self.sigma = sigma_new

        free_energy = self.calc_variational_free_energy(observation, mu1_prev, sigma1_prev)

        # 8. Store history
        self._store_history(delta, omega, alpha, epsilon2, alpha2, free_energy)

    def _store_history(
        self,
        delta: float,
        omega: float,
        alpha: float,
        epsilon2: float = 0.0,
        alpha2: float = 0.0,
        free_energy: float = 0.0,
    ) -> None:
        """Store trial results in history."""
        row_data = {
            "beliefs": self.mu,
            "prediction_errors": delta,
            "learning_rates": alpha,
            "uncertainties": self.sigma,
            "change_point_probs": omega,
            "variational_free_energy": free_energy,
        }

        if self.add_second_level:
            row_data.update(
                {
                    "mu2": self.mu2,
                    "epsilon2": epsilon2,
                    "alpha2": alpha2,
                }
            )

        self.history = pd.concat([self.history, pd.DataFrame([row_data])], ignore_index=True)

    def calc_variational_free_energy(
        self, observation: float, mu1_prev: float, sigma1_prev: float
    ) -> float:
        """
        Calculate variational free energy for the current state.

        Returns
        -------
        float
            Variational free energy
        """
        stability_normal = stats.norm.pdf(
            x=observation,
            loc=mu1_prev,
            scale=np.sqrt(sigma1_prev**2 + self.w1_var + self.obs_noise_var),
        )

        stability_normal *= 1 - self.hazard_rate

        change_normal = stats.norm.pdf(
            observation, loc=0.0, scale=np.sqrt(self.obs_noise_var + self.w2_var)
        )

        change_normal *= self.hazard_rate

        likelihood = stability_normal + change_normal
        likelihood = np.maximum(likelihood, 1e-10)  # Prevent log(0)
        free_energy = np.log(likelihood)

        return free_energy

    # -------------------- Interface Methods --------------------------

    def run(self, observations: np.ndarray) -> pd.DataFrame:
        """
        Run the CPM on the full observation sequence.

        Parameters
        ----------
        mu_true : array-like, optional
            True hidden states (e.g., helicopter positions) for evaluation

        Returns
        -------
        pd.DataFrame
            Trial-by-trial results with columns:
            - Trial: Trial number (1-indexed)
            - BagDrop: Observed outcome
            - Belief: Model's posterior expectation (μ^(1))
            - CPP: Change-point probability (Ω)
            - Uncertainty: Posterior uncertainty (σ^(1))
            - LearningRate: Learning rate (α^(1))
            - PredictionError: Prediction error (δ)

            If add_second_level=True, also includes:
            - Mu2: Second-level belief (μ^(2))
            - Epsilon2: Second-level prediction error (ε^(2))
            - Alpha2: Second-level learning rate (α^(2))
            - FreeEnergy: Variational free energy
        """
        if self.add_second_level:
            self.mu2 = 0.0
            self.sigma2 = 1.0

        # Initialize history with first trial (no update)
        initial_data = {
            "beliefs": self.mu,
            "prediction_errors": 0.0,
            "learning_rates": 0.0,
            "uncertainties": self.sigma,
            "change_point_probs": 0.0,
            "variational_free_energy": 0.0,
        }

        if self.add_second_level:
            initial_data.update(  # does that need to be self.history?
                {
                    "mu2": self.mu2,
                    "epsilon2": 0.0,
                    "alpha2": 0.0,
                }
            )

        self.history = pd.DataFrame([initial_data])

        # Run updates for trials 0 to T-1
        for t in range(0, len(observations)):
            self.update(observations[t])

        output_columns = [
            "beliefs",
            "prediction_errors",
            "learning_rates",
            "variational_free_energy",
        ]

        output = self.history[output_columns].copy()

        output["updates"] = self.history["beliefs"].diff().shift(-1)

        return output

    def set_parameters_cma(self, theta: np.ndarray) -> None:
        """
        theta = [mu0, log_sigma0, log_obs_noise, log_w1, log_w2, logit_h]
        """
        mu0, log_sigma0, log_s, log_w1, log_w2, logit_h = map(float, theta)

        self.mu0 = mu0
        self.sigma0 = float(np.exp(log_sigma0))
        self.obs_noise_std = float(np.exp(log_s))
        self.w1_std = float(np.exp(log_w1))
        self.w2_std = float(np.exp(log_w2))
        self.hazard_rate = float(1.0 / (1.0 + np.exp(-logit_h)))  # sigmoid

        # variance conversion for internal use
        self.obs_noise_var = self.obs_noise_std**2
        self.w1_var = self.w1_std**2
        self.w2_var = self.w2_std**2

        self.mu = self.mu0
        self.sigma = self.sigma0

        if self.add_second_level:
            self.mu2 = 0.0
            self.sigma2 = 1.0

        # reset history
        self.history = pd.DataFrame(columns=self.history.columns)

    def objective_cma(self, observations: np.ndarray) -> float:
        """
        CMA-ES objective: negative total variational free energy over the sequence.

        Uses the per-trial free energy computed in 'update' via calc_variational_free_energy
        and stored in history. We sum (or average) this over all trials to get the objective value.

        CMA minimizes this objective, so we return the negative free energy (or negative average free energy).
        """
        # reset state and history for this sequence
        self.mu = self.mu0
        self.sigma = self.sigma0
        if self.add_second_level:
            self.mu2 = 0.0
            self.sigma2 = 1.0

        # clear history
        self.history = pd.DataFrame(columns=self.history.columns)

        # run the full sequency
        self.run(observations)
        F = self.history["variational_free_energy"].to_numpy(dtype=float)

        if not np.all(np.isfinite(F)):
            raise FloatingPointError(f"Non-finite free energy values encountered: {F}")

        total_F = np.sum(F)

        return float(-total_F)  # CMA minimizes, so we return negative free energy

    @staticmethod
    def decode_cma_theta(theta: np.ndarray) -> dict:
        """
        Map CMA parameter vector back to named, interpretable parameters.
        theta = [mu0, log_sigma0, log_obs_noise, log_w1, log_w2, logit_h]
        """
        mu0, log_sigma0, log_s, log_w1, log_w2, logit_h = map(float, theta)

        sigma0 = float(np.exp(log_sigma0))
        obs_noise = float(np.exp(log_s))
        w1 = float(np.exp(log_w1))
        w2 = float(np.exp(log_w2))
        h = float(1.0 / (1.0 + np.exp(-logit_h)))

        return {
            "mu0": mu0,
            "sigma0": sigma0,
            "obs_noise": obs_noise,
            "w1": w1,
            "w2": w2,
            "hazard_rate": h,
            "log_sigma0": log_sigma0,
            "log_obs_noise": log_s,
            "log_w1": log_w1,
            "log_w2": log_w2,
            "logit_h": logit_h,
        }

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    def plot_results(self, results_df, figsize=(14, 12)):
        """
        Visualize model performance and internal variables.

        Creates a 4-panel plot showing:
        1. Belief tracking vs true position
        2. Learning rate trajectory
        3. Change-point probability
        4. Second-level variables (if enabled)

        Parameters
        ----------
        results_df : pd.DataFrame
            Output from run() method
        figsize : tuple, optional
            Figure size (width, height)
        """
        n_plots = 4 if self.add_second_level else 3
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)

        # Plot 1: Belief tracking
        ax = axes[0]
        if "TruePosition" in results_df.columns:
            ax.plot(
                results_df["Trial"],
                results_df["TruePosition"],
                "k--",
                label="True Position",
                linewidth=2,
                alpha=0.7,
            )
        ax.plot(
            results_df["Trial"],
            results_df["Belief"],
            "b-",
            label="Model Belief (μ¹)",
            linewidth=1.5,
        )
        ax.fill_between(
            results_df["Trial"],
            results_df["Belief"] - results_df["Uncertainty"],
            results_df["Belief"] + results_df["Uncertainty"],
            alpha=0.2,
            color="blue",
            label="Uncertainty (±σ¹)",
        )
        ax.set_ylabel("Position")
        ax.set_title("Belief Tracking")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Learning rate
        ax = axes[1]
        ax.plot(
            results_df["Trial"],
            results_df["LearningRate"],
            "g-",
            linewidth=1.5,
            label="Learning Rate (α¹)",
        )
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.3, linewidth=1)
        ax.set_ylabel("Learning Rate")
        ax.set_title("Adaptive Learning Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])

        # Plot 3: Change-point probability
        ax = axes[2]
        ax.plot(
            results_df["Trial"],
            results_df["CPP"],
            "orange",
            linewidth=1.5,
            label="Change-Point Probability (Ω)",
        )
        ax.set_ylabel("Probability")
        ax.set_title("Change-Point Detection")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])

        # Plot 4: Second-level (if enabled)
        if self.add_second_level and "Mu2" in results_df.columns:
            ax = axes[3]
            ax.plot(
                results_df["Trial"],
                results_df["Mu2"],
                "purple",
                linewidth=1.5,
                label="Volatility (μ²)",
            )
            ax.set_ylabel("Log-odds")
            ax.set_xlabel("Trial")
            ax.set_title("Second-Level Representation (Volatility)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            axes[2].set_xlabel("Trial")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing ChangePointModel_Variational (CPM 2016 Adjusted)")
    logger.info("=" * 60)

    # Generate change-point environment
    logger.info("\n1. Testing with Change-Point Environment...")
    df_change_point = generate_change_point_environment(
        n_trials=400, oddball_hazard_rate=0.1, sigma=25, change_point_hazard_rate=0.1, seed=555
    )

    # Initialize CPM model with change-point environment
    cpm_model_change = ChangePointModelVariational(
        x=df_change_point["x"].values,
        mu0=df_change_point["x"].iloc[0],
        sigma0=25,
        obs_noise_std=25,
        w1_std=0.1,  # std dev (squares to 0.01 in variance)
        w2_std=30,  # std dev (squares to 900 in variance)
        h=0.1,
        add_second_level=True,
    )
    # Run model
    results_change = cpm_model_change.run(mu_true=df_change_point["mu"].values)

    # Calculate performance metrics
    abs_errors = np.abs(results_change["Belief"] - results_change["TruePosition"])
    mae = abs_errors.mean()

    # Calculate negative log-likelihood (lower is better)
    pred_errors = results_change["PredictionError"].values[1:]  # Skip first trial
    uncertainties = results_change["Uncertainty"].values[:-1]  # Use uncertainty from previous trial

    # Print summary
    logger.info("\nChange-Point Environment Results:")
    logger.info(f"  Mean learning rate: {results_change['LearningRate'].mean():.4f}")
    logger.info(f"  Max learning rate: {results_change['LearningRate'].max():.4f}")
    logger.info(f"  Mean CPP: {results_change['CPP'].mean():.4f}")
    logger.info(f"  Max CPP: {results_change['CPP'].max():.4f}")
    logger.info("\nPerformance Metrics:")
    logger.info(f"  MAE (mean absolute error): {mae:.4f}")

    # Plot
    logger.info("\nGenerating plots for change-point environment...")
    cpm_model_change.plot_results(results_change)

    # Generate random walk environment
    logger.info("\n2. Testing with Random Walk Environment...")
    df_random_walk = generate_random_walk_environment(
        n_trials=400,
        oddball_hazard_rate=0.1,
        sigma=25,
        drift_sigma=10,
        seed=555,
    )

    # Initialize CPM model with random walk environment
    cpm_model_walk = ChangePointModelVariational(
        x=df_random_walk["x"].values,
        mu0=df_random_walk["x"].iloc[0],
        sigma0=25,
        obs_noise_std=25,
        w1_std=3.16,  # std dev (squares to 10 drift steps in variance)
        w2_std=30,  # (of no significance in random walk environment) std dev (squared to 900 in variance)
        h=0.1,
        add_second_level=True,
    )

    # Run model
    results_walk = cpm_model_walk.run(mu_true=df_random_walk["mu"].values)

    # Calculate performance metrics
    abs_errors = np.abs(results_walk["Belief"] - results_walk["TruePosition"])
    mae = abs_errors.mean()

    # Calculate negative log-likelihood (lower is better)
    pred_errors = results_walk["PredictionError"].values[1:]  # Skip first trial
    uncertainties = results_walk["Uncertainty"].values[:-1]  # Use uncertainty from previous trial

    # Print summary
    logger.info("\nRandom Walk Environment Results:")
    logger.info(f"  Mean learning rate: {results_walk['LearningRate'].mean():.4f}")
    logger.info(f"  Max learning rate: {results_walk['LearningRate'].max():.4f}")
    logger.info(f"  Mean CPP: {results_walk['CPP'].mean():.4f}")
    logger.info(f"  Max CPP: {results_walk['CPP'].max():.4f}")
    logger.info("\nPerformance Metrics:")
    logger.info(f"  MAE (mean absolute error): {mae:.4f}")

    # Plot
    logger.info("\nGenerating plots for random walk environment...")
    cpm_model_walk.plot_results(results_walk)

    logger.info("\nDone!")
