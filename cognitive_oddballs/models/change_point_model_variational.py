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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from cognitive_oddballs.environments.change_point_oddball import generate_change_point_environment
from cognitive_oddballs.environments.random_walk_oddball import generate_random_walk_environment
from cognitive_oddballs.models.model import Model


class ChangePointModelVariational(Model):
    """
    Change Point Model using Variational Bayesian Inference.

    Implements the switching-state-space formulation with variational
    Bayesian inference from Marković & Kiebel (2016), Section 4.1.2.

    The model assumes the environment switches between:
    - Stability: x_t = x_{t-1} + √w₁ · noise
    - Change-point: x_t = √w₂ · noise

    Parameters
    ----------
    x : array-like
        Observed outcomes (e.g., bag drop positions)
    mu0 : float
        Initial belief about hidden state (μ₀¹)
    sigma0 : float
        Initial uncertainty (σ₀¹)
    obs_noise : float
        Observation noise standard deviation (s)
    w1 : float
        Diffusion rate during stability periods (w₁)
        Typical value: 0.01 (small drift)
    w2 : float
        Change-point variance (w₂)
        Typical value: 100-1000 (large jumps)
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
    >>> model = ChangePointModel_Variational(
    ...     x=observations,
    ...     mu0=250,
    ...     sigma0=10,
    ...     obs_noise=10,
    ...     w1=0.01,
    ...     w2=100,
    ...     h=0.1
    ... )
    >>> results = model.run()
    >>> print(results[['Trial', 'Belief', 'LearningRate']].head())
    """

    def __init__(self, mu0, sigma0, obs_noise, w1, w2, h, add_second_level=True):
        # ===== Perceptual free parameters =====
        self.mu0 = mu0  # μ₀¹ - Initial belief
        self.sigma0 = sigma0  # σ₀¹ - Initial uncertainty
        self.obs_noise = obs_noise  # s - Observation noise
        self.w1 = w1  # w₁ - Stability diffusion
        self.w2 = w2  # w₂ - Change-point variance
        self.hazard_rate = h  # h - Hazard rate

        # ===== First-level latent states =====
        self.mu = mu0  # μ^(1) - Posterior expectation
        self.sigma = sigma0  # σ^(1) - Posterior std dev

        # ===== Second-level states (for HGF comparability) =====
        self.add_second_level = add_second_level
        if add_second_level:
            self.mu2 = 0.0  # μ^(2) - Volatility (log-odds of Ω)
            self.sigma2 = 1.0  # σ^(2) - Second-level uncertainty

        # ===== History tracking =====
        columns = [
            "beliefs",  # μ^(1)
            "prediction_errors",  # δ_t
            "learning_rates",  # α^(1)
            "uncertainties",  # σ^(1)
            "change_point_probs",  # Ω_t
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

    # --------------------------------------------------
    # Second-level transformations (for HGF comparability)
    # --------------------------------------------------

    def _omega_to_mu2(self, omega, scaling=1.0):
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

    def _mu2_to_omega(self, mu2, scaling=1.0):
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

    def _change_point_probability(self, delta):
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
        var_stability = self.sigma**2 + self.w1 + self.obs_noise**2
        like_stability = stats.norm.pdf(delta, 0.0, np.sqrt(var_stability))

        # Likelihood under change-point
        # x_t is drawn from a wide distribution (large w2)
        var_change = self.obs_noise**2 + self.w2
        like_change = stats.norm.pdf(delta, 0.0, np.sqrt(var_change))

        # Bayes rule with hazard rate as prior
        numerator = self.hazard_rate * like_change
        denominator = numerator + (1 - self.hazard_rate) * like_stability

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
        # 1. Prediction error
        delta = observation - self.mu

        # 2. Change-point probability (Bayes rule)
        omega = self._change_point_probability(delta)

        # 3. Update posterior uncertainty (inverse variance form)
        # This is the key equation that links change-points to learning
        inv_sigma_squared = (1 - omega) / (self.sigma**2 + self.w1) + 1 / self.obs_noise**2
        sigma_new = 1 / np.sqrt(inv_sigma_squared)

        # 4. Learning rate (ratio of posterior to observation uncertainty)
        alpha = sigma_new / self.obs_noise
        alpha = np.clip(alpha, 0.0, 1.0)  # Ensure valid range

        # 5. Update posterior expectation (delta rule)
        mu_new = self.mu + alpha * delta
        mu_new = np.clip(mu_new, 0, 500)  # Clip to valid screen range

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

        # 8. Store history
        self._store_history(delta, omega, alpha, epsilon2, alpha2)

    def _store_history(self, delta, omega, alpha, epsilon2=0.0, alpha2=0.0):
        """Store trial results in history."""
        row_data = {
            "beliefs": self.mu,
            "prediction_errors": delta,
            "learning_rates": alpha,
            "uncertainties": self.sigma,
            "change_point_probs": omega,
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

    # --------------------------------------------------

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
        }

        if self.add_second_level:
            initial_data.update(
                {
                    "mu2": self.mu2,
                    "epsilon2": 0.0,
                    "alpha2": 0.0,
                }
            )

        self.history = pd.DataFrame([initial_data])

        # Run updates for trials 1 to T-1
        for t in range(1, len(observations)):
            self.update(observations[t])

        # Create output DataFrame
        # df_dict = {
        #     "Trial": np.arange(1, len(observations) + 1),
        #     "BagDrop": observations,
        #     "Belief": self.history["beliefs"].values,
        #     "CPP": self.history["change_point_probs"].values,
        #     "Uncertainty": self.history["uncertainties"].values,
        #     "LearningRate": self.history["learning_rates"].values,
        #     "PredictionError": self.history["prediction_errors"].values,
        # }

        # if self.add_second_level:
        #     df_dict.update(
        #         {
        #             "Mu2": self.history["mu2"].values,
        #             "Epsilon2": self.history["epsilon2"].values,
        #             "Alpha2": self.history["alpha2"].values,
        #         }
        #     )

        # df = pd.DataFrame(df_dict)

        # beliefs (Nassar: Belief; Weber: x_0_expected_mean)
        # observations/targets (location bag drops: Nassar: BagDrop; Weber: x_0_mean; for all models, depends on environment)
        # responses (are computed in eval)
        # prediction_errors (Nassar: PredictionError, Weber: x_0_prediction_error)
        # updates (new belief; Nassar: Belief of t+1; Weber: x_0_expected_mean of t+1)
        # log_likelihood (computed in eval)

        output_columns = [
            # "Beliefs",
            # "Prediction Errors",
            # "Updates",
            # "Log Likelihoods",
        ]

        output = self.history[["beliefs"]]

        rename_dict = {"beliefs": "raw_responses"}

        # output["Beliefs"] = self.history["beliefs"].values
        # output["Prediction Errors"] = self.history["prediction_errors"].values
        # output["Updates"] = self.history["beliefs"].shift(-1).values
        # output["Log Likelihoods"] = self.history["change_point_probs"].values

        return output.rename(columns=rename_dict)

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
    print("=" * 60)
    print("Testing ChangePointModel_Variational (CPM 2016 Adjusted)")
    print("=" * 60)

    # Generate change-point environment
    print("\n1. Testing with Change-Point Environment...")
    df_change_point = generate_change_point_environment(
        n_trials=400, oddball_hazard_rate=0.1, sigma=25, change_point_hazard_rate=0.1, seed=555
    )

    # Initialize CPM model with change-point environment
    cpm_model_change = ChangePointModelVariational(
        x=df_change_point["x"].values,
        mu0=df_change_point["x"].iloc[0],  # Start at first observation
        sigma0=25,  # Match environment noise
        obs_noise=25,  # Match sigma
        w1=0.01,  # Small stability drift
        w2=1000,  # Large change-point variance
        h=0.1,  # Match environment hazard rate
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
    print("\nChange-Point Environment Results:")
    print(f"  Mean learning rate: {results_change['LearningRate'].mean():.4f}")
    print(f"  Max learning rate: {results_change['LearningRate'].max():.4f}")
    print(f"  Mean CPP: {results_change['CPP'].mean():.4f}")
    print(f"  Max CPP: {results_change['CPP'].max():.4f}")
    print("\nPerformance Metrics:")
    print(f"  MAE (mean absolute error): {mae:.4f}")

    # Plot
    print("\nGenerating plots for change-point environment...")
    cpm_model_change.plot_results(results_change)

    # Generate random walk environment
    print("\n2. Testing with Random Walk Environment...")
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
        obs_noise=25,
        w1=10,  # Higher drift for random walk
        w2=1000,
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
    print("\nRandom Walk Environment Results:")
    print(f"  Mean learning rate: {results_walk['LearningRate'].mean():.4f}")
    print(f"  Max learning rate: {results_walk['LearningRate'].max():.4f}")
    print(f"  Mean CPP: {results_walk['CPP'].mean():.4f}")
    print(f"  Max CPP: {results_walk['CPP'].max():.4f}")
    print("\nPerformance Metrics:")
    print(f"  MAE (mean absolute error): {mae:.4f}")

    # Plot
    print("\nGenerating plots for random walk environment...")
    cpm_model_walk.plot_results(results_walk)

    print("\nDone!")
