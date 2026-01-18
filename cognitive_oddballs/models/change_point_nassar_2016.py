"""
Nassar et al. (2016) Normative Model Implementation

This module implements the normative Bayesian learning model from Nassar et al. (2016)
for the helicopter task, sans flexible model variants.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# Normative Model Class
class ChangePointNassarModel:
    """
    Normative Bayesian learning model from Nassar et al. (2016).

    Implements optimal learning in dynamic environments with change points.
    The model dynamically adjusts learning rates based on:
    - Change-point probability (Ω): how likely a change just occurred
    - Relative uncertainty (τ): Balance between estimation and noise uncertainty

    Parameters:
        x: Array of observed bag drop positions (trial outcomes)
        sigma_sequence: Array specifying noise standard deviation for each trial
        h: Hazard rate (prior probability of change point), default=0.1
    """

    def __init__(self, x, sigma_sequence, h=0.1):
        # Store inputs
        self.x = x
        self.sigma_sequence = sigma_sequence
        self.n_trials = len(x)

        # Model parameters
        self.hazard_rate = h

        # Initial belief
        self.initial_belief = x[0]
        self.belief = x[0]

        # Get noise from first trial (will update per trial)
        self.sigma_n = sigma_sequence[0]
        self.sigma_n_squared = sigma_sequence[0] ** 2

        # State variables
        self.sigma_mu_squared = self.sigma_n_squared
        self.tau = 0.1  # Initial relative uncertainty
        self.alpha = 0.3  # Initial learning rate

        # History tracking
        self.history = {
            "beliefs": [],
            "prediction_errors": [],
            "learning_rates": [],
            "uncertainties": [],
            "change_point_probs": [],
        }

    def update(self, t):
        """
        Update belief for trial t.

        Args:
            t: Trial index

        Returns:
            Updated belief value
        """
        # Spec Functions

        def prediction_error(x_t, b_t):
            """
            Calculate prediction error (surprise magnitude).
            Args:
                x_t: Observed outcome at trial t
                b_t: Belief (bucket placement) at trial t
            Returns:
                Prediction error δ_t
            """
            return x_t - b_t

        def relative_uncertainty(sig_mu, sig_n):
            """
            Calculate relative uncertainty (learning rate component).

            τ_{t+1} = σ_μ² / (σ_μ² + σ_N²)

            Args:
                sig_mu: Standard deviation of predicted distribution over helicopter locations
                sig_n: Standard deviation of noise distribution
            Returns:
                Relative uncertainty τ
            """
            return sig_mu**2 / (sig_mu**2 + sig_n**2)

        def predictive_variance(omega_t, sigma_n, tau_t, delta_t):
            """
            Calculate predictive variance (estimation uncertainty).

            σ_μ² = Ω_t * σ_N² + (1 - Ω_t) * σ_N² * τ_t + Ω_t * (1 - Ω_t) * δ_t * (1 - τ_t)

            Args:
                omega_t: Change-point probability at trial t
                sigma_n: Standard deviation of noise
                tau_t: Relative uncertainty at trial t
                delta_t: Prediction error at trial t

            Returns:
                Predictive variance σ_μ²
            """
            sigma_mu_sq = (
                omega_t * (sigma_n**2)
                + (1 - omega_t) * (sigma_n**2) * tau_t
                + omega_t * (1 - omega_t) * delta_t * (1 - tau_t)
            )
            return sigma_mu_sq

        def learning_rate(omega_t1, tau_t1):
            """
            Calculate learning rate from change-point probability and uncertainty.

            α_t = Ω_t + τ_t * (1 - Ω_t)

            Args:
                omega_t1: Change-point probability at trial t+1
                tau_t1: Relative uncertainty at trial t+1

            Returns:
                Learning rate α_t
            """
            return omega_t1 + (1 - omega_t1) * tau_t1

        def update_belief(b_t, alpha_t1, delta_t):
            """
            Update belief using delta rule.

            B_{t+1} = b_t + α_t * δ_t

            Args:
                b_t: Current belief at trial t
                alpha_t1: Learning rate at trial t+1
                delta_t: Prediction error at trial t

            Returns:
                Updated belief B_{t+1}
            """
            return b_t + alpha_t1 * delta_t

        # Get current trial's noise level
        self.sigma_n = self.sigma_sequence[t]
        self.sigma_n_squared = self.sigma_sequence[t] ** 2

        # 1. Prediction error
        delta = prediction_error(self.x[t], self.belief)

        # 2. Change-point probability
        omega = self._compute_change_point_prob(delta)

        # 3. Predictive variance
        sig_mu_sq = predictive_variance(omega, self.sigma_n, self.tau, delta)

        # 4. Update relative uncertainty
        self.tau = relative_uncertainty(np.sqrt(sig_mu_sq), self.sigma_n)

        # 5. Learning rate
        self.alpha = learning_rate(omega, self.tau)

        # 6. Update belief
        self.belief = update_belief(self.belief, self.alpha, delta)
        self.belief = np.clip(self.belief, 0, 500)  # Clip to valid range

        # 7. Store history
        self._store_history(delta, omega)

        return self.belief

    def _compute_change_point_prob(self, delta):
        """
        Compute change-point probability using Bayes rule.

        Ω_{t+1} = (h/300) / (h/300 + N(δ|0, σ²/(1-τ)) * (1-h))

        Args:
            delta: Prediction error

        Returns:
            Change-point probability Ω
        """
        # Compute likelihood of observation under no-change-point hypothesis
        var_no_cp = self.sigma_n_squared / (1 - self.tau)
        likelihood = stats.norm.pdf(delta, 0, np.sqrt(var_no_cp))

        # Bayes rule
        num = self.hazard_rate / 300.0
        den = num + likelihood * (1 - self.hazard_rate)
        omega = num / den

        return np.clip(omega, 1e-6, 1 - 1e-6)

    def _store_history(self, delta, omega):
        """Store trial results in history."""
        self.history["beliefs"].append(self.belief)
        self.history["prediction_errors"].append(delta)
        self.history["learning_rates"].append(self.alpha)
        self.history["uncertainties"].append(self.tau)
        self.history["change_point_probs"].append(omega)

    def run(self, mu=None):
        """
        Run model on full task sequence.

        Args:
            mu: Optional array of true helicopter positions (for DataFrame output)

        Returns:
            DataFrame with trial-by-trial results containing:
                - Trial: Trial number
                - TruePosition: True helicopter position (if mu provided)
                - BagDrop: Observed bag drop position
                - Belief: Model's belief about helicopter position
                - CPP: Change-point probability
                - RelUncertainty: Relative uncertainty
                - LearningRate: Learning rate
                - PredictionError: Prediction error (surprise)
        """
        # Reset for fresh run
        self.belief = self.x[0]
        self.tau = 0.1
        self.history = {
            "beliefs": [],
            "prediction_errors": [],
            "learning_rates": [],
            "uncertainties": [],
            "change_point_probs": [],
        }

        # Store first trial (no update, just initialization)
        self.history["beliefs"].append(self.belief)
        self.history["prediction_errors"].append(0.0)
        self.history["learning_rates"].append(0.0)
        self.history["uncertainties"].append(self.tau)
        self.history["change_point_probs"].append(0.0)

        # Run trials 1 through n_trials-1
        for t in range(1, self.n_trials):
            self.update(t)

        # Create DataFrame
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

        # Add true position if provided
        if mu is not None:
            df.insert(1, "TruePosition", mu)

        return df
    
    def plot_results(self, results_df, env_df=None, noise_switch_trial=None, zoom_start=180, zoom_end=220):
        """
        Create visualization of model performance.

        Args:
            results_df: DataFrame from run() method
            env_df: Optional environment DataFrame with 'is_oddball' column
            noise_switch_trial: Trial number where noise switches (optional, for shading)
            zoom_start: Start trial for zoom-in plot
            zoom_end: End trial for zoom-in plot
        """
        plt.figure(figsize=(14, 10))
    
        # Plot 1: Trial vs Screen Position
        plt.subplot(3, 1, 1)

        # Plot true position if available
        if 'TruePosition' in results_df.columns:
            plt.plot(results_df['Trial'], results_df['TruePosition'], 
                    'k--', label='True Helicopter', linewidth=2)

        # Plot model belief
        plt.plot(results_df['Trial'], results_df['Belief'], 
                'b-', label='Model Belief', linewidth=2)
    
        # Mark oddball trials if environment data provided
        if env_df is not None and 'is_oddball' in env_df.columns:
            oddball_trials = env_df[env_df['is_oddball']]['trial'] + 1  # +1 because Trial starts at 1
            oddball_beliefs = results_df[results_df['Trial'].isin(oddball_trials)]['Belief']
            plt.scatter(oddball_trials, oddball_beliefs, 
                    color='red', s=100, marker='x', linewidths=3,
                    label='Oddball Trials', zorder=5)
    
        # Add noise shading if switch point provided
        if noise_switch_trial is not None:
            plt.axvspan(0, noise_switch_trial, color='green', alpha=0.1, label='Low Noise')
            plt.axvspan(noise_switch_trial, len(results_df), color='red', alpha=0.1, label='High Noise')
    
        plt.ylabel("Position (0-500)")
        plt.title("Belief vs True Helicopter Position")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        # Plot 2: Surprise and Uncertainty
        plt.subplot(3, 1, 2)
        plt.plot(results_df['Trial'], results_df['PredictionError'], 
                color='orange', label='Surprise (δ)', linewidth=1.5)
        plt.plot(results_df['Trial'], results_df['RelUncertainty'], 
                color='purple', label='Relative Uncertainty (τ)', linewidth=1.5)
    
        # Mark oddball trials
        if env_df is not None and 'is_oddball' in env_df.columns:
            for trial in oddball_trials:
                plt.axvline(trial, color='red', linestyle=':', alpha=0.3, linewidth=1)
    
        # Add noise shading if switch point provided
        if noise_switch_trial is not None:
            plt.axvspan(0, noise_switch_trial, color='green', alpha=0.1)
            plt.axvspan(noise_switch_trial, len(results_df), color='red', alpha=0.1)
    
        plt.ylabel("Model Estimates")
        plt.xlabel("Trial")
        plt.title("Surprise and Relative Uncertainty per Trial")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        # Plot 3: Zoom-in
        plt.subplot(3, 1, 3)
    
        # Get zoom data
        zoom_mask = (results_df['Trial'] >= zoom_start) & (results_df['Trial'] <= zoom_end)
        trials_zoom = results_df['Trial'][zoom_mask]
    
        # Normalize for visual clarity
        pred_error_zoom = np.abs(results_df['PredictionError'][zoom_mask])
        pred_error_norm = pred_error_zoom / np.max(pred_error_zoom) if np.max(pred_error_zoom) > 0 else pred_error_zoom
    
        rel_unc_zoom = results_df['RelUncertainty'][zoom_mask]
        rel_unc_norm = rel_unc_zoom / np.max(rel_unc_zoom) if np.max(rel_unc_zoom) > 0 else rel_unc_zoom
    
        plt.plot(trials_zoom, pred_error_norm, color='orange', 
                label='Surprise (δ, normalized)', linewidth=2)
        plt.plot(trials_zoom, rel_unc_norm, color='purple', 
                label='Relative Uncertainty (τ, normalized)', linewidth=2)
    
        # Mark oddball trials in zoom
        if env_df is not None and 'is_oddball' in env_df.columns:
            zoom_oddballs = [t for t in oddball_trials if zoom_start <= t <= zoom_end]
            for trial in zoom_oddballs:
                plt.axvline(trial, color='red', linestyle=':', alpha=0.5, linewidth=2)
    
        # Add noise shading if switch point provided
        if noise_switch_trial is not None:
            if zoom_start < noise_switch_trial < zoom_end:
                plt.axvspan(zoom_start, noise_switch_trial, color='green', alpha=0.1, label='Low Noise')
                plt.axvspan(noise_switch_trial, zoom_end, color='red', alpha=0.1, label='High Noise')
    
        plt.ylabel("Normalized Values (0-1)")
        plt.xlabel("Trial")
        plt.title(f"Zoom-in: Surprise and Relative Uncertainty (Trials {zoom_start}-{zoom_end})")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()

    