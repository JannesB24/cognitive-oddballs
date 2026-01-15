"""
Nassar et al. (2016) Normative Model Implementation

This module implements the normative Bayesian learning model from Nassar et al. (2016)
for the helicopter task, sans flexible model variants. 
"""

import numpy as np
import pandas as pd
from scipy import stats


# Spec Functions

def prediction_error(X_t, B_t):
    """
    Calculate prediction error (surprise magnitude).
    
    Args:
        X_t: Observed outcome at trial t
        B_t: Belief (bucket placement) at trial t
    
    Returns:
        Prediction error δ_t
    """
    return X_t - B_t


def relative_uncertainty(sig_mu, sig_N):
    """
    Calculate relative uncertainty (learning rate component).
    
    τ_{t+1} = σ_μ² / (σ_μ² + σ_N²)
    
    Args:
        sig_mu: Standard deviation of predicted distribution over helicopter locations
        sig_N: Standard deviation of noise distribution
    
    Returns:
        Relative uncertainty τ
    """
    return sig_mu**2 / (sig_mu**2 + sig_N**2)


def predictive_variance(Omega_t, sigma_N, tau_t, delta_t):
    """
    Calculate predictive variance (estimation uncertainty).
    
    σ_μ² = Ω_t * σ_N² + (1 - Ω_t) * σ_N² * τ_t + Ω_t * (1 - Ω_t) * δ_t * (1 - τ_t)
    
    Args:
        Omega_t: Change-point probability at trial t
        sigma_N: Standard deviation of noise
        tau_t: Relative uncertainty at trial t
        delta_t: Prediction error at trial t
    
    Returns:
        Predictive variance σ_μ²
    """
    sigma_mu_sq = (
        Omega_t * (sigma_N ** 2)
        + (1 - Omega_t) * (sigma_N ** 2) * tau_t
        + Omega_t * (1 - Omega_t) * delta_t * (1 - tau_t)
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


def update_belief(B_t, alpha_t1, delta_t):
    """
    Update belief using delta rule.
    
    B_{t+1} = B_t + α_t * δ_t
    
    Args:
        B_t: Current belief at trial t
        alpha_t1: Learning rate at trial t+1
        delta_t: Prediction error at trial t
    
    Returns:
        Updated belief B_{t+1}
    """
    return B_t + alpha_t1 * delta_t


# Normative Model Class

class NormativeBaseModel:
    """
    Normative Bayesian learning model from Nassar et al. (2016).
    
    Implements optimal learning in dynamic environments with change points.
    The model dynamically adjusts learning rates based on:
    - Change-point probability (Ω): How likely a change just occurred
    - Relative uncertainty (τ): Balance between estimation and noise uncertainty
    
    Parameters:
        X: Array of observed bag drop positions (trial outcomes)
        sigma_sequence: Array specifying noise standard deviation for each trial
        H: Hazard rate (prior probability of change point), default=0.1
    """
    
    def __init__(self, X, sigma_sequence, H=0.1):
        # Store inputs
        self.X = X
        self.sigma_sequence = sigma_sequence
        self.n_trials = len(X)
        
        # Model parameters
        self.hazard_rate = H
        
        # Initial belief
        self.initial_belief = X[0]
        self.belief = X[0]
        
        # Get noise from first trial (will update per trial)
        self.sigma_n = sigma_sequence[0]
        self.sigma_n_squared = sigma_sequence[0] ** 2
        
        # State variables
        self.sigma_mu_squared = self.sigma_n_squared
        self.tau = 0.1  # Initial relative uncertainty
        self.alpha = 0.3  # Initial learning rate
        
        # History tracking
        self.history = {
            'beliefs': [],
            'prediction_errors': [],
            'learning_rates': [],
            'uncertainties': [],
            'change_point_probs': []
        }
    
    def update(self, t):
        """
        Update belief for trial t.
        
        Args:
            t: Trial index
        
        Returns:
            Updated belief value
        """
        # Get current trial's noise level
        self.sigma_n = self.sigma_sequence[t]
        self.sigma_n_squared = self.sigma_sequence[t] ** 2
        
        # 1. Prediction error
        delta = prediction_error(self.X[t], self.belief)
        
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
        
        Ω_{t+1} = (H/300) / (H/300 + N(δ|0, σ²/(1-τ)) * (1-H))
        
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
        self.history['beliefs'].append(self.belief)
        self.history['prediction_errors'].append(delta)
        self.history['learning_rates'].append(self.alpha)
        self.history['uncertainties'].append(self.tau)
        self.history['change_point_probs'].append(omega)
    
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
        self.belief = self.X[0]
        self.tau = 0.1
        self.history = {
            'beliefs': [],
            'prediction_errors': [],
            'learning_rates': [],
            'uncertainties': [],
            'change_point_probs': []
        }
        
        # Store first trial (no update, just initialization)
        self.history['beliefs'].append(self.belief)
        self.history['prediction_errors'].append(0.0)
        self.history['learning_rates'].append(0.0)
        self.history['uncertainties'].append(self.tau)
        self.history['change_point_probs'].append(0.0)
        
        # Run trials 1 through n_trials-1
        for t in range(1, self.n_trials):
            self.update(t)
        
        # Create DataFrame
        df = pd.DataFrame({
            "Trial": np.arange(1, self.n_trials + 1),
            "BagDrop": self.X,
            "Belief": self.history['beliefs'],
            "CPP": self.history['change_point_probs'],
            "RelUncertainty": self.history['uncertainties'],
            "LearningRate": self.history['learning_rates'],
            "PredictionError": self.history['prediction_errors']
        })
        
        # Add true position if provided
        if mu is not None:
            df.insert(1, "TruePosition", mu)
        
        return df
    
    
