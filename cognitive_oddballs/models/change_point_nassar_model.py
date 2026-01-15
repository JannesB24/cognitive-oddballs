import numpy as np
import pandas as pd
from scipy import stats

class ChangePointNassarModel:
    """
    Base class for Nassar et al. (2016) models
    """
    def __init__(self, X, sigma_sequence, true_position, H=0.1,
                 uncertainty_scale=1.0, surprise_sensitivity=1.0):
        # Store inputs
        self.X = X
        self.sigma_sequence = sigma_sequence
        self.n_trials = len(X)
        self.true_position = true_position

        # Model parameters
        self.hazard_rate = H
        self.uncertainty_scale = uncertainty_scale
        self.surprise_sensitivity = surprise_sensitivity

        # Initial belief
        self.initial_belief = X[0]
        self.belief = X[0]

        # Get noise from first trial (will update per trial)
        self.sigma_n = sigma_sequence[0]
        self.sigma_n_squared = sigma_sequence[0] ** 2

        # State variables
        self.sigma_mu_squared = self.sigma_n_squared
        self.tau = 0.1  # Match your initial tau
        self.alpha = 0.3

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
        Update belief for trial t
        """
        def prediction_error(X_t, B_t):
            return X_t - B_t
        
        def relative_uncertainty(sig_mu,sig_N):
            return sig_mu**2 / (sig_mu**2 + sig_N**2)
        
        def predictive_variance(Omega_t, sigma_N, tau_t, delta_t):
            sigma_mu_sq = (
                Omega_t * (sigma_N ** 2)
                + (1 - Omega_t) * (sigma_N ** 2) * tau_t
                + Omega_t * (1 - Omega_t) * delta_t * (1 - tau_t))
            return sigma_mu_sq
        
        def learning_rate(omega_t1, tau_t1):
            return omega_t1 + (1 - omega_t1) * tau_t1
        
        def update_belief(B_t, alpha_t1, delta_t):
            return B_t + alpha_t1 * delta_t
        
        # Get current trial's noise level
        self.sigma_n = self.sigma_sequence[t]
        self.sigma_n_squared = self.sigma_sequence[t] ** 2

        # 1. Prediction error
        delta = prediction_error(self.X[t], self.belief)

        # 2. Change-point probability (with surprise sensitivity)
        omega = self._compute_change_point_prob(delta)

        # 3. Predictive variance
        sig_mu_sq = predictive_variance(omega, self.sigma_n, self.tau, delta)


        # 4. Update uncertainty
        # Estimation uncertainty divided by uncertainty scale
        sig_mu_sq /= self.uncertainty_scale
        # Update relative uncertainty
        self.tau = relative_uncertainty(np.sqrt(sig_mu_sq), self.sigma_n)

        # 5. Learning rate
        self.alpha = learning_rate(omega, self.tau)

        # 6. Update belief
        self.belief = update_belief(self.belief, self.alpha, delta)
        self.belief = np.clip(self.belief, 0, 300)

        # 7. Store history
        self._store_history(delta, omega)

        return self.belief

    def _compute_change_point_prob(self, delta):
        """Compute change-point probability with surprise sensitivity"""
        # Compute likelihood with surprise power
        var_no_cp = self.sigma_n_squared / (1 - self.tau)
        likelihood = stats.norm.pdf(delta, 0, np.sqrt(var_no_cp))
        likelihood_powered = likelihood ** self.surprise_sensitivity

        # Bayes rule
        num = self.hazard_rate / (300.0 ** self.surprise_sensitivity)
        den = num + likelihood_powered * (1 - self.hazard_rate)
        omega = num / den

        return np.clip(omega, 1e-6, 1 - 1e-6)

    def _store_history(self, delta, omega):
        """Store trial results"""
        self.history['beliefs'].append(self.belief)
        self.history['prediction_errors'].append(delta)
        self.history['learning_rates'].append(self.alpha)
        self.history['uncertainties'].append(self.tau)
        self.history['change_point_probs'].append(omega)

    def run(self):
        """
        Run model on full task sequence
        Returns DataFrame compatible with function-based approach
        """
        # Initialize with first trial
        self.belief = self.X[0]
        self.tau = 0.1

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
            "TruePosition": self.true_position,
            "BagDrop": self.X,
            "Belief": self.history['beliefs'],
            "CPP": self.history['change_point_probs'],
            "RelUncertainty": self.history['uncertainties'],
            "LearningRate": self.history['learning_rates'],
            "PredictionError": self.history['prediction_errors']
        })

        return df