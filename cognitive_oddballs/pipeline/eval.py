"""
Pipeline for evaluating models on oddball tasks.

Environments (already implemented):
- generate_change_point_environment
- generate_random_walk_environment

Perceptual Models based on Hierarchical Gaussian Filter (Mathys et al. 2011, 2014) and
Change Point Model (Nassar et al. 2010, 2016):

- Two model types
- Two variants per model type

Response Model as defined in Markovic and Kiebel (2016)

Evaluation inspired by:
- Markovic and Kiebel (2016)
- Nassar et al. (2010, 2016, 2019)
- Razmi and  Nassar (2022)
- Foucault et al. (2025)


Evaluation Metrics:
    Similar to the model performance evaluation by Markovic and Kiebel (2016)
    we use RMSE and Variational Free Energy (VFE) as evaluation metrics.
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cognitive_oddballs.environments.change_point_oddball import generate_change_point_environment
from cognitive_oddballs.environments.random_walk_oddball import generate_random_walk_environment
from cognitive_oddballs.models.change_point_model_variational import ChangePointModelVariational
from cognitive_oddballs.models.hgf.hgf2_gaussian import HGFPaper2Gaussian
from cognitive_oddballs.models.model import Model
from cognitive_oddballs.models.weber_model import WeberModel

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Utilities
def set_seed(seed: int = 42):
    np.random.seed(seed)


# Evaluation metrics
def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Square Error (RMSE) between predictions and targets."""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def log_likelihood(predictions: np.ndarray, targets: np.ndarray, noise_std: float) -> float:
    residuals = targets - predictions
    return -0.5 * np.sum((residuals / noise_std) ** 2 + np.log(2 * np.pi * noise_std**2))


def compute_apparent_learning_rate(updates, prediction_errors):
    """
    Apparent learning rate (Nassar et al., Foucault et al.)

    alpha_t = update_t / prediction_error_t
    """
    updates = np.asarray(updates)
    pes = np.asarray(prediction_errors)

    lr = np.full_like(updates, np.nan, dtype=float)
    valid = pes != 0
    lr[valid] = updates[valid] / pes[valid]

    return lr


# Response Model
class GaussianResponseModel:
    r"""
    r_t = \mu_t + \epsilon
    \epsilon \sim \mathcal{N}(0, \sigma_r^2)
    """

    def __init__(self, response_noise_std: float):
        self.sigma = response_noise_std

    def sample(self, raw_response: float) -> float:
        return raw_response + np.random.randn() * self.sigma

    # def log_likelihood(self, response: float, belief_mean: float) -> float:
    #     return -0.5 * ((response - belief_mean) / self.sigma) ** 2 - np.log(
    #         np.sqrt(2 * np.pi) * self.sigma
    #     )


def evaluate_outputs(outputs: dict) -> dict:
    """
    Compute model-agnostic metrics.
    """
    lr = compute_apparent_learning_rate(outputs["updates"], outputs["prediction_errors"])

    return {
        "learning_rate": lr,
        "mean_learning_rate": np.nanmean(lr),
        "prediction_errors": np.asarray(outputs["prediction_errors"]),
        "updates": np.asarray(outputs["updates"]),
    }


# Core simulation loop
def run_model_on_environment(
    model: Model, response_model: GaussianResponseModel, environments: pd.DataFrame
) -> dict:
    """
    Runs perceptual + response model on a sequence
    """
    observations = environments["x"].to_numpy()
    output = model.run(observations)

    response_model = GaussianResponseModel(0.1)
    output["responses"] = output["beliefs"].apply(lambda response: response_model.sample(response))

    # TODO: Add log likelihoods if needed

    return output


# Experiment runner


def run_experiment(
    environment_fn: Callable,
    models: dict[str, Model],
    n_trials: int,
    response_noise_std: float = 5.0,
    n_simulations: int = 1,  # Default to 1 simulation if not specified
) -> dict:
    """
    Run all models on a single environment.
    """

    results = {}

    for model_name, model in models.items():
        # Initialize arrays to collect data across simulations
        all_learning_rates = np.zeros((n_simulations, n_trials))
        all_prediction_errors = np.zeros((n_simulations, n_trials))
        all_updates = np.zeros((n_simulations, n_trials))
        all_beliefs = np.zeros((n_simulations, n_trials))
        all_responses = np.zeros((n_simulations, n_trials))
        all_loglik = np.zeros(n_simulations)
        all_rmse = np.zeros(n_simulations)

        for sim in range(n_simulations):
            # Generate a new environment for each simulation
            environment = environment_fn(n_trials=n_trials)

            response_model = GaussianResponseModel(response_noise_std)
            outputs = run_model_on_environment(model, response_model, environment)

            lr = compute_apparent_learning_rate(outputs["updates"], outputs["prediction_errors"])

            # Store the results for this simulation
            # all_learning_rates[sim] = lr
            # all_prediction_errors[sim] = outputs["prediction_errors"]
            # all_updates[sim] = outputs["updates"]
            # all_beliefs[sim] = outputs["beliefs"]
            # all_responses[sim] = outputs["responses"]
            # all_loglik[sim] = np.sum(outputs["log_likelihoods"])
            # all_rmse[sim] = rmse(outputs["beliefs"], outputs["responses"])

        # results[model_name] = {
        #     "learning_rate": all_learning_rates,
        #     "prediction_errors": all_prediction_errors,
        #     "updates": all_updates,
        #     "beliefs": all_beliefs,
        #     "responses": all_responses,
        #     "log_likelihood": all_loglik,
        #     "rmse": all_rmse,
        # }

    return results


# Experiment 1:
# Changepoint oddball


def experiment_changepoint():
    models: dict[str, Model] = {
        "CPM": ChangePointModelVariational(
            mu0=250, sigma0=50, obs_noise_std=25, w1_std=0.1, w2_std=30, h=0.1
        ),
        "gHGF": WeberModel(node4=True, node_4_type="volatility_parent", n4_p=3.0),
        "HGF": HGFPaper2Gaussian(
            eta=0.005, s=15.0**2, mu1_init=0.0, sig1_init=10.0, mu2_init=-4.0, sig2_init=1.0
        ),
    }

    return run_experiment(
        environment_fn=generate_change_point_environment,
        models=models,
        n_trials=1000,
        n_simulations=1,
    )


# Experiment 2:
# Random-walk oddball


def experiment_randomwalk():
    models = {
        "OG": ChangePointNassarModel(x=[250], sigma_sequence=[25]),
        "CPM": ChangePointModelVariational(
            mu0=250,  # Start at center
            sigma0=25,
            obs_noise_std=25,
            w1_std=3.16,  # Higher drift for random walk
            w2_std=30,  # Does not matter for RW
            h=0.1,
            add_second_level=True,
        ),
        "gHGF": WeberModel(node4=True, node_4_type="volatility_parent", n4_p=3.0),
        "HGF": HGFPaper2Gaussian(
            eta=0.005, s=15.0**2, mu1_init=0.0, sig1_init=10.0, mu2_init=-4.0, sig2_init=1.0
        ),
    }

    return run_experiment(
        environment_fn=generate_random_walk_environment,
        models=models,
        n_trials=1000,
        n_simulations=10,
    )


# Plotting


def plot_learning_rate_vs_error(results: dict, title: str):
    """
    Replicates Nassar / Foucault style plots:
    learning rate as a function of |prediction error|
    """
    plt.figure(figsize=(6, 5))

    for model_name, metrics in results.items():
        pe = np.abs(metrics["prediction_errors"])
        lr = metrics["learning_rate"]

        plt.scatter(pe, lr, s=10, alpha=0.4, label=model_name)

    plt.xlabel("|Prediction error|")
    plt.ylabel("Apparent learning rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.savefig(FIGURES_DIR / f"learning_rate_vs_error_{title.replace(' ', '_')}.png")


# Main


if __name__ == "__main__":
    set_seed(42)

    results_cp = experiment_changepoint()
    # results_rw = experiment_randomwalk()

    plot_learning_rate_vs_error(results_cp, "Changepoint oddball environment")

    # plot_learning_rate_vs_error(results_rw, "Random-walk oddball environment")


# beliefs (Nassar: Belief; Weber: x_0_expected_mean)
# observations/targets (location bag drops: Nassar: BagDrop; Weber: x_0_mean; for all models, depends on environment)
# responses (are computed in eval)
# prediction_errors (Nassar: PredictionError, Weber: x_0_prediction_error)
# updates (new belief; Nassar: Belief of t+1; Weber: x_0_expected_mean of t+1)
# log_likelihood (computed in eval)
