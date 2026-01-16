"""
Pipeline for evaluating models on oddball tasks

Environments (already implemented):
- generate_change_point_environment
- generate_random_walk_environment

Models:
- Two model types
- Two variants per model type

Evaluation inspired by:
- Nassar et al. (2010, 2016, 2019)
- Razmi and  Nassar (2022)
- Foucault et al. (2025)
"""


# Imports


import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable


# External imports (existing)


from environments import (
    generate_change_point_environment,
    generate_random_walk_environment
)

# from models.model_type_a import model_a_v1, model_a_v2
# from models.model_type_b import model_b_v1, model_b_v2


# Paths


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


# Utilities


def set_seed(seed: int = 42):
    np.random.seed(seed)



# Evaluation metrics


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


def evaluate_outputs(outputs: Dict) -> Dict:
    """
    Compute model-agnostic metrics.
    """
    lr = compute_apparent_learning_rate(
        outputs["updates"],
        outputs["prediction_errors"]
    )

    return {
        "learning_rate": lr,
        "mean_learning_rate": np.nanmean(lr),
        "prediction_errors": np.asarray(outputs["prediction_errors"]),
        "updates": np.asarray(outputs["updates"]),
    }



# Core simulation loop


def run_model_on_environment(
    model_fn: Callable,
    observations: np.ndarray
) -> Dict:
    """
    Runs a single model on a fixed observation sequence.

    Assumptions about model_fn:
    - model_fn.reset()
    - model_fn.predict()
    - model_fn.update(observation)
    - model_fn.last_update
    """

    model_fn.reset()

    outputs = {
        "prediction_errors": [],
        "updates": [],
    }

    for obs in observations:
        pred = model_fn.predict()
        pe = obs - pred

        model_fn.update(obs)

        outputs["prediction_errors"].append(pe)
        outputs["updates"].append(model_fn.last_update)

    return evaluate_outputs(outputs)



# Experiment runner


def run_experiment(
    environment_fn: Callable,
    models: Dict[str, Callable],
    n_trials: int,
    experiment_name: str
) -> Dict:
    """
    Run all models on a single environment.
    """

    observations = environment_fn(n_trials=n_trials)

    results = {}

    for model_name, model_fn in models.items():
        results[model_name] = run_model_on_environment(
            model_fn,
            observations
        )

    return results



# Experiment 1:
# Changepoint oddball


def experiment_changepoint():
    models = {
        "ModelA_v1": None,  # model_a_v1(...)
        "ModelA_v2": None,
        "ModelB_v1": None,
        "ModelB_v2": None,
    }

    return run_experiment(
        environment_fn=generate_change_point_environment,
        models=models,
        n_trials=1000,
        experiment_name="changepoint_oddball",
    )



# Experiment 2:
# Random-walk oddball


def experiment_randomwalk():
    models = {
        "ModelA_v1": None,
        "ModelA_v2": None,
        "ModelB_v1": None,
        "ModelB_v2": None,
    }

    return run_experiment(
        environment_fn=generate_random_walk_environment,
        models=models,
        n_trials=1000,
        experiment_name="randomwalk_oddball",
    )



# Plotting


def plot_learning_rate_vs_error(results: Dict, title: str):
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


# Main


if __name__ == "__main__":
    set_seed(42)

    results_cp = experiment_changepoint()
    results_rw = experiment_randomwalk()

    plot_learning_rate_vs_error(
        results_cp,
        "Changepoint oddball environment"
    )

    plot_learning_rate_vs_error(
        results_rw,
        "Random-walk oddball environment"
    )
