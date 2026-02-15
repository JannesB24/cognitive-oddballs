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

import json
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


# Evaluation metrics
def rmse(posterior_belief: np.ndarray, hidden_state: np.ndarray) -> float:
    """Root Mean Square Error (RMSE) between the posterior belief and the hidden state."""
    return np.sqrt(np.mean((posterior_belief - hidden_state) ** 2))


def log_likelihood(predictions: np.ndarray, targets: np.ndarray, noise_std: float) -> float:
    residuals = targets - predictions
    return -0.5 * np.sum((residuals / noise_std) ** 2 + np.log(2 * np.pi * noise_std**2))


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

    return output


# Experiment runner


def run_experiment(
    environment_fn: Callable,
    models: dict[str, Model],
    n_trials: int,
    response_noise_std: float = 1.0,
    n_simulations: int = 1,  # Default to 1 simulation if not specified
) -> dict:
    """
    Run all models on a single environment.
    """

    results = {}

    for model_name, model in models.items():
        model_sim_results = []

        for _ in range(n_simulations):
            # Generate a new environment for each simulation
            environment = environment_fn(n_trials=n_trials, oddball_hazard_rate=0.1, seed=None)

            response_model = GaussianResponseModel(response_noise_std)
            outputs = run_model_on_environment(model, response_model, environment)

            # Calculate RMSE: How well does the model's mechanism update the belief for timestep t
            total_rmse = rmse(outputs["beliefs"].to_numpy()[1:], environment["mu"].to_numpy())

            total_free_energy = np.sum(outputs["variational_free_energy"])

            model_sim_result = {
                "environment": environment,
                "model_outputs": outputs,
                "rmse": total_rmse,
                "free_energy": total_free_energy,
            }

            model_sim_results.append(model_sim_result)

        results[model_name] = model_sim_results

    return results


# Experiment 1:
# Changepoint oddball

# mu0 = 250, ln_sigma0 = -5, ln_s = 0.1, ln_w1 = 0.01, ln_w2 = 8.0, ln_h_div_1_h= -3.0,


def experiment_changepoint(n_trials: int, n_simulations: int):
    models: dict[str, Model] = {
        "CPM_OPTIM": ChangePointModelVariational(
            mu0=246.4316841,
            sigma0=0.1,
            obs_noise_std=38.3689664562455,
            w1_std=0.0001,
            w2_std=285.1587011,
            h=0.164325442,
        ),
        "CPM": ChangePointModelVariational(
            mu0=250, sigma0=50, obs_noise_std=25, w1_std=0.1, w2_std=30, h=0.1
        ),
        "HGF": HGFPaper2Gaussian(eta=0.005, s=15.0, mu1_init=250.0),
        "gHGF": WeberModel(),
    }

    return run_experiment(
        environment_fn=generate_change_point_environment,
        models=models,
        n_trials=n_trials,
        n_simulations=n_simulations,
    )


# Experiment 2:
# Random-walk oddball


def experiment_randomwalk(n_trials: int, n_simulations: int):
    models: dict[str, Model] = {
        "CPM_OPTIM": ChangePointModelVariational(
            mu0=246.30069841471,
            sigma0=0.1,
            obs_noise_std=28.3168277436353,
            w1_std=0.0001,
            w2_std=266.426872348825,
            h=0.184822388990129,
        ),
        "CPM": ChangePointModelVariational(
            mu0=250, sigma0=50, obs_noise_std=25, w1_std=0.1, w2_std=30, h=0.1
        ),
        "HGF": HGFPaper2Gaussian(eta=0.005, s=15.0, mu1_init=250.0),
        "gHGF": WeberModel(),
    }

    return run_experiment(
        environment_fn=generate_random_walk_environment,
        models=models,
        n_trials=n_trials,
        n_simulations=n_simulations,
    )


# Visualization


def create_comparison_boxplot(
    results_dict: dict, models: list[str], colors: dict[str, str], save_path: Path | None = None
):
    """
    Create a two-panel boxplot comparing CPM, HGF, and gHGF across environments.

    Args:
        results_dict: Dict with structure:
            {
                'changepoint': {'Model 1': [...],'Model 2': [...], ...]},
                'randomwalk': {'Model 1': [...],'Model 2': [...], ...]}
            }
            where each list contains simulation results with 'rmse' and 'free_energy' keys
        save_path: Optional path to save the figure

    LLM generated code inspired by:
    - Markovic and Kiebel (2016)
    """
    from matplotlib.patches import Patch

    environments = ["changepoint", "randomwalk"]
    # models = ["CPM_OPTIM", "CPM", "HGF", "gHGF"]
    # colors = {"CPM": "#0066cc", "HGF": "#3399ff", "gHGF": "#66b3ff"}

    # Extract metrics from results
    def extract_metric(metric_name):
        return {
            env: {
                model: [sim[metric_name] for sim in results_dict[env].get(model, [])]
                for model in models
            }
            for env in environments
        }

    rmse_data = extract_metric("rmse")
    free_energy_data = extract_metric("free_energy")

    # Create figure
    fig, (ax_rmse, ax_fe) = plt.subplots(1, 2, figsize=(12, 5))
    ax_rmse.text(-0.1, 1.05, "A", transform=ax_rmse.transAxes, fontsize=16, fontweight="bold")
    ax_fe.text(-0.1, 1.05, "B", transform=ax_fe.transAxes, fontsize=16, fontweight="bold")

    # Common boxplot parameters
    box_props = {"patch_artist": True, "widths": 0.6}

    # Helper function to plot data on axis
    def plot_metric(ax, data, ylabel):
        positions = []
        for env_idx, env in enumerate(environments):
            base_pos = env_idx * 3  # Space between environment groups
            for model_idx, model in enumerate(models):
                if data[env][model]:
                    pos = base_pos + model_idx * 0.8
                    positions.append(pos)
                    _ = ax.boxplot(
                        [data[env][model]],
                        positions=[pos],
                        boxprops={"facecolor": colors[model], "alpha": 0.7},
                        medianprops={"color": "black", "linewidth": 1.5},
                        flierprops={
                            "marker": "D",
                            "markerfacecolor": colors[model],
                            "markersize": 4,
                            "alpha": 0.5,
                        },
                        **box_props,
                    )

        # Set labels and grid
        env_centers = [i * 3 + 0.4 for i in range(len(environments))]
        ax.set_xticks(env_centers)
        ax.set_xticklabels(environments)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plot_metric(ax_rmse, rmse_data, "RMSE")
    plot_metric(ax_fe, free_energy_data, "free-energy")

    # Add legend to first subplot
    legend_elements = [
        Patch(facecolor=colors[m], alpha=0.7, edgecolor="black", label=m) for m in models
    ]
    ax_rmse.legend(
        handles=legend_elements,
        loc="upper left",
        frameon=True,
        fontsize=11,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()
    return fig


if __name__ == "__main__":
    n_trials = 100
    n_simulations = 1000

    results_cp = experiment_changepoint(n_trials=n_trials, n_simulations=n_simulations)
    results_rw = experiment_randomwalk(n_trials=n_trials, n_simulations=n_simulations)

    # Organize results for visualization
    results_dict = {
        "changepoint": {"CPM": [], "HGF": [], "gHGF": []},
        "randomwalk": {"CPM": [], "HGF": [], "gHGF": []},
    }

    # Extract CPM, HGF, and gHGF from changepoint
    for model_name in ["CPM", "HGF", "gHGF"]:
        if model_name in results_cp:
            results_dict["changepoint"][model_name] = results_cp[model_name]

    # Extract CPM, HGF, and gHGF from randomwalk
    for model_name in ["CPM", "HGF", "gHGF"]:
        if model_name in results_rw:
            results_dict["randomwalk"][model_name] = results_rw[model_name]

    with (RESULTS_DIR / "results.json").open("w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    # Create visualization
    create_comparison_boxplot(
        results_dict,
        models=["CPM", "HGF", "gHGF"],
        colors={"CPM": "#0052cc", "HGF": "#0084ff", "gHGF": "#40b3ff"},
        save_path=FIGURES_DIR / "model_comparison.png",
    )

    results_dict = {
        "changepoint": {"CPM": [], "CPM_OPTIM": []},
        "randomwalk": {"CPM": [], "CPM_OPTIM": []},
    }

    for model_name in ["CPM", "CPM_OPTIM"]:
        if model_name in results_cp:
            results_dict["changepoint"][model_name] = results_cp[model_name]

        if model_name in results_rw:
            results_dict["randomwalk"][model_name] = results_rw[model_name]

    with (RESULTS_DIR / "results_optim.json").open("w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    # Create visualization
    create_comparison_boxplot(
        results_dict,
        models=["CPM", "CPM_OPTIM"],
        colors={"CPM": "#0052cc", "CPM_OPTIM": "#0084ff"},
        save_path=FIGURES_DIR / "model_comparison_optim.png",
    )
