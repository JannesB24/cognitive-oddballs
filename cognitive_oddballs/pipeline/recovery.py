"""
Doing model recovery for the different models on the oddball task using BIC. 

Additional assumptions (compared to eval.py) about model_fn:
    - model_fn.convert_to_loglike(prediction_error)
    - model_fn.n_params  

Requires information about fitted model parameters and parameter space of each model.

"""
import numpy as np
from typing import Dict, Callable

from environments import (
    generate_change_point_environment,
    generate_random_walk_environment
)

from eval import(
    experiment_changepoint,
    experiment_randomwalk,
    run_experiment,
    run_model_on_environment
)

def set_seed(seed: int = 42):
    np.random.seed(seed)


def calculate_bic(num_params: int, num_data_points: int, ll: float) -> float:
    """Calculates Bayesian Information Criterion to be used in model comparison.

    Args:
      num_params (int): Number of free parameters that the model has
      num_data_points (int): Number of data points the model has been fitted to
      ll (float): Maximum log likelihood estimation for the model given data
    Returns:
      bic (float): Bayesian Information Criterion
    """
    bic = num_params * np.log(num_data_points) - 2 * ll
    return bic


def grid_search(
        model_cls: Callable,
        param_grid: np.ndarray,
        observations: np.ndarray,
    ):
    """
    Grid search for given model class over given parameter space, based on log likelihood.

    Args:
        model_cls (Callable): Model class to run grid search on.
        param_grid (np.ndarray): Parameter space to be used for grid search.
        observations (np.ndarray): Dataset to be fitted.
    Returns:
        best_params: Parameter configuration with highest log likelihood for given dataset.
        best_ll: Max. log likelihood value achieved during grid search.
    """
    best_ll = -np.inf
    best_params = None
    for params in param_grid:
        model_fn = model_cls(*params)
        outputs = run_model_on_environment(model_fn, observations)
        pe = outputs["prediction_errors"]
        ll = model_fn.ll_convert_to_loglike(pe)

        if ll > best_ll:
            best_ll = ll
            best_params = params
    
    return best_params, best_ll


def model_recovery_per_synth_set(
        synth: np.ndarray,
        models: Dict[str, Callable],
        n_trials: int,
        param_grids: Dict[str, Callable]
    ) -> Dict: 
    """
    Exectues the model recovery for a dictionary of models and compares them via their BIC score.

    Additional assumptions about model_fn:
    - model_fn.convert_to_loglike(prediction_error)
    - model_fn.n_params

    Args:
        synth (np.ndarray): Synthetic dataset to run model recovery on.
        models (Dict[str, Callable]): Dictionary of model names and instances to evaluate.
        n_trials (int): Amount of trials in dataset.
        param_grids (Dict[str, Callable]): Parameter space to evaluate in gridsearch.
    Returns:
        bic_scores (Dict[str, Callable]): BIC score for best-performing parameter combination of every model class on given synthetic dataset.
    """
    # initialize storage for BICs
    bic_scores = {}

    # iterate through models 
    for name, model_cls in models.items():
        # identify best parameters for model and obs
        best_params, best_ll = grid_search(
            model_cls, param_grids[name], synth
        )

        # calculat BIC score for that combination
        bic = calculate_bic(
            num_params=model_cls.n_params, 
            num_data_points=n_trials, 
            ll=best_ll
        )

        bic_scores[name] = {
            "bic": bic,
            "best_params": best_params,
            "best_ll": best_ll
        }

    return bic_scores

def model_recovery_per_env(
        models: Dict[str, Callable],
        true_params: Dict[str, Callable],
        environment_fn: Callable,
        n_trials: int,
        param_grids: Dict[str, Callable]
    ) -> Dict:
    """
    Evaluates BIC of all given models for a given environment.
    Generates synthetic data for the given amount of trials for every model in dicitonary, 
    given "ground-truth" model parameters and environment instance.
    Calls model_recovery_per_synth on that dataset.

    Args:
        models (Dict[str, Callable]): Dictionary of model names and instances to evaluate.
        true_params (Dict[str, Callable]): Parameters used as basis for synthetic data generation.
        environment_fn(Callable): Environment instance, either random walk or changepoint.
        n_trials (int): Amount of trials to generate/in dataset.
        param_grids (Dict[str, Callable]): Parameter space to evaluate in gridsearch.
    Returns:
        bic_per_env (Dict[str, Callable]): Dictionary of BIC scores for given environment -- find BIC scores of different models given name of ground truth model.
    """
    bic_per_env = {}

    for true_name, true_cls in models.items():
        # generate synthetic dataset
        true_model = true_cls(*true_params[true_name])
        observations = environment_fn(n_trials=n_trials)
        synthetic_data = run_model_on_environment(true_model, observations)
        synthetic_data = np.asarray(synthetic_data["updates"]) # TODO, might be synthetic_data["prediciton errors"] or sth else entirely

        # perform actual model recovery
        # TODO: this dictionary entry is going to be somewhat misleading, since it saves the different models given the true model, should find other solution
        bic_per_env[true_name] = model_recovery_per_synth_set(synth=synthetic_data, models=models, n_trials=n_trials, param_grids=param_grids)

    return bic_per_env


def modelrec_changepoint(
        # TODO
):
    models = {
        "ModelA_v1": None,
        "ModelA_v2": None,
        "ModelB_v1": None,
        "ModelB_v2": None,
    }

    return model_recovery_per_env(
        environment_fn=generate_change_point_environment,
        models=models,
        # TODO: finish function call
    )


def modelrec_randomwalk(
        # TODO
):
    models = {
        "ModelA_v1": None,
        "ModelA_v2": None,
        "ModelB_v1": None,
        "ModelB_v2": None,
    }

    return model_recovery_per_env(
        environment_fn=generate_random_walk_environment,
        models=models,
        # TODO
    )

# Main
if __name__ == "__main__":
    set_seed(42)

    results_modelrec_cp = modelrec_changepoint(
        # TODO
    )
    results_modelrec_rw = modelrec_randomwalk(
        # TODO
    )