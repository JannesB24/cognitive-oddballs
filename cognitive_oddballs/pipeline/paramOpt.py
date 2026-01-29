"""
Docstring for cognitive_oddballs.pipeline.paramOpt

Perform parameter optimisation for cognitive oddball models using CMA-ES.
"""

from collections.abc import Callable
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cma

from cognitive_oddballs.environments.change_point_oddball import generate_change_point_environment
from cognitive_oddballs.environments.random_walk_oddball import generate_random_walk_environment
from cognitive_oddballs.models.change_point_model_variational import ChangePointModelVariational
from cognitive_oddballs.models.hgf.hgf2_gaussian import HGFPaper2Gaussian
from cognitive_oddballs.models.model import Model
from cognitive_oddballs.models.weber_model import WeberModel
from cognitive_oddballs.utils import set_seed

# Configs
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


logger = logging.getLogger(__name__)

# so that we use same environments for each model during optimization
def generate_environments(environment_generator: Callable, n_envs: int, n_trials: int):
    return [environment_generator(n_trials) for _ in range(n_envs)]

# format cma-es expects
def make_cma_objective(model_cls: type[Model], envs: list[np.ndarray]):
    """
    Returns a function f(theta) suitable for cma.fmin,
    with model class and environments closed over.
    """
    def obj(theta: np.ndarray) -> float:
        return objective_function_cma_theta(theta, model_cls, envs)
    return obj

def objective_function_cma_theta(
    theta: np.ndarray,
    model_cls: type[Model],
    envs: pd.DataFrame,
    penalty: float = 1e12,   # big penalty in case of failure
) -> float:
    """
    CMA-ES objective over a set of environments, for a given model class.
    Robust version: catch numerical failures and NaNs and return a penalty.

    theta:       parameter vector for CMA-ES
    model_cls:   class of the model (e.g. ChangePointModelVariational)
    envs:        pd.df containg info of pre-generated environment
    """
    total_obj = 0.0
    for env in envs: # env type: <class 'pandas.core.frame.DataFrame'>
        #for observations x in env:
        if "x" in env.columns:
            observations = env["x"].to_numpy(dtype=float)
        
            model = model_cls() # fresh model each run
            model.set_parameters_cma(theta)

            obj = model.objective_cma(observations)
            total_obj += float(obj)

    return total_obj / len(envs)

    """ try:
    for env in envs:
        # --- convert environment to 1D float array of observations ---
        if isinstance(env, pd.DataFrame):
            # adjust 'x' if your generator uses a different column name
            if "x" in env.columns:
                observations = env["x"].to_numpy(dtype=float)
            else:
                raise ValueError(
                    f"DataFrame environment has no 'x' column; "
                    f"columns={list(env.columns)}"
                )
        else:
            # assume it's already array-like
            observations = np.asarray(env, dtype=float)

        model = model_cls()  # fresh model each run
        model.set_parameters_cma(theta)

        obj = model.objective_cma(observations)

        if not np.isfinite(obj):
            raise FloatingPointError(f"Non-finite objective: {obj}")

        total_obj += float(obj)
    """

    """try:
        for observations in envs:
            model = model_cls() # fresh model each run
            model.set_parameters_cma(theta)

            obj = model.objective_cma(observations)

            if not np.isfinite(obj):
                raise FloatingPointError(f"Non-finite objective: {obj}")

            total_obj += float(obj)

        return total_obj / len(envs)

    except Exception as e:
        msg = (
            f"[CMA safeguard] {model_cls.__name__} failed for theta={theta}: "
            f"{type(e).__name__}: {e}"
        )
        logger.warning(msg)

        # large penalty so CMA-ES moves away from region
        return penalty"""

        
def cma_optimization(cma_params: dict, envs: list[np.ndarray], seed: int = 42):
    """
    cma_params: dict mapping model class -> dict of CMA config
    envs: list of observation arrays (same set used for all models in this call)
    """
    optimal_thetas = {}

    for model_cls, params in cma_params.items():
        logger.info(f"Optimizing {model_cls.__name__}...")

        # need objective function with only theta as parameters
        objective = make_cma_objective(model_cls, envs)

        es_result = cma.fmin(
            objective,
            x0=params["x0"],
            sigma0=params["sigma0"], # initial global step-size
            options={
                "bounds": params["bounds"],
                "maxfevals": params["maxfevals"], # limit evaluations
                "verb_disp": params["verb_disp"], # verbosity
                # "popsize": 16, # optional: control population size
                "seed": seed, # for reproducibility
            },
        )

        theta_best = es_result[0]  # best CMA-ES parameter vector
        optimal_thetas[model_cls.__name__] = theta_best

    return optimal_thetas    


def run_param_optimization(n_envs: int = 1000, n_trials: int = 100, seed: int = 42):
    set_seed(seed)

    cp_envs = generate_environments(generate_change_point_environment, n_envs, n_trials)
    rw_envs = generate_environments(generate_random_walk_environment, n_envs, n_trials)

    # TODO: Define proper parameter settings for each model
    cma_params_cmp = {
        # [log_obs_noise, log_w1, log_w2, logit_h]
        "x0": [np.log(10.0), np.log(0.01), np.log(100.0), np.log(0.1 / 0.9)],
        "bounds": (
            [np.log(1.0),  np.log(1e-4), np.log(1.0),   -5.0],  # lower
            [np.log(50.0), np.log(1.0),  np.log(1e4),    5.0],  # upper
        ),
        "sigma0": 0.5,  # initial global step-size
        "maxfevals": 5000,  # maximum number of function evaluations
        "verb_disp": 1,  # verbosity level
    }

    cma_params_hgf = {
        # have parameters: mu1_init, sig1_init, mu2_init, sig2_init, eta, s
        # --> which ones to optimize?
        # initial guess: log_eta, log_s, mu2_0, log_sig2_0
        "x0": [np.log(0.01), np.log(15.0**2), -4.0, np.log(1.0)],
        "bounds": (
            [np.log(1e-5), np.log(1.0), -10.0, np.log(1e-3)],   # lower
            [np.log(1.0),  np.log(1e4),  10.0, np.log(1e2)],    # upper
        ),
        "sigma0": 0.5,  # initial global step-size
        "maxfevals": 5000,  # maximum number of function evaluations
        "verb_disp": 100,  # verbosity level
    }

    cma_params_weber = {
        "x0": [np.log(3.0), np.log(5.0), np.log(5.0)],  # initial guesses
        "bounds": (
            [np.log(0.1), np.log(0.1), np.log(0.1)],    # lower
            [np.log(100.0), np.log(100.0), np.log(100.0)],  # upper
        ),
        "sigma0": 0.5,  # initial global step-size
        "maxfevals": 5000,  # maximum number of function evaluations
        "verb_disp": 1,  # verbosity level
    }

    # TODO: auskommentiert for testing, toggle that back when done
    cma_optimization_params = {
        #ChangePointModelVariational: cma_params_cmp,
        #HGFPaper2Gaussian: cma_params_hgf,
        WeberModel: cma_params_weber,
    }

    logger.info("Optimizing models on Change Point Environments")
    cp_results = cma_optimization(cma_optimization_params, cp_envs)
    logger.info("\nChange Point Environment Optimization Results:")
    for model_name, result in cp_results.items():
        logger.info(f"  {model_name}: {result}")

    logger.info("\nOptimizing models on Random Walk Environments")
    rw_results = cma_optimization(cma_optimization_params, rw_envs)
    logger.info("\nRandom Walk Environment Optimization Results:")
    for model_name, result in rw_results.items():
        logger.info(f"  {model_name}: {result}")

    return cp_results, rw_results