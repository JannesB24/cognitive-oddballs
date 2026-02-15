"""
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

# Configs
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)

# so that we use same environments for each model during optimization
def generate_environments(environment_generator: Callable, n_simulations: int, n_trials: int):
    return [environment_generator(n_trials) for _ in range(n_simulations)]


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

    try:
        for env in envs: # env type: <class 'pandas.core.frame.DataFrame'>
            #for observations x in env:
            if "x" in env.columns:
                observations = env["x"].to_numpy(dtype=float)
            
                model = model_cls() # fresh model each run
                model.set_parameters_cma(theta)

                obj = model.objective_cma(observations)

                if not np.isfinite(obj):
                    raise FloatingPointError(f"Non-finite objective: {obj}")

                total_obj += float(obj)

            else:
                raise ValueError(
                    f"DataFrame environment has no 'x' column; "
                    f"columns={list(env.columns)}"
                )
        return total_obj / len(envs)
    
    except Exception as e:
        msg = (
            f"[CMA safeguard] {model_cls.__name__} failed for theta={theta}: "
            f"{type(e).__name__}: {e}"
        )
        logger.warning(msg)

        # large penalty so CMA-ES moves away from region
        return penalty    

def cma_optimization(cma_params: dict, envs: list[np.ndarray], seed: int = 42):
    """
    Returns:
        results: dict mapping model name -> {
            'theta_best': np.ndarray,
            'decoded': dict,
            'f_best': float,
        }
    """
    results = {}

    for model_cls, params in cma_params.items():
        logger.info(f"Optimizing {model_cls.__name__}...")

        objective = make_cma_objective(model_cls, envs)

        es_result = cma.fmin(
            objective,
            x0=params["x0"],
            sigma0=params["sigma0"],
            options={
                "bounds": params["bounds"],
                "maxfevals": params["maxfevals"],
                "verb_disp": params["verb_disp"],
                "seed": seed,
            },
        )

        theta_best = np.asarray(es_result[0], dtype=float)
        f_best = float(es_result[1])  # best function value

        # decode if available
        decoded = model_cls.decode_cma_theta(theta_best)

        results[model_cls.__name__] = {
            "theta_best": theta_best,
            "decoded": decoded,
            "f_best": f_best,
        }

    return results

def save_param_results(results: dict, env_type: str, filename: str) -> None:
    """
    results: output of cma_optimization
    env_type: "changepoint" / "randomwalk" etc.
    filename: e.g. "cma_params_changepoint.csv"
    """
    records = []
    for model_name, res in results.items():
        rec = {
            "env_type": env_type,
            "model": model_name,
            "f_best": res["f_best"],
        }
        # add decoded parameters with names
        for k, v in res["decoded"].items():
            rec[k] = float(v)
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    out_path = RESULTS_DIR / filename
    df.to_csv(out_path, index=False)
    logger.info("Saved CMA-ES results to %s", out_path)

def run_param_optimization(n_simulations: int = 1000, n_trials: int = 100, seed: int = 42):
    set_seed(seed)

    cp_envs = generate_environments(generate_change_point_environment, n_simulations, n_trials)
    rw_envs = generate_environments(generate_random_walk_environment, n_simulations, n_trials)

    cma_params_cmp = {
        # [mu0, log_sigma0, log_obs_noise_std, log_w1_std, log_w2_std, logit_h]
        "x0": [
            250.0,               # mu0
            np.log(10.0),        # sigma0
            np.log(10.0),        # s
            np.log(0.01),        # w1_std
            np.log(100.0),       # w2_std
            np.log(0.1 / 0.9),   # h = 0.1
        ],
        "bounds": (
            [  0.0,
            np.log(0.1),      # sigma0_min
            np.log(1.0),      # s_min
            np.log(1e-4),     # w1_min
            np.log(20.0),     # w2_min
            np.log(0.01 / 0.99),  # logit(h_min)
            ],
            [ 500.0,
            np.log(100.0),    # sigma0_max
            np.log(50.0),     # s_max
            np.log(1.0),      # w1_max
            np.log(500.0),    # w2_max
            np.log(0.5 / 0.5),  # logit(h_max) = 0.0
            ],
        ),
        "sigma0": 0.5,
        "maxfevals": 5000,
        "verb_disp": 100,
    }

    cma_params_hgf = {
        "x0": [
            250.0,             # mu1_0
            np.log(10.0),    # sig1_0 (variance)
            -4.0,            # mu2_0
            np.log(1.0),     # sig2_0 (variance)
            np.log(0.005),   # eta
            np.log(15.0**2), # s
        ],
        "bounds": (
            [  # lower
                0.0,              # mu1_0
                np.log(1e-3),     # sig1_0
                -10.0,            # mu2_0
                np.log(1e-3),     # sig2_0
                np.log(1e-5),     # eta
                np.log(1.0),      # s
            ],
            [  # upper
                500.0,            # mu1_0
                np.log(1e3),      # sig1_0
                10.0,             # mu2_0
                np.log(1e3),      # sig2_0
                np.log(1.0),      # eta
                np.log(1e4),      # s
            ],
        ),
        "sigma0": 0.5,
        "maxfevals": 5000,
        "verb_disp": 100,
    }

    cma_params_weber = {
        "x0": [
            np.log(5.0), # tonic volatility of node 0
            15, # initial mean of node 2 (volatility parent)
            40 # initial mean of node 3
        ],  
        "bounds": (
            [np.log(0.1), 0.0, 0.0],    # lower
            [np.log(100.0), 40.0, 80.0],  # upper
        ),
        "sigma0": 0.5,  # initial global step-size
        "maxfevals": 3000,  # maximum number of function evaluations
        "verb_disp": 100,  # verbosity level
    }

    cma_optimization_params = {
        ChangePointModelVariational: cma_params_cmp,
        HGFPaper2Gaussian: cma_params_hgf,
        WeberModel: cma_params_weber,
    }

    logger.info("Optimizing models on Change Point Environments")
    cp_results = cma_optimization(cma_optimization_params, cp_envs, seed=seed)
    logger.info("\nChange Point Environment Optimization Results:")
    for model_name, result in cp_results.items():
        logger.info(f"  {model_name}: {result}")

    save_param_results(cp_results, "changepoint", "cma_params_changepoint.csv")

    logger.info("\nOptimizing models on Random Walk Environments")
    rw_results = cma_optimization(cma_optimization_params, rw_envs, seed=seed)
    logger.info("\nRandom Walk Environment Optimization Results:")
    for model_name, result in rw_results.items():
        logger.info(f"  {model_name}: {result}")

    save_param_results(rw_results, "randomwalk", "cma_params_randomwalk.csv")

    return cp_results, rw_results

if __name__ == "__main__":
    set_seed(42)

    logging.basicConfig(
        level=logging.INFO,  
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    param_results_cp, param_results_rw = run_param_optimization(n_simulations=7, n_trials=100, seed=42) # Adjust to 1000, 100 to match Markovic and Kiebel (2016).