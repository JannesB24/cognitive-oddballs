"""
Model recovery on oddball tasks
MLE-BIC vs Bayesian Inference (Laplace Approximation)

Based on Marković & Kiebel (2016)
"""


# TODO: refactor common code with eval.py
# TODO: add the other models
# TODO: add confusion matrices and parameter recovery plots


# Imports

from collections.abc import Callable

import numpy as np
from environments import generate_change_point_environment, generate_random_walk_environment
from models import ChangePointNassarModel

# Utilities


def set_seed(seed: int = 42):
    np.random.seed(seed)


# Response likelihood


def response_log_likelihood(responses, beliefs, sigma_r):
    """
    log p(r_t | mu_t, sigma_r)
    """
    residuals = responses - beliefs
    return -0.5 * np.sum((residuals / sigma_r) ** 2 + np.log(2 * np.pi * sigma_r**2))


# Core simulation loop


def run_model_on_environment(
    model_fn, observations, sigma_r, generate_responses=True, fixed_responses=None
):
    """
    Runs perceptual model and (optionally) response model.
    """

    model_fn.reset()

    beliefs = []
    responses = []
    prediction_errors = []
    updates = []

    for t, obs in enumerate(observations):
        mu = model_fn.predict()
        beliefs.append(mu)

        pe = obs - mu
        model_fn.update(obs)

        prediction_errors.append(pe)
        updates.append(model_fn.last_update)

        r = mu + np.random.randn() * sigma_r if generate_responses else fixed_responses[t]

        responses.append(r)

    return {
        "beliefs": np.asarray(beliefs),
        "responses": np.asarray(responses),
        "prediction_errors": np.asarray(prediction_errors),
        "updates": np.asarray(updates),
    }


# MLE grid search + BIC


def calculate_bic(k, n_trials, ll):
    return k * np.log(n_trials) - 2 * ll


def grid_search_mle(model_cls, param_grid, observations, responses, sigma_r):
    best_ll = -np.inf
    best_params = None

    for params in param_grid:
        model = model_cls(*params)

        outputs = run_model_on_environment(
            model, observations, sigma_r, generate_responses=False, fixed_responses=responses
        )

        ll = response_log_likelihood(responses, outputs["beliefs"], sigma_r)

        if ll > best_ll:
            best_ll = ll
            best_params = params

    return best_params, best_ll


# MAP estimation (Bayesian inference)


def grid_search_map(model_cls, param_grid, observations, responses, sigma_r):
    best_logpost = -np.inf
    best_params = None

    for params in param_grid:
        model = model_cls(*params)

        outputs = run_model_on_environment(
            model, observations, sigma_r, generate_responses=False, fixed_responses=responses
        )

        ll = response_log_likelihood(responses, outputs["beliefs"], sigma_r)
        lp = model.log_prior(params)

        logpost = ll + lp

        if logpost > best_logpost:
            best_logpost = logpost
            best_params = params

    return best_params, best_logpost


# Laplace approximation


def estimate_hessian(model_cls, map_params, observations, responses, sigma_r, eps=1e-4):
    k = len(map_params)
    hessian = np.zeros((k, k))

    def neg_log_post(params):
        model = model_cls(*params)
        outputs = run_model_on_environment(
            model, observations, sigma_r, generate_responses=False, fixed_responses=responses
        )
        ll = response_log_likelihood(responses, outputs["beliefs"], sigma_r)
        return -(ll + model.log_prior(params))

    for i in range(k):
        for j in range(k):
            ei = np.zeros(k)
            ei[i] = eps
            ej = np.zeros(k)
            ej[j] = eps

            hessian[i, j] = (
                neg_log_post(map_params + ei + ej)
                - neg_log_post(map_params + ei - ej)
                - neg_log_post(map_params - ei + ej)
                + neg_log_post(map_params - ei - ej)
            ) / (4 * eps**2)

    return hessian


def laplace_model_evidence(logpost_map, hessian, k):
    return logpost_map + 0.5 * k * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(hessian))


# Model recovery


def model_recovery_per_env(
    models: dict[str, Callable],
    true_params: dict[str, np.ndarray],
    param_grids: dict[str, np.ndarray],
    environment_fn: Callable,
    n_trials: int,
    sigma_r: float,
):
    results = {}

    for true_name, true_model_cls in models.items():
        # --- generate synthetic dataset ---
        true_model = true_model_cls(*true_params[true_name])
        observations = environment_fn(n_trials=n_trials)

        synth = run_model_on_environment(true_model, observations, sigma_r, generate_responses=True)

        responses = synth["responses"]

        results[true_name] = {}

        for fit_name, fit_model_cls in models.items():
            # ----- MLE + BIC -----
            best_params, best_ll = grid_search_mle(
                fit_model_cls, param_grids[fit_name], observations, responses, sigma_r
            )

            bic = calculate_bic(fit_model_cls.n_params, n_trials, best_ll)

            # ----- Bayesian (Laplace) -----
            map_params, logpost = grid_search_map(
                fit_model_cls, param_grids[fit_name], observations, responses, sigma_r
            )

            hessian = estimate_hessian(
                fit_model_cls, np.asarray(map_params), observations, responses, sigma_r
            )

            log_evidence = laplace_model_evidence(logpost, hessian, fit_model_cls.n_params)

            results[true_name][fit_name] = {
                "MLE": {"best_params": best_params, "loglik": best_ll, "BIC": bic},
                "Bayesian": {"MAP": map_params, "log_evidence": log_evidence},
            }

    return results


# Example experiment wrappers


def modelrec_changepoint():
    models = {"CPM": ChangePointNassarModel}

    true_params = {
        "CPM": np.array([0.1])  # example hazard
    }

    param_grids = {"CPM": np.linspace(0.01, 0.3, 30).reshape(-1, 1)}

    return model_recovery_per_env(
        models=models,
        true_params=true_params,
        param_grids=param_grids,
        environment_fn=generate_change_point_environment,
        n_trials=300,
        sigma_r=5.0,
    )


def modelrec_randomwalk():
    models = {"CPM": ChangePointNassarModel}

    true_params = {"CPM": np.array([0.1])}

    param_grids = {"CPM": np.linspace(0.01, 0.3, 30).reshape(-1, 1)}

    return model_recovery_per_env(
        models=models,
        true_params=true_params,
        param_grids=param_grids,
        environment_fn=generate_random_walk_environment,
        n_trials=300,
        sigma_r=5.0,
    )


# Main


if __name__ == "__main__":
    set_seed(42)

    cp_results = modelrec_changepoint()
    rw_results = modelrec_randomwalk()

    print("Changepoint recovery:")
    print(cp_results)

    print("\nRandom-walk recovery:")
    print(rw_results)
