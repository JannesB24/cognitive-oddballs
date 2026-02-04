#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from collections import Counter
from model_recovery.models import CPM, HGF
from model_recovery.evaluation.recovery import recover_one_environment
from model_recovery.evaluation.summary import confusion_matrix, param_recovery_stats
from model_recovery.data.utils import get_observations
from cognitive_oddballs.environments.change_point_oddball import generate_change_point_environment

def sample_true_params(model_name: str) -> np.ndarray:
    """Draw a random parameter set that spans the regimes used in the grids."""
    if model_name == "CPM":
        w1 = np.random.uniform(0.01, 0.3)
        w2 = np.random.choice([50.0, 200.0, 1000.0])
        h  = np.random.uniform(0.01, 0.3)
        return np.array([w1, w2, h])
    if model_name == "HGF":
        eta = 10 ** np.random.uniform(-4, -1)
        s   = 10 ** np.random.uniform(np.log10(1.0), np.log10(50.0 ** 2))
        return np.array([eta, s])
    raise KeyError(model_name)

def many_sims(
    n_sims: int,
    models: dict[str, Callable[..., Any]],
    grids: dict[str, np.ndarray],
    env_fn: Callable[..., Any],
    n_trials: int,
    sigma_r: float,
    decision_rule: str = "BIC",
) -> Tuple[Dict[str, Counter], dict]:
    """
    Run `n_sims` independent recoveries, accumulate a confusion matrix
    and keep the true↔recovered parameter pairs for the correctly identified
    simulations (used later for correlation statistics).
    """
    winners: Dict[str, Counter] = {m: Counter() for m in models}
    param_pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {m: [] for m in models}

    for i in range(n_sims):
        # --------------------------------------------------------------
        # 1️. draw a *different* true parameter set for each model
        # --------------------------------------------------------------
        true_params = {m: sample_true_params(m) for m in models}

        # --------------------------------------------------------------
        # 2️. run the full recovery routine (identical to the sanity‑check)
        # --------------------------------------------------------------
        res = recover_one_environment(
            models,
            true_params,
            grids,
            env_fn,
            n_trials,
            sigma_r,
        )

        # --------------------------------------------------------------
        # 3️. decide the winner for each true generator
        # --------------------------------------------------------------
        for true_name, fit_dict in res.items():
            scores = {
                fit_name: (info["MLE"]["BIC"] if decision_rule == "BIC"
                           else info["Bayesian"]["log_evidence"])
                for fit_name, info in fit_dict.items()
            }
            # BIC → minimise, evidence → maximise
            winner = min(scores, key=scores.get) if decision_rule == "BIC" else max(scores, key=scores.get)
            winners[true_name][winner] += 1

            # keep parameter recovery *only* when the winner matches the truth
            if winner == true_name:
                recovered = (fit_dict[true_name]["MLE"]["best_params"]
                             if decision_rule == "BIC"
                             else fit_dict[true_name]["Bayesian"]["MAP"])
                param_pairs[true_name].append((true_params[true_name], np.asarray(recovered)))

    return winners, param_pairs


if __name__ == "__main__":
    np.random.seed(1)

    models = {"CPM": CPM, "HGF": HGF}
    # --------------------------------------------------------------
    # Parameter grids (same as in the original script)
    # --------------------------------------------------------------
    w1_grid = np.linspace(0.05, 0.5, 6)
    w2_grid = np.array([50.0, 200.0, 1000.0])
    h_grid = np.linspace(0.01, 0.3, 6)
    cpm_grid = np.array(
        [(w1, w2, h) for w1 in w1_grid for w2 in w2_grid for h in h_grid],
        dtype=float,
    )
    eta_grid = np.logspace(-4, -1, 8)
    s_grid = np.logspace(np.log10(1.0), np.log10(50.0 ** 2), 8)
    hgf_grid = np.array([(e, s) for e in eta_grid for s in s_grid], dtype=float)

    grids = {"CPM": cpm_grid, "HGF": hgf_grid}

    winners, param_pairs = many_sims(
        n_sims=200,
        models=models,
        grids=grids,
        env_fn=generate_change_point_environment,
        n_trials=200,
        sigma_r=5.0,
        decision_rule="BIC",
    )

    # --------------------------------------------------------------
    # 4️. pretty printing
    # --------------------------------------------------------------
    confusion_matrix(winners)
    param_recovery_stats(param_pairs)