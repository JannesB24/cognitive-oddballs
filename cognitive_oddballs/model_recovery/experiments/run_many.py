#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_many.py
===========

Runs many independent model‑recovery simulations, builds a confusion matrix,
collects parameter‑recovery statistics and finally produces the three classic
figures from Marković & Kiebel (2016).

The script now imports the plotting helpers from ``model_recovery.plots`` and
writes the images to *figures/<environment_name>/*.
"""

from __future__ import annotations

import os
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from model_recovery.models import CPM, HGF
from model_recovery.evaluation.recovery import recover_one_environment
from model_recovery.evaluation.summary import confusion_matrix, param_recovery_stats
from model_recovery.data.utils import get_observations
from cognitive_oddballs.environments.change_point_oddball import (
    generate_change_point_environment,
)

# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------
from model_recovery.plots import (
    plot_confusion_matrix,
    plot_parameter_recovery,
    plot_score_distributions,
    make_all_plots,
)


# ----------------------------------------------------------------------
# 1️.  Helper that draws a random true‑parameter set
# ----------------------------------------------------------------------
def sample_true_params(model_name: str) -> np.ndarray:
    """Draw a random parameter set that spans the regimes used in the grids."""
    if model_name == "CPM":
        w1 = np.random.uniform(0.01, 0.3)
        w2 = np.random.choice([50.0, 200.0, 1000.0])
        h = np.random.uniform(0.01, 0.3)
        return np.array([w1, w2, h])
    if model_name == "HGF":
        eta = 10 ** np.random.uniform(-4, -1)
        s = 10 ** np.random.uniform(np.log10(1.0), np.log10(50.0 ** 2))
        return np.array([eta, s])
    raise KeyError(model_name)


# ----------------------------------------------------------------------
# 2️.  Core Monte‑Carlo routine – now also returns the raw score differences
# ----------------------------------------------------------------------
def many_sims(
    n_sims: int,
    models: dict[str, Callable[..., Any]],
    grids: dict[str, np.ndarray],
    env_fn: Callable[..., Any],
    n_trials: int,
    sigma_r: float,
    decision_rule: str = "BIC",
) -> Tuple[Dict[str, Counter], dict, Dict[str, Dict[str, List[float]]]]:
    """
    Run ``n_sims`` independent recoveries.

    Returns
    -------
    winners      : dict[true_model] → Counter(recovered_model)
    param_pairs  : dict[model] → list[(true_params, recovered_params)]
    score_diffs  : dict[true_model][competing_model] → list of score differences
                  (BIC: fit – true;   Laplace: true – fit)
    """
    # ------------------------------------------------------------------
    # Containers for the three outputs
    # ------------------------------------------------------------------
    winners: Dict[str, Counter] = {m: Counter() for m in models}
    param_pairs: dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
        m: [] for m in models
    }
    # 🔧 NEW: store the per‑simulation score differences
    score_diffs: Dict[str, Dict[str, List[float]]] = {
        true_m: {fit_m: [] for fit_m in models if fit_m != true_m}
        for true_m in models
    }

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    for i in range(n_sims):
        # --------------------------------------------------------------
        # 1️. draw a *different* true‑parameter set for each model
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
            # ----- collect the BIC / log‑evidence for every competing model -----
            scores = {
                fit_name: (
                    info["MLE"]["BIC"]
                    if decision_rule == "BIC"
                    else info["Bayesian"]["log_evidence"]
                )
                for fit_name, info in fit_dict.items()
            }

            # ----- pick the winner ------------------------------------------------
            # BIC → minimise, evidence → maximise
            winner = (
                min(scores, key=scores.get)
                if decision_rule == "BIC"
                else max(scores, key=scores.get)
            )
            winners[true_name][winner] += 1

            # ----- keep parameter recovery only for correctly identified fits -----
            if winner == true_name:
                recovered = (
                    fit_dict[true_name]["MLE"]["best_params"]
                    if decision_rule == "BIC"
                    else fit_dict[true_name]["Bayesian"]["MAP"]
                )
                param_pairs[true_name].append(
                    (true_params[true_name], np.asarray(recovered))
                )

            # ----- store score differences for the histogram plot ---------------
            for comp_name in models:
                if comp_name == true_name:
                    continue
                if decision_rule == "BIC":
                    # ΔBIC = BIC(comp) – BIC(true)   (positive → true favoured)
                    diff = scores[comp_name] - scores[true_name]
                else:
                    # Δlog‑evidence = logev(true) – logev(comp)
                    diff = scores[true_name] - scores[comp_name]
                score_diffs[true_name][comp_name].append(diff)

    return winners, param_pairs, score_diffs


# ----------------------------------------------------------------------
# 3️.  Main entry‑point – runs the simulation and produces the figures
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(1)

    # ------------------------------------------------------------------
    # Model definitions
    # ------------------------------------------------------------------
    models = {"CPM": CPM, "HGF": HGF}

    # ------------------------------------------------------------------
    # Parameter grids (exactly the same as in the original script)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Run the Monte‑Carlo recovery
    # ------------------------------------------------------------------
    n_sims = 200                # feel free to increase this for a final paper
    env_name = "changepoint"
    env_fn = generate_change_point_environment

    winners, param_pairs, score_diffs = many_sims(
        n_sims=n_sims,
        models=models,
        grids=grids,
        env_fn=env_fn,
        n_trials=200,
        sigma_r=5.0,
        decision_rule="BIC",    # can be "laplace" as well
    )

    # ------------------------------------------------------------------
    # 4️.  Textual summaries (as before)
    # ------------------------------------------------------------------
    confusion_matrix(winners)
    param_recovery_stats(param_pairs)

    # ------------------------------------------------------------------
    # 5️.  Visualise the results
    # ------------------------------------------------------------------
    # Folder where all PNG files will be stored
    out_dir = Path("figures") / env_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # (a) Confusion matrix heat‑map
    # ------------------------------------------------------------------
    plot_confusion_matrix(winners, save_dir=out_dir, title="Confusion matrix (BIC)")

    # ------------------------------------------------------------------
    # (b) Parameter‑recovery scatter plots
    # ------------------------------------------------------------------
    plot_parameter_recovery(
        param_pairs,
        save_dir=out_dir,
        title="Parameter recovery (true vs. recovered)",
    )

    # ------------------------------------------------------------------
    # (c) Distribution of model‑selection scores
    # ------------------------------------------------------------------
    plot_score_distributions(
        score_diffs,
        decision_rule="BIC",
        save_dir=out_dir,
        title="Δ BIC distribution across simulations",
    )

    # ------------------------------------------------------------------
    # (d) One‑liner that returns the three Figure objects (optional)
    # ------------------------------------------------------------------
    # cm_fig, pr_fig, sd_fig = make_all_plots(
    #     winners,
    #     param_pairs,
    #     score_diffs,                 # works because the function accepts the same dict
    #     decision_rule="BIC",
    #     save_dir=out_dir,
    # )

    print("\nAll figures have been written to:", out_dir.resolve())