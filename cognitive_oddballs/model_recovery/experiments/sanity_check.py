#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from model_recovery.models import CPM, HGF
from model_recovery.data.utils import get_observations
from model_recovery.evaluation.recovery import recover_one_environment
from model_recovery.inference.mle import bic
from model_recovery.inference.laplace import map_via_grid, laplace_evidence
from cognitive_oddballs.environments.change_point_oddball import generate_change_point_environment
from cognitive_oddballs.environments.random_walk_oddball import generate_random_walk_environment

def tiny_grid():
    """Two‑point grids that keep the sanity check fast."""
    cpm = np.array(
        [(w1, w2, h) for w1 in (0.01, 0.1) for w2 in (100, 1000) for h in (0.05, 0.15)],
        dtype=float,
    )
    hgf = np.array(
        [(e, s) for e in (1e-3, 1e-2) for s in (5.0 ** 2, 15.0 ** 2)], dtype=float
    )
    return {"CPM": cpm, "HGF": hgf}

def main():
    np.random.seed(0)
    models = {"CPM": CPM, "HGF": HGF}
    grids   = tiny_grid()
    sigma_r = 2.0
    n_trials = 50

    # ------------------------------------------------------------------
    # 1️.  Change‑point environment
    # ------------------------------------------------------------------
    print("\n=== SANITY CHECK – CHANGE‑POINT ENV ===")
    results = recover_one_environment(
        models,
        true_params={"CPM": np.array([0.01, 1000.0, 0.1]), "HGF": np.array([0.05, 15.0 ** 2])},
        grids=grids,
        env_fn=generate_change_point_environment,
        n_trials=n_trials,
        sigma_r=sigma_r,
    )
    _show_summary(results, decision_rule="BIC")

    # ------------------------------------------------------------------
    # 2️.  Random‑walk environment (only CPM is a sensible generator)
    # ------------------------------------------------------------------
    print("\n=== SANITY CHECK – RANDOM‑WALK ENV ===")
    results = recover_one_environment(
        models,
        true_params={"CPM": np.array([0.01, 1000.0, 0.1]), "HGF": np.array([0.05, 15.0 ** 2])},
        grids=grids,
        env_fn=generate_random_walk_environment,
        n_trials=n_trials,
        sigma_r=sigma_r,
    )
    _show_summary(results, decision_rule="BIC")

def _show_summary(res, decision_rule="BIC"):
    """Pretty‑print a tiny table for the sanity check."""
    for true_name, fit_dict in res.items():
        print(f"\nTrue model: {true_name}")
        for fit_name, info in fit_dict.items():
            if decision_rule == "BIC":
                score = info["MLE"]["BIC"]
                param = info["MLE"]["best_params"]
                print(f"  fit={fit_name:3s} | BIC={score:8.2f} | params={param}")
            else:
                score = info["Bayesian"]["log_evidence"]
                param = info["Bayesian"]["MAP"]
                print(f"  fit={fit_name:3s} | logEv={score:8.2f} | MAP={param}")

if __name__ == "__main__":
    main()