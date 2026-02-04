#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py
=======

Entry point for the “model‑recovery on odd‑ball tasks” package.

Typical ways to call it:

    $ python -m model_recovery.main --sanity               # fast sanity‑check (≈ a few seconds)
    $ python -m model_recovery.main --n‑sims 200          # full recovery, 200 simulations
    $ python -m model_recovery.main --rule laplace        # use Laplace evidence instead of BIC
"""

from __future__ import annotations

import argparse
import sys
import warnings

# ----------------------------------------------------------------------
# Import the public API that the CLI will use
# ----------------------------------------------------------------------
from model_recovery.experiments.sanity_check import main as sanity_check
from model_recovery.experiments.run_many   import many_sims
from model_recovery.evaluation.summary     import confusion_matrix, param_recovery_stats

# ----------------------------------------------------------------------
# Helper that builds the model / grid dictionaries used everywhere
# ----------------------------------------------------------------------
def build_models_and_grids():
    """
    Returns
    -------
    models : dict[str, Callable[..., BaseModel]]
        {'CPM': CPM, 'HGF': HGF}
    grids  : dict[str, np.ndarray]
        {'CPM': <(n_grid,3) array>, 'HGF': <(m_grid,2) array>}
    """
    # --------------------------------------------------------------
    # 1️.  Model wrappers (they all inherit from BaseModel)
    # --------------------------------------------------------------
    from model_recovery.models import CPM, HGF
    models = {"CPM": CPM, "HGF": HGF}

    # --------------------------------------------------------------
    # 2️.  Parameter grids – the same ones you used in the original script
    # --------------------------------------------------------------
    import numpy as np

    # ----- CPM grid ------------------------------------------------
    w1_grid = np.linspace(0.05, 0.5, 6)
    w2_grid = np.array([50.0, 200.0, 1000.0])
    h_grid  = np.linspace(0.01, 0.3, 6)
    cpm_grid = np.array(
        [(w1, w2, h) for w1 in w1_grid for w2 in w2_grid for h in h_grid],
        dtype=float,
    )

    # ----- HGF grid ------------------------------------------------
    eta_grid = np.logspace(-4, -1, 8)
    s_grid   = np.logspace(np.log10(1.0), np.log10(50.0 ** 2), 8)
    hgf_grid = np.array(
        [(e, s) for e in eta_grid for s in s_grid],
        dtype=float,
    )

    grids = {"CPM": cpm_grid, "HGF": hgf_grid}
    return models, grids


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model‑recovery experiments (MLE‑BIC vs Laplace)."
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run the fast sanity‑check (no Monte‑Carlo loops).",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=0,
        help="Number of simulated datasets for the full recovery. 0 = skip.",
    )
    parser.add_argument(
        "--rule",
        choices=["BIC", "laplace"],
        default="BIC",
        help="Decision rule used to pick the winning model.",
    )
    parser.add_argument(
        "--env",
        choices=["changepoint", "randomwalk"],
        default="changepoint",
        help="Which environment to use for the full recovery run.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=200,
        help="Number of trials per simulated experiment.",
    )
    parser.add_argument(
        "--sigma-r",
        type=float,
        default=5.0,
        help="Std‑dev of the response noise (σᵣ).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # Global seed – everything downstream is deterministic
    # ------------------------------------------------------------------
    import numpy as np
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1️.  Fast sanity‑check (optional, runs in < 5 s)
    # ------------------------------------------------------------------
    if args.sanity:
        print("\n=== RUNNING QUICK SANITY CHECK ===")
        # The sanity‑check script already prints its own tables.
        sanity_check()
        # After a sanity‑check we normally stop – you can continue if you like.
        if args.n_sims == 0:
            return 0

    # ------------------------------------------------------------------
    # 2️.  Full Monte‑Carlo model‑recovery (optional)
    # ------------------------------------------------------------------
    if args.n_sims > 0:
        models, grids = build_models_and_grids()

        # Choose the appropriate environment generator
        if args.env == "changepoint":
            from cognitive_oddballs.environments.change_point_oddball import (
                generate_change_point_environment,
            )
            env_fn = generate_change_point_environment
        else:  # randomwalk
            from cognitive_oddballs.environments.random_walk_oddball import (
                generate_random_walk_environment,
            )
            env_fn = generate_random_walk_environment

        print("\n=== RUNNING FULL MODEL RECOVERY ===")
        print(
            f" n_sims={args.n_sims}, env={args.env}, rule={args.rule}, "
            f"trials={args.trials}, sigma_r={args.sigma_r}"
        )

        winners, param_pairs = many_sims(
            n_sims=args.n_sims,
            models=models,
            grids=grids,
            env_fn=env_fn,
            n_trials=args.trials,
            sigma_r=args.sigma_r,
            decision_rule=args.rule,
            seed=args.seed,
        )

        # ------------------------------------------------------------------
        # 3️.  Summarise results
        # ------------------------------------------------------------------
        confusion_matrix(winners)
        param_recovery_stats(param_pairs)

    # ----------------------------------------------------------------------
    # Normal termination
    # ----------------------------------------------------------------------
    return 0


if __name__ == "__main__":
    # ``sys.exit`` makes the exit‑code visible to the shell (0 = success)
    sys.exit(main())