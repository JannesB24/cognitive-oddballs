#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
visualisation.py
================

Helper functions that turn the raw dictionaries produced by the recovery
routines into the four figures that appear in Marković & Kiebel (2016):

1️.  Confusion matrix (heat‑map, with percentages and colour bar)
2️.  Parameter‑recovery scatter plots (one panel per model)
3️.  BIC / log‑evidence difference histograms (separate panel per true model)
4️.  Empirical CDFs of the model‑selection scores (optional)

The functions are deliberately **stateless**: they just take the data structures
that `run_many.py` returns and either show the figure (`.show()`) or save it
to a directory (`save_dir`).  They use only `matplotlib` / `seaborn`, add
to `requirements.txt` if they are not already there.

    matplotlib>=3.8
    seaborn>=0.14
"""

from __future__ import annotations

import os
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# 1️⃣ Confusion matrix (heat‑map)
# ----------------------------------------------------------------------
def plot_confusion_matrix(
    winners: Dict[str, Dict[str, int]],
    *,
    title: str = "Model‑recovery confusion matrix",
    cmap: str = "Blues",
    save_dir: str | os.PathLike | = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Parameters
    ----------
    winners : dict
        `winners[true_model][recovered_model]` → count of simulations.
    title   : str, optional
        Figure title.
    cmap   : str, optional
        Matplotlib colour map.
    save_dir : path‑like or None
        If given, the figure is saved as ``confusion_matrix.png`` inside the folder.
    dpi    : int
        Image resolution for the saved file.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure (so you can further customise it if you want).
    """
    # Convert counts to percentages (row‑wise normalisation)
    models = list(winners.keys())
    n_models = len(models)

    matrix = np.zeros((n_models, n_models), dtype=float)
    for i, true_m in enumerate(models):
        row = winners[true_m]
        total = sum(row.values()) or 1  # avoid div‑by‑zero
        for j, rec_m in enumerate(models):
            matrix[i, j] = 100.0 * row.get(rec_m, 0) / total

    # ---- Plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(1.2 * n_models, 1.0 * n_models))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        cbar_kws={"label": "Proportion [%]"},
        xticklabels=models,
        yticklabels=models,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )
    ax.set_xlabel("Recovered model")
    ax.set_ylabel("True model")
    ax.set_title(title, pad=15)

    # ---- Save ---------------------------------------------------------
    if save_dir is not None:
        out_path = Path(save_dir) / "confusion_matrix.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"✔︎ Confusion matrix saved to {out_path}")

    return fig


# ----------------------------------------------------------------------
# 2️⃣ Parameter‑recovery scatter plots (one panel per model)
# ----------------------------------------------------------------------
def plot_parameter_recovery(
    param_pairs: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    *,
    title: str = "Parameter recovery (true vs. recovered)",
    save_dir: str | os.PathLike | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Scatter plot of true vs. recovered parameters for each model.
    The panels are arranged in a grid (one column per model, rows = #params).

    Parameters
    ----------
    param_pairs : dict
        `param_pairs[model]` → list of (true_params, recovered_params) tuples,
        **only for simulations where the model was correctly identified**.
    title      : str, optional
        Figure title.
    save_dir   : path‑like or None
        If given, the figure is saved as ``parameter_recovery.png``.
    dpi        : int
        Resolution of the saved file.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Count max number of parameters across models – determines the grid height
    max_n_params = max(p[0].shape[0] for lst in param_pairs.values() for p in lst) if param_pairs else 0
    n_models = len(param_pairs)

    if max_n_params == 0:
        raise ValueError("param_pairs is empty – no correctly recovered simulations.")

    fig, axes = plt.subplots(
        max_n_params,
        n_models,
        figsize=(3 * n_models, 3 * max_n_params),
        squeeze=False,
        sharex="col",
        sharey="row",
    )
    fig.suptitle(title, fontsize=14, y=1.02)

    # ------------------------------------------------------------------
    #  Loop over models & their parameters
    # ------------------------------------------------------------------
    for col, (model_name, pairs) in enumerate(param_pairs.items()):
        # Stack true & recovered for easier slicing
        true_arr = np.vstack([p[0] for p in pairs])
        rec_arr  = np.vstack([p[1] for p in pairs])

        n_params = true_arr.shape[1]
        for row in range(n_params):
            ax = axes[row, col]
            ax.scatter(true_arr[:, row], rec_arr[:, row], alpha=0.6, edgecolor="k", linewidth=0.4)

            # 45° identity line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, "k--", lw=0.8)

            # Correlation coefficient (Pearson)
            r = np.corrcoef(true_arr[:, row], rec_arr[:, row])[0, 1]
            ax.text(
                0.05,
                0.85,
                f"$r = {r:.2f}$",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

            if row == max_n_params - 1:
                ax.set_xlabel("True")
            if col == 0:
                ax.set_ylabel("Recovered")

            ax.set_title(f"{model_name} – p{row+1}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ---- Save ---------------------------------------------------------
    if save_dir is not None:
        out_path = Path(save_dir) / "parameter_recovery.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"✔︎ Parameter‑recovery plot saved to {out_path}")

    return fig


# ----------------------------------------------------------------------
# 3️⃣ Distribution of model‑selection scores (BIC or log‑evidence)
# ----------------------------------------------------------------------
def plot_score_distributions(
    results: Dict[str, Dict[str, Dict[str, float]]],
    *,
    decision_rule: str = "BIC",
    title: str | None = None,
    save_dir: str | os.PathLike | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    For every *true* model, plot a histogram of the scores obtained for the
    competing models.  When the decision rule is BIC we plot the *difference*
    (BIC_other – BIC_true); for Laplace we plot the *difference* of the
    log‑evidence (logev_true – logev_other).  Positive values therefore
    indicate that the true model is favoured.

    Parameters
    ----------
    results : dict
        Output of ``recover_one_environment`` – i.e. the nested dict that
        contains `"MLE"`/`"Bayesian"` information for each (true, fit) pair.
    decision_rule : {"BIC", "laplace"}
        Which column of the nested dict to use.
    title : str | None
        Figure title (default depends on the rule).
    save_dir : path‑like or None
        If given, the figure is saved as ``score_distributions.png``.
    dpi : int
        Saved‑image resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if decision_rule.lower() == "bic":
        key = "BIC"
        ylabel = "Δ BIC (= BIC_other – BIC_true)"
    else:
        key = "log_evidence"
        ylabel = r"$\Delta \log \mathcal{E}$ (= logev_true – logev_other)"

    models = list(results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(4 * n_models, 4),
        sharey=True,
        squeeze=False,
    )
    if title is None:
        title = f"Score distributions ({decision_rule.upper()})"
    fig.suptitle(title, fontsize=14, y=1.02)

    for col, true_m in enumerate(models):
        ax = axes[0, col]
        diffs = []          # collect all differences for this true model
        labels = []         # which competing model the difference belongs to

        for fit_m in models:
            if fit_m == true_m:
                continue
            # Extract the relevant numbers
            if decision_rule.lower() == "bic":
                score_true = results[true_m][true_m]["MLE"][key]
                score_fit  = results[true_m][fit_m]["MLE"][key]
                diff = score_fit - score_true
            else:
                ev_true = results[true_m][true_m]["Bayesian"][key]
                ev_fit  = results[true_m][fit_m]["Bayesian"][key]
                diff = ev_true - ev_fit          # note the reversed sign for evidence
            diffs.append(diff)
            labels.append(fit_m)

        # Plot as overlapping half‑density rugs (or simple histograms)
        for d, lab in zip(diffs, labels):
            sns.histplot(
                d,
                bins=15,
                kde=True,
                stat="density",
                element="step",
                fill=False,
                label=lab,
                ax=ax,
                linewidth=1.5,
            )

        ax.axvline(0, color="k", linestyle="--", lw=1)
        ax.set_xlabel(ylabel)
        ax.set_ylabel("Density")
        ax.set_title(f"True = {true_m}")
        ax.legend(title="Competing model")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir is not None:
        out_path = Path(save_dir) / "score_distributions.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"✔︎ Score‑distribution plot saved to {out_path}")

    return fig


# ----------------------------------------------------------------------
# 4️⃣ Helper to generate *all* figures at once (convenient for notebooks)
# ----------------------------------------------------------------------
def make_all_plots(
    winners: Dict[str, Dict[str, int]],
    param_pairs: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    results: Dict[str, Dict[str, Dict[str, float]]],
    *,
    decision_rule: str = "BIC",
    save_dir: str | os.PathLike | None = None,
    dpi: int = 150,
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Calls the three plot functions above and returns the three Figure objects.
    This is handy if you want to embed the figures in a Jupyter notebook:

        >>> fig1, fig2, fig3 = make_all_plots(...)
        >>> display(fig1)
        >>> display(fig2)
        >>> display(fig3)

    Parameters
    ----------
    winners, param_pairs, results : see the three individual functions.
    decision_rule : {"BIC","laplace"}
        Determines which score‑distribution plot is produced.
    save_dir : path‑like or None
        Folder for saving PNG files; if ``None`` no files are written.
    dpi : int
        Resolution for PNG output.

    Returns
    -------
    (conf_mat_fig, param_rec_fig, score_dist_fig)
    """
    cm = plot_confusion_matrix(winners, save_dir=save_dir, dpi=dpi)
    pr = plot_parameter_recovery(param_pairs, save_dir=save_dir, dpi=dpi)
    sd = plot_score_distributions(results, decision_rule=decision_rule,
                                  save_dir=save_dir, dpi=dpi)
    return cm, pr, sd