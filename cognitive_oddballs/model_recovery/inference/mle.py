from __future__ import annotations
import numpy as np
from itertools import product
from typing import Tuple, Iterable, Dict, Any
from .utils import response_log_likelihood
from .run import run_on_sequence

def grid_search_mle(
    model_cls: Any,
    param_grid: Iterable[Tuple[float, ...]],
    observations: np.ndarray,
    responses: np.ndarray,
    sigma_r: float,
) -> Tuple[np.ndarray, float]:
    """
    Exhaustive grid search for the parameters that maximise the response log‑likelihood.
    Returns (best_params, best_loglik).
    """
    best_ll = -np.inf
    best_params = None

    for params in param_grid:
        model = model_cls(*params)               # instantiate wrapper
        beliefs, _, _ = run_on_sequence(model, observations, sigma_r, responses)
        ll = response_log_likelihood(responses, beliefs, sigma_r)

        if np.isfinite(ll) and ll > best_ll:
            best_ll = ll
            best_params = np.asarray(params, dtype=float)

    return best_params, best_ll


def bic(k: int, n: int, ll: float) -> float:
    """Bayesian Information Criterion (lower is better)."""
    return -2 * ll + k * np.log(n)