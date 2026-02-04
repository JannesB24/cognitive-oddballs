from __future__ import annotations
import numpy as np
from typing import Callable

def response_log_likelihood(
    responses: np.ndarray,
    beliefs: np.ndarray,
    sigma_r: np.ndarray | float,
) -> float:
    """Gaussian response model r_t ~ 𝒩(μ_t, σ_r²)."""
    r = np.asarray(responses, dtype=float)
    m = np.asarray(beliefs, dtype=float)
    s = np.asarray(sigma_r, dtype=float)

    if np.any(s <= 0):
        return -np.inf

    var = s ** 2
    return -0.5 * np.sum(((r - m) ** 2) / var + np.log(2 * np.pi * var))

def safe_log_prior(model: Any, params: np.ndarray) -> float:
    """Return log‑prior if the model implements `log_prior`, else 0."""
    fn = getattr(model, "log_prior", None)
    return float(fn(params)) if callable(fn) else 0.0

def numerical_hessian(
    fun: Callable[[np.ndarray], float],
    theta: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Central‑difference Hessian (symmetrised)."""
    k = len(theta)
    H = np.zeros((k, k))
    for i in range(k):
        ei = np.zeros(k); ei[i] = eps
        for j in range(i, k):
            ej = np.zeros(k); ej[j] = eps
            f_pp = fun(theta + ei + ej)
            f_pm = fun(theta + ei - ej)
            f_mp = fun(theta - ei + ej)
            f_mm = fun(theta - ei - ej)
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps ** 2)
            H[j, i] = H[i, j]
    return H