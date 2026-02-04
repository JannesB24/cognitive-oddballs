from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .utils import response_log_likelihood, safe_log_prior, numerical_hessian
from .run import run_on_sequence

def map_via_grid(
    model_cls: Any,
    param_grid: Iterable[Tuple[float, ...]],
    observations: np.ndarray,
    responses: np.ndarray,
    sigma_r: float,
) -> Tuple[np.ndarray, float]:
    """
    Returns the MAP estimate (grid‑search + prior) and its joint log‑posterior.
    """
    best_logpost = -np.inf
    best_params = None

    for params in param_grid:
        model = model_cls(*params)
        beliefs, _, _ = run_on_sequence(model, observations, sigma_r, responses)
        ll = response_log_likelihood(responses, beliefs, sigma_r)
        if not np.isfinite(ll):
            continue
        lp = safe_log_prior(model, np.asarray(params))
        logpost = ll + lp
        if logpost > best_logpost:
            best_logpost = logpost
            best_params = np.asarray(params, dtype=float)

    return best_params, best_logpost


def laplace_evidence(
    model_cls: Any,
    map_params: np.ndarray,
    logpost_map: float,
    observations: np.ndarray,
    responses: np.ndarray,
    sigma_r: float,
    eps: float = 1e-4,
) -> float:
    """
    Laplace approximation of the marginal likelihood.
    Handles the transformation to the unconstrained (log) space internally.
    """
    k = len(map_params)

    # ---- objective in *unconstrained* space ----
    def neg_log_joint(theta_u: np.ndarray) -> float:
        theta = np.exp(theta_u)                     # back‑transform
        model = model_cls(*theta)
        beliefs, _, _ = run_on_sequence(model, observations, sigma_r, responses)
        ll = response_log_likelihood(responses, beliefs, sigma_r)
        if not np.isfinite(ll):
            return np.inf
        lp = safe_log_prior(model, theta)
        return -(ll + lp)

    theta_u_map = np.log(map_params)

    # ---- Hessian of the *negative* joint posterior ----
    H = numerical_hessian(neg_log_joint, theta_u_map, eps=eps)

    # Laplace formula (log‑scale)
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf

    log_evidence = logpost_map + 0.5 * k * np.log(2 * np.pi) - 0.5 * logdet
    return log_evidence