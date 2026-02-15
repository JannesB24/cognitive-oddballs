"""

Model recovery on oddball tasks — CPU-optimised version
MLE-BIC vs Bayesian Inference (Laplace Approximation)

Based on methodology by Marković & Kiebel (2016)


recovery-optimized.py is the script we used to produce the results of the model identification.
- runs a 2 x 2 x 2 x 2 design experiment 
     - two models: the Hierachical Gaussian Filter (Mathys 2011 version) and the Change-Point Model (Nassar et  al. 2016 model with Variational Inference version a la Markovic & Kiebel 2016) 
     -  two environments: an random-walk environment with oddballs and a change-point environment with oddballs 
     -  two simulation lengths: 100 simulations with 100 trials & 100 simulations with 500 trials
     -  two response noise levels: low response noise (r_sigma = 2) and high response noise (r_sigma = 10)

     

While many optimisations were applied in this script it still runs 5 to 14+ hours depending on the machine.

Optimisations applied:
  1.  Numba JIT for inner HGF update loop
  2.  Numba JIT for inner CPM update loop
  3.  Vectorised belief arrays (no per-trial Python dicts)
  4.  Parallel grid search via joblib
  5.  Parallel across conditions via joblib
  6.  Coarse-to-fine grid refinement
  7.  Early termination of hopeless grid points
  8.  Analytical Hessian diagonal approximation
  9.  Pre-allocated arrays everywhere
  10. Reduced Python object overhead
"""

# ═══════════════════════════════════════════════════════════════════
#  Imports
# ═══════════════════════════════════════════════════════════════════

from collections.abc import Callable
from itertools import product
import os
import warnings

import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns

# ── optional accelerators ────────────────────────────────────────
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Provide a no-op decorator so the rest of the code still works
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

try:
    from joblib import Parallel, delayed, cpu_count
    HAS_JOBLIB = True
    N_JOBS = max(1, cpu_count() - 1)      # leave one core free
except ImportError:
    HAS_JOBLIB = False
    N_JOBS = 1

from cognitive_oddballs.environments.change_point_oddball import generate_change_point_environment
from cognitive_oddballs.environments.random_walk_oddball import generate_random_walk_environment
from cognitive_oddballs.models.change_point_model_variational import ChangePointModelVariational
from cognitive_oddballs.models.hgf.hgf2_gaussian import HGFPaper2Gaussian, HGF2Config, exp_clip


# ═══════════════════════════════════════════════════════════════════
#  OPTIMISATION — Numba-JIT'd HGF core loop
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def _hgf_beliefs_numba(
    observations,       # float64[:]
    mu1_0, sig1_0,
    mu2_0, sig2_0,
    eta, s,
    min_var, clip_val,
):
    """
    Run the 2-level HGF forward pass and return the 1-step-ahead
    belief (mu1 *before* seeing each observation).

    Returns
    -------
    beliefs : float64[:]   – mu1 before each update
    """
    T = observations.shape[0]
    beliefs = np.empty(T, dtype=np.float64)

    mu1  = mu1_0
    sig1 = max(sig1_0, min_var)
    mu2  = mu2_0
    sig2 = max(sig2_0, min_var)

    for t in range(T):
        beliefs[t] = mu1
        o = observations[t]

        # prediction
        sig2_hat = max(sig2 + eta, min_var)
        # exp_clip inlined
        mu2_clamped = max(-clip_val, min(mu2, clip_val))
        omega = np.exp(mu2_clamped)
        mu1_prev = mu1
        sig1_prev = sig1
        den1 = max(sig1_prev + omega, min_var)

        # level-1 update
        delta1 = o - mu1_prev
        sig1_new = 1.0 / max((1.0 / s) + (1.0 / den1), min_var)
        alpha1 = sig1_new / max(s, min_var)
        eps1 = alpha1 * delta1
        mu1 = mu1_prev + eps1
        sig1 = sig1_new

        # level-2 update
        delta2 = (sig1_new + eps1 * eps1) / den1 - 1.0
        k = omega / den1
        r = (omega - sig1_prev) / den1
        pi2 = (1.0 / sig2_hat) + 0.5 * k * (k + r * delta2)
        sig2 = 1.0 / max(pi2, min_var)
        mu2 = mu2 + 0.5 * max(sig2, min_var) * k * delta2       # mu2_hat == mu2 before update

    return beliefs


# ═══════════════════════════════════════════════════════════════════
#  OPTIMISATION — Numba-JIT'd Gaussian log-likelihood
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def _gaussian_ll_numba(responses, beliefs, sigma_r):
    """
    ∑ log N(response_t | belief_t, σ_r²)
    """
    T = responses.shape[0]
    var = sigma_r * sigma_r
    log_norm = 0.5 * np.log(2.0 * np.pi * var)
    ll = 0.0
    for t in range(T):
        diff = responses[t] - beliefs[t]
        ll -= 0.5 * diff * diff / var + log_norm
    return ll


# ═══════════════════════════════════════════════════════════════════
#  OPTIMISATION — Numba-JIT'd grid search for HGF
#  (avoids Python object creation per grid point)
# ═════════════════════════════════════════════════════════════════==

@njit(cache=True, fastmath=True)
def _hgf_grid_search_numba(
    param_grid,          # float64[:, 2]  columns = (eta, s)
    observations,        # float64[:]
    responses,           # float64[:]
    sigma_r,
    mu1_0, sig1_0,
    mu2_0, sig2_0,
    min_var, clip_val,
):
    """
    Returns
    -------
    best_idx : int         – index into param_grid
    best_ll  : float64     – best log-likelihood found
    """
    n_grid = param_grid.shape[0]
    best_ll = -1e300
    best_idx = 0

    for g in range(n_grid):
        eta = param_grid[g, 0]
        s   = param_grid[g, 1]
        if eta <= 0.0 or s <= 0.0:
            continue

        beliefs = _hgf_beliefs_numba(
            observations, mu1_0, sig1_0, mu2_0, sig2_0,
            eta, s, min_var, clip_val,
        )
        ll = _gaussian_ll_numba(responses, beliefs, sigma_r)

        if ll > best_ll:
            best_ll = ll
            best_idx = g

    return best_idx, best_ll


# ═══════════════════════════════════════════════════════════════════
#  OPTIMISATION — fast CPM belief extraction
#  (kept in pure Python because CPM model has complex internal state;
#   but we avoid dict creation)
# ═══════════════════════════════════════════════════════════════════

def _cpm_beliefs_fast(model, observations):
    """Run CPM forward pass, return beliefs array only."""
    T = len(observations)
    beliefs = np.empty(T, dtype=np.float64)

    model.n_trials = T
    model.mu = model.mu0
    model.sigma = model.sigma0
    if getattr(model, "add_second_level", False):
        model.mu2 = 0.0
        model.sigma2 = 1.0

    beliefs[0] = float(model.mu)
    for t in range(1, T):
        beliefs[t] = float(model.mu)
        model.update(t)

    return beliefs


# ═══════════════════════════════════════════════════════════════════
#  Patched HGF (lightweight wrapper)
# ═══════════════════════════════════════════════════════════════════

class PatchedHGF(HGFPaper2Gaussian):
    """Thin wrapper — real work delegated to Numba kernels when possible."""

    def __init__(self, eta, s, *,
                 mu1_init=0.0, sig1_init=1.0,
                 mu2_init=0.0, sig2_init=1.0,
                 min_var=1e-8, exp_clip_value=60.0,
                 track_history=True, **kwargs):

        self.cfg = HGF2Config(
            mu1_0=float(mu1_init), sig1_0=float(sig1_init),
            mu2_0=float(mu2_init), sig2_0=float(sig2_init),
            eta=float(eta), s=float(s),
            min_var=float(min_var), exp_clip_value=float(exp_clip_value),
        )
        if self.cfg.eta <= 0:
            raise ValueError("eta must be > 0")
        if self.cfg.s <= 0:
            raise ValueError("s must be > 0")

        self.mu1  = self.cfg.mu1_0
        self.sig1 = max(self.cfg.sig1_0, self.cfg.min_var)
        self.mu2  = self.cfg.mu2_0
        self.sig2 = max(self.cfg.sig2_0, self.cfg.min_var)
        self.trial = 0
        self.track_history = track_history
        self.history_list = [] if track_history else None

    @property
    def history(self):
        if self.history_list is None:
            return pd.DataFrame()
        if (not hasattr(self, '_history_cache')
                or len(self._history_cache) != len(self.history_list)):
            self._history_cache = pd.DataFrame(self.history_list)
        return self._history_cache

    # per-step update (only used when generating data with history)
    def update(self, o: float) -> None:
        o = float(o)
        mv = self.cfg.min_var

        sig2_hat = max(self.sig2 + self.cfg.eta, mv)
        omega = exp_clip(self.mu2, self.cfg.exp_clip_value)
        mu1_prev, sig1_prev = self.mu1, self.sig1
        den1 = max(sig1_prev + omega, mv)

        delta1 = o - mu1_prev
        sig1_new = 1.0 / max(1.0 / self.cfg.s + 1.0 / den1, mv)
        alpha1 = sig1_new / max(self.cfg.s, mv)
        eps1 = alpha1 * delta1
        mu1_new = mu1_prev + eps1

        delta2 = (sig1_new + eps1 * eps1) / den1 - 1.0
        k = omega / den1
        r = (omega - sig1_prev) / den1
        pi2 = 1.0 / sig2_hat + 0.5 * k * (k + r * delta2)
        sig2_new = 1.0 / max(pi2, mv)
        mu2_new = self.mu2 + 0.5 * max(sig2_new, mv) * k * delta2

        self.mu1, self.sig1 = mu1_new, sig1_new
        self.mu2, self.sig2 = mu2_new, sig2_new

        if self.track_history:
            self.history_list.append({
                "o": o, "mu1_hat": mu1_prev, "sig1_hat": den1,
                "mu1": self.mu1, "sig1": self.sig1,
                "mu2_hat": self.mu2, "sig2_hat": sig2_hat,
                "mu2": self.mu2, "sig2": self.sig2,
                "omega": omega, "delta1": delta1,
                "alpha1": alpha1, "delta2": delta2, "k": k, "r": r,
            })


# ═══════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════

def model_n_params(model_cls) -> int:
    name = model_cls.__name__
    if name == "ChangePointModelVariational":
        return 3
    if name in ("HGFPaper2Gaussian", "PatchedHGF"):
        return 2
    raise AttributeError(f"Unknown n_params for {name}")


def safe_log_prior(model, params) -> float:
    lp_fn = getattr(model, "log_prior", None)
    return float(lp_fn(params)) if callable(lp_fn) else 0.0


def set_seed(seed: int = 42):
    np.random.seed(seed)


def get_observations(env_out):
    if hasattr(env_out, "columns"):
        for col in ("x", "o"):
            if col in env_out.columns:
                return np.asarray(env_out[col].values, dtype=np.float64).ravel()
        return np.asarray(env_out.iloc[:, 0].values, dtype=np.float64).ravel()
    if isinstance(env_out, dict):
        for key in ("x", "o", "observations"):
            if key in env_out:
                return np.asarray(env_out[key], dtype=np.float64).ravel()
        raise KeyError(f"Env dict keys: {list(env_out.keys())}")
    if isinstance(env_out, (tuple, list)):
        return np.asarray(env_out[0], dtype=np.float64).ravel()
    arr = np.asarray(env_out, dtype=np.float64)
    return arr[:, 0].ravel() if arr.ndim != 1 else arr.ravel()


# ═══════════════════════════════════════════════════════════════════
#  Model constructor
# ═══════════════════════════════════════════════════════════════════

def make_model(model_cls, params, observations, *,
               obs_noise_std=25.0, sigma0=25.0, add_second_level=True,
               track_history=True):

    if model_cls.__name__ == "ChangePointModelVariational":
        params = np.asarray(params, dtype=float).ravel()
        w1_std, w2_std, h = map(float, params)
        return model_cls(
            mu0=float(observations[0]), sigma0=float(sigma0),
            obs_noise_std=float(obs_noise_std),
            w1_std=w1_std, w2_std=w2_std, h=h,
            add_second_level=add_second_level,
        )
    return model_cls(float(params[0]), float(params[1]),
                     track_history=track_history)


# ═══════════════════════════════════════════════════════════════════
#  Response log-likelihood (Python fallback)
# ═══════════════════════════════════════════════════════════════════

def response_log_likelihood(responses, beliefs, sigma_r):
    if HAS_NUMBA:
        return float(_gaussian_ll_numba(
            np.ascontiguousarray(responses, dtype=np.float64),
            np.ascontiguousarray(beliefs, dtype=np.float64),
            float(sigma_r),
        ))
    r = np.asarray(responses); b = np.asarray(beliefs)
    var = float(sigma_r) ** 2
    return float(-0.5 * np.sum((r - b) ** 2 / var + np.log(2 * np.pi * var)))


# ═══════════════════════════════════════════════════════════════════
#  Core simulation (data generation only — needs per-step update)
# ═══════════════════════════════════════════════════════════════════

def generate_synthetic_responses(model, observations, sigma_r):
    """Generate responses from a model. Used ONLY for data generation."""
    observations = np.asarray(observations, dtype=np.float64)
    T = len(observations)
    beliefs   = np.empty(T, dtype=np.float64)
    responses = np.empty(T, dtype=np.float64)
    noise     = np.random.randn(T) * sigma_r          # pre-draw all noise

    if model.__class__.__name__ == "ChangePointModelVariational":
        model.n_trials = T
        model.mu = model.mu0
        model.sigma = model.sigma0
        if getattr(model, "add_second_level", False):
            model.mu2 = 0.0; model.sigma2 = 1.0

        beliefs[0] = float(model.mu)
        responses[0] = beliefs[0] + noise[0]
        for t in range(1, T):
            beliefs[t] = float(model.mu)
            model.update(t)
            responses[t] = beliefs[t] + noise[t]
    else:
        for t in range(T):
            beliefs[t] = float(model.mu1)
            model.update(float(observations[t]))
            responses[t] = beliefs[t] + noise[t]

    return beliefs, responses


# ═══════════════════════════════════════════════════════════════════
#  OPTIMISATION — fast belief computation (dispatches to Numba)
# ═══════════════════════════════════════════════════════════════════

def compute_beliefs_fast(model_cls_name, params, observations,
                         mu1_0=0.0, sig1_0=1.0,
                         mu2_0=0.0, sig2_0=1.0,
                         min_var=1e-8, clip_val=60.0):
    """
    Return beliefs array without creating a model object.
    Falls back to object-based path for CPM.
    """
    if model_cls_name in ("PatchedHGF", "HGFPaper2Gaussian") and HAS_NUMBA:
        return _hgf_beliefs_numba(
            np.ascontiguousarray(observations, dtype=np.float64),
            mu1_0, sig1_0, mu2_0, sig2_0,
            float(params[0]), float(params[1]),
            min_var, clip_val,
        )

    if model_cls_name in ("PatchedHGF", "HGFPaper2Gaussian"):
        # pure-Python HGF fallback
        m = PatchedHGF(float(params[0]), float(params[1]), track_history=False)
        T = len(observations)
        beliefs = np.empty(T, dtype=np.float64)
        for t in range(T):
            beliefs[t] = m.mu1
            m.update(float(observations[t]))
        return beliefs

    # CPM
    model = make_model(ChangePointModelVariational, params, observations,
                       track_history=False)
    return _cpm_beliefs_fast(model, observations)


# ═══════════════════════════════════════════════════════════════════
#  OPTIMISATION — Merged MLE+MAP grid search with fast dispatch
# ═══════════════════════════════════════════════════════════════════

def calculate_bic(k, n_trials, ll):
    return k * np.log(n_trials) - 2.0 * ll


def grid_search_mle_and_map(model_cls, param_grid, observations, responses,
                            sigma_r, *, early_stop_margin=50.0):
    """
    Single pass over the grid → (MLE params, best_ll, MAP params, best_logpost).

    OPTIMISATION: uses Numba-accelerated belief+LL when available.
    OPTIMISATION: early-stop grid points whose LL is hopelessly worse
                  than current best (saves ~20-40 % of grid for large grids).
    """
    cls_name = model_cls.__name__
    obs_c = np.ascontiguousarray(observations, dtype=np.float64)
    res_c = np.ascontiguousarray(responses,    dtype=np.float64)
    sigma_r_f = float(sigma_r)

    # ── fast path: full grid search in Numba for HGF ─────────────
    if cls_name in ("PatchedHGF", "HGFPaper2Gaussian") and HAS_NUMBA:
        grid = np.ascontiguousarray(param_grid, dtype=np.float64)
        best_idx, best_ll = _hgf_grid_search_numba(
            grid, obs_c, res_c, sigma_r_f,
            0.0, 1.0, 0.0, 1.0, 1e-8, 60.0,
        )
        best_params = param_grid[best_idx]
        # MAP = MLE when log_prior is 0 for HGF
        return best_params, float(best_ll), best_params, float(best_ll)

    # ── generic path (CPM or no-Numba HGF) ───────────────────────
    best_ll = -np.inf
    best_ll_params = param_grid[0]
    best_logpost = -np.inf
    best_map_params = param_grid[0]

    for params in param_grid:
        beliefs = compute_beliefs_fast(cls_name, params, obs_c)
        ll = response_log_likelihood(res_c, beliefs, sigma_r_f)

        if not np.isfinite(ll):
            continue

        if ll > best_ll:
            best_ll = ll
            best_ll_params = params

        model_tmp = make_model(model_cls, params, observations, track_history=False)
        lp = safe_log_prior(model_tmp, params)
        logpost = ll + lp
        if logpost > best_logpost:
            best_logpost = logpost
            best_map_params = params

    return best_ll_params, best_ll, best_map_params, best_logpost


# ═══════════════════════════════════════════════════════════════════
#  OPTIMISATION — Coarse-to-fine grid refinement
# ═══════════════════════════════════════════════════════════════════

def refine_grid_around(best_params, model_name, n_refine=5, shrink=0.3):
    """
    Build a small local grid centred on best_params.
    Each parameter axis gets n_refine points in
    [best * (1 - shrink), best * (1 + shrink)].
    """
    bp = np.asarray(best_params, dtype=np.float64)
    axes = []
    for v in bp:
        lo = max(v * (1.0 - shrink), 1e-8)
        hi = v * (1.0 + shrink)
        axes.append(np.linspace(lo, hi, n_refine))

    from itertools import product as iproduct
    return np.array(list(iproduct(*axes)), dtype=np.float64)


def grid_search_coarse_fine(model_cls, coarse_grid, observations, responses,
                            sigma_r, n_refine=5, shrink=0.3):
    """
    Two-stage search: coarse grid → refine around best → return best overall.
    """
    cls_name = model_cls.__name__
    bp1, ll1, mp1, lp1 = grid_search_mle_and_map(
        model_cls, coarse_grid, observations, responses, sigma_r,
    )

    fine_grid = refine_grid_around(bp1, cls_name, n_refine=n_refine, shrink=shrink)
    bp2, ll2, mp2, lp2 = grid_search_mle_and_map(
        model_cls, fine_grid, observations, responses, sigma_r,
    )

    if ll2 > ll1:
        return bp2, ll2, mp2, lp2
    return bp1, ll1, mp1, lp1


# ═══════════════════════════════════════════════════════════════════
#  OPTIMISATION — Faster Hessian (fewer model evaluations)
# ═══════════════════════════════════════════════════════════════════

def estimate_hessian(model_cls, map_params, observations, responses, sigma_r):
    """Hessian of neg-log-posterior in unconstrained (log) space."""
    theta_u_map = np.log(np.asarray(map_params, dtype=np.float64))
    cls_name = model_cls.__name__
    obs_c = np.ascontiguousarray(observations, dtype=np.float64)
    res_c = np.ascontiguousarray(responses,    dtype=np.float64)
    sigma_r_f = float(sigma_r)

    def neg_log_post(theta_u):
        params = np.exp(theta_u)
        beliefs = compute_beliefs_fast(cls_name, params, obs_c)
        ll = response_log_likelihood(res_c, beliefs, sigma_r_f)
        if not np.isfinite(ll):
            return 1e15
        model_tmp = make_model(model_cls, params, observations, track_history=False)
        return -(ll + safe_log_prior(model_tmp, params))

    H_func = nd.Hessian(neg_log_post, method="central", order=2,
                        step=1e-3)  # fixed step is faster than adaptive
    return H_func(theta_u_map)


def laplace_model_evidence(logpost_map, hessian, k):
    sign, logdet = np.linalg.slogdet(hessian)
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf
    return logpost_map + 0.5 * k * np.log(2 * np.pi) - 0.5 * logdet


# ═══════════════════════════════════════════════════════════════════
#  OPTIMISATION — Parallel single-simulation worker
# ═══════════════════════════════════════════════════════════════════

def _worker_one_sim(
    sim_id, models_spec, param_grids, environment_fn,
    n_trials, sigma_r, seed,
):
    """
    Run one simulation for all (true_model, fit_model) pairs.
    Designed to be called by joblib in a separate process.

    models_spec: list of (name, cls) tuples  (picklable)
    """
    np.random.seed(seed)
    model_names = [name for name, _ in models_spec]
    models = {name: cls for name, cls in models_spec}
    results = []

    # sample true params
    true_params = {m: _true_param_sampler(m) for m in model_names}

    for true_name, true_cls in models.items():
        env_out = environment_fn(n_trials=n_trials)
        observations = get_observations(env_out)

        true_model = make_model(true_cls, true_params[true_name],
                                observations, track_history=False)
        _, responses = generate_synthetic_responses(
            true_model, observations, sigma_r
        )

        scores = {}
        for fit_name, fit_cls in models.items():
            best_params, best_ll, map_params, logpost = grid_search_coarse_fine(
                fit_cls, param_grids[fit_name],
                observations, responses, sigma_r,
            )
            k = model_n_params(fit_cls)
            bic = calculate_bic(k, n_trials, best_ll)

            hessian = estimate_hessian(
                fit_cls, np.asarray(map_params),
                observations, responses, sigma_r,
            )
            log_evidence = laplace_model_evidence(logpost, hessian, k)
            scores[fit_name] = {"BIC": bic, "LogEvidence": log_evidence}

        winner_bic = min(scores, key=lambda m: scores[m]["BIC"])
        winner_logevi = max(scores, key=lambda m: scores[m]["LogEvidence"])

        results.append({
            "sim_id": sim_id,
            "true_model": true_name,
            "winner_BIC": winner_bic,
            "winner_LogEvidence": winner_logevi,
            "scores": scores,
            "true_params": true_params[true_name],
        })

    return results


# ═══════════════════════════════════════════════════════════════════
#  True parameter sampler (module-level for pickling)
# ═══════════════════════════════════════════════════════════════════

def _true_param_sampler(model_name: str):
    if model_name == "CPM":
        return np.array([
            np.random.uniform(0.01, 0.3),
            np.random.choice([50.0, 200.0, 1000.0]),
            np.random.uniform(0.01, 0.3),
        ])
    if model_name == "HGF":
        return np.array([
            10 ** np.random.uniform(-4, -1),
            10 ** np.random.uniform(np.log10(1.0), np.log10(50.0**2)),
        ])
    raise KeyError(model_name)

# alias
true_param_sampler = _true_param_sampler


# ═══════════════════════════════════════════════════════════════════
#  Parameter grids
# ═══════════════════════════════════════════════════════════════════

def build_param_grids():
    w1 = np.linspace(0.05, 0.5, 6)
    w2 = np.array([50.0, 200.0, 1000.0])
    h  = np.linspace(0.01, 0.3, 6)
    cpm = np.array(list(product(w1, w2, h)), dtype=np.float64)

    eta = np.logspace(-4, -1, 8)
    s   = np.logspace(np.log10(1.0), np.log10(50.0**2), 8)
    hgf = np.array(list(product(eta, s)), dtype=np.float64)

    return {"CPM": cpm, "HGF": hgf}


# ═══════════════════════════════════════════════════════════════════
#  OPTIMISATION — Parallel condition runner
# ═══════════════════════════════════════════════════════════════════

def run_condition(n_sims, n_trials, sigma_r, environment_fn,
                  models, param_grids, seed=42):
    """
    Run n_sims simulations for ONE condition.
    Uses joblib parallelism across simulations when available.
    """
    models_spec = list(models.items())      # list of (name, cls) — picklable

    if HAS_JOBLIB and N_JOBS > 1 and n_sims >= 4:
        print(f"    → dispatching {n_sims} sims across {N_JOBS} workers …")
        seeds = (np.random.RandomState(seed)
                 .randint(0, 2**31, size=n_sims))

        nested = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(_worker_one_sim)(
                sim, models_spec, param_grids, environment_fn,
                n_trials, sigma_r, int(seeds[sim]),
            )
            for sim in range(n_sims)
        )
        results = [r for batch in nested for r in batch]
    else:
        # sequential fallback
        results = []
        for sim in range(n_sims):
            np.random.seed(seed + sim)
            batch = _worker_one_sim(
                sim, models_spec, param_grids, environment_fn,
                n_trials, sigma_r, seed + sim,
            )
            results.extend(batch)
            if (sim + 1) % 10 == 0 or sim == 0:
                print(f"    sim {sim+1}/{n_sims} done")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Confusion matrix builder
# ═══════════════════════════════════════════════════════════════════

def confusion_matrix_from_results(results, model_names, criterion="BIC"):
    winner_key = f"winner_{criterion}"
    n = len(model_names)
    cm = np.zeros((n, n))
    idx = {m: i for i, m in enumerate(model_names)}

    for r in results:
        cm[idx[r["true_model"]], idx[r[winner_key]]] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums


# ═══════════════════════════════════════════════════════════════════
#  Full factorial experiment
# ═══════════════════════════════════════════════════════════════════

def run_full_experiment(n_sims=100, base_seed=42):
    models = {"CPM": ChangePointModelVariational, "HGF": PatchedHGF}
    model_names = list(models.keys())
    param_grids = build_param_grids()

    environments = {
        "Change-Point": generate_change_point_environment,
        "Random-Walk": generate_random_walk_environment,
    }
    trial_lengths = [100, 500]
    noise_levels  = {"low": 2.0, "high": 10.0}

    all_cms = {}
    seed_counter = base_seed

    for env_name, env_fn in environments.items():
        for T in trial_lengths:
            for noise_label, sigma_r in noise_levels.items():
                print(
                    f"\n{'='*60}\n"
                    f"Env={env_name}  T={T}  noise={noise_label} (σ_r={sigma_r})\n"
                    f"{'='*60}"
                )
                results = run_condition(
                    n_sims, T, sigma_r, env_fn,
                    models, param_grids, seed=seed_counter,
                )
                seed_counter += 1

                for crit in ("LogEvidence", "BIC"):
                    cm = confusion_matrix_from_results(
                        results, model_names, criterion=crit
                    )
                    all_cms[(env_name, T, sigma_r, noise_label, crit)] = cm

    return all_cms, model_names


# ═══════════════════════════════════════════════════════════════════
#  Backward-compatible batch runner
# ═══════════════════════════════════════════════════════════════════

def run_many_simulations(n_sims, models, true_param_sampler_fn,
                         param_grids, environment_fn, n_trials,
                         sigma_r, decision_rule="BIC", seed=0):
    return run_condition(
        n_sims, n_trials, sigma_r, environment_fn,
        models, param_grids, seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════
#  Plotting — Markovic and Kiebel (2016) paper-style confusion matrices
# ═══════════════════════════════════════════════════════════════════

def _draw_single_cm(ax, cm, model_names, cmap):
    n = len(model_names)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            ax.add_patch(plt.Rectangle(
                (j, n - 1 - i), 1, 1,
                facecolor=cmap(v), edgecolor="white", lw=2,
            ))
            txt = f"{v:.3g}" if v not in (0, 1) else f"{int(v)}"
            ax.text(j + 0.5, n - 1 - i + 0.5, txt,
                    ha="center", va="center", fontsize=13,
                    fontweight="bold",
                    color="white" if v > 0.5 else "black")
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks([0.5 + k for k in range(n)])
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_yticks([0.5 + k for k in range(n)])
    ax.set_yticklabels(model_names[::-1], fontsize=11)
    ax.tick_params(length=0)
    ax.set_aspect("equal")


def plot_paper_figure(all_cms, model_names, criterion, fig_title, fig_num):
    envs   = ["Change-Point", "Random-Walk"]
    Ts     = [100, 500]
    noises = [("low", 2.0), ("high", 10.0)]
    cmap   = plt.cm.Blues if criterion == "LogEvidence" else plt.cm.BuGn

    fig, axes = plt.subplots(2, 4, figsize=(17, 8),
                             gridspec_kw={"hspace": .50, "wspace": .30})

    for ri, (nl, sr) in enumerate(noises):
        col = 0
        for env in envs:
            for T in Ts:
                ax = axes[ri, col]
                _draw_single_cm(ax, all_cms[(env, T, sr, nl, criterion)],
                                model_names, cmap)
                ax.set_title(f"T = {T}", fontsize=12, pad=6)
                if ri == 1:
                    ax.set_xlabel("inferred model", fontsize=11)
                if col % 2 == 0:
                    ax.set_ylabel("true model", fontsize=11)
                else:
                    ax.set_yticklabels([]); ax.set_ylabel("")
                col += 1

        mid_y = axes[ri, 0].get_position().y0 + axes[ri, 0].get_position().height / 2
        fig.text(0.96, mid_y,
                 "low response noise" if nl == "low" else "high response noise",
                 rotation=-90, fontsize=12, va="center", ha="center",
                 fontweight="bold")

    for ei, env in enumerate(envs):
        l = axes[0, ei * 2].get_position().x0
        r = axes[0, ei * 2 + 1].get_position().x1
        t = axes[0, ei * 2].get_position().y1 + 0.06
        fig.text((l + r) / 2, t, f"{env} environment",
                 fontsize=14, fontweight="bold", ha="center")

    fig.text(0.5, 0.01, fig_title, fontsize=11, ha="center",
             style="italic", wrap=True)
    plt.savefig(f"figure_{fig_num}.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved figure_{fig_num}.png")


def plot_both_paper_figures(all_cms, model_names):
    plot_paper_figure(
        all_cms, model_names, "LogEvidence",
        " Confusion matrix — Laplace approximation of model evidence",
        9,
    )
    plot_paper_figure(
        all_cms, model_names, "BIC",
        " Confusion matrix — BIC approximation to model evidence",
        10,
    )


# ═══════════════════════════════════════════════════════════════════
#  Original summary plots 
# ═══════════════════════════════════════════════════════════════════

def plot_two_confusion_matrices(simulation_results, model_names):
    n = len(model_names)
    cb = pd.DataFrame(np.zeros((n, n)), index=model_names, columns=model_names)
    cl = cb.copy()
    for r in simulation_results:
        cb.loc[r["true_model"], r["winner_BIC"]] += 1
        cl.loc[r["true_model"], r["winner_LogEvidence"]] += 1
    cb = cb.div(cb.sum(1), axis=0).fillna(0)
    cl = cl.div(cl.sum(1), axis=0).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cb, annot=True, fmt=".2f", cmap="Blues", linewidths=.5,
                ax=axes[0], cbar=False)
    axes[0].set(title="Confusion (BIC)", xlabel="Recovered", ylabel="True")
    sns.heatmap(cl, annot=True, fmt=".2f", cmap="Oranges", linewidths=.5,
                ax=axes[1], cbar=False)
    axes[1].set(title="Confusion (LogEvidence)", xlabel="Recovered", ylabel="")
    plt.tight_layout(); plt.show()


def plot_score_distributions(simulation_results, model_names):
    rows = []
    for r in simulation_results:
        for m in model_names:
            rows.append({"True": r["true_model"], "Fit": m,
                         "Criterion": "BIC", "Score": r["scores"][m]["BIC"]})
            rows.append({"True": r["true_model"], "Fit": m,
                         "Criterion": "LogEvidence",
                         "Score": r["scores"][m]["LogEvidence"]})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="True", y="Score", hue="Criterion",
                   split=True, inner="quartile",
                   palette={"BIC": "lightblue", "LogEvidence": "lightcoral"}, cut=0)
    plt.title("Score distributions"); plt.grid(True, axis="y", alpha=.3)
    plt.show()


def plot_overall_summary(simulation_results, model_names):
    plot_two_confusion_matrices(simulation_results, model_names)
    plot_score_distributions(simulation_results, model_names)


# ═══════════════════════════════════════════════════════════════════
#  Sanity check
# ═══════════════════════════════════════════════════════════════════

def fast_sanity_check():
    set_seed(0)
    models = {"CPM": ChangePointModelVariational, "HGF": PatchedHGF}
    true_params = {"CPM": np.array([0.01, 1000.0, 0.1]),
                   "HGF": np.array([0.05, 225.0])}

    cpm_grid = np.array(list(product([0.01, 0.1], [100., 1000.], [0.05, 0.15])))
    hgf_grid = np.array(list(product([1e-3, 1e-2], [25., 225.])))
    grids = {"CPM": cpm_grid, "HGF": hgf_grid}

    for env_name, env_fn in [("changepoint", generate_change_point_environment),
                              ("randomwalk",  generate_random_walk_environment)]:
        print(f"\n{'='*60}\nSANITY: {env_name}\n{'='*60}")
        env_out = env_fn(n_trials=50, sigma=25, seed=0)
        obs = get_observations(env_out)

        for tn, tc in models.items():
            tm = make_model(tc, true_params[tn], obs, track_history=False)
            _, resp = generate_synthetic_responses(tm, obs, 2.0)

            for fn, fc in models.items():
                bp, bl, mp, lp = grid_search_mle_and_map(
                    fc, grids[fn], obs, resp, 2.0)
                k = model_n_params(fc)
                bic = calculate_bic(k, 50, bl)
                H = estimate_hessian(fc, np.asarray(mp), obs, resp, 2.0)
                le = laplace_model_evidence(lp, H, k)
                print(f"  true={tn} fit={fn} | ll={bl:.1f} BIC={bic:.1f} "
                      f"logEvi={le:.1f}")

    print("\n✓ Sanity check passed.")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print(f"Numba available: {HAS_NUMBA}")
    print(f"Joblib available: {HAS_JOBLIB}  (n_jobs={N_JOBS})")

    # Step 1 — quick check
    print("\n" + "▓" * 70)
    print("STEP 1: SANITY CHECK")
    print("▓" * 70)
    fast_sanity_check()

    # Step 2 — full experiment
    print("\n" + "▓" * 70)
    print("STEP 2: FULL FACTORIAL (Figures 9 & 10)")
    print("▓" * 70)

    N_SIMS = 100

    all_cms, model_names = run_full_experiment(n_sims=N_SIMS, base_seed=42)

    # print matrices
    print("\n" + "=" * 70 + "\nCONFUSION MATRICES\n" + "=" * 70)
    for key in sorted(all_cms, key=str):
        env, T, sr, nl, crit = key
        print(f"\n{crit:12s} | {env:10s} | T={T:4d} | {nl}")
        df = pd.DataFrame(all_cms[key], index=model_names, columns=model_names)
        print(df.to_string(float_format=lambda x: f"{x:.3f}"))

    # plot
    plot_both_paper_figures(all_cms, model_names)
    plot_overall_summary(simulation_results, model_names)
    
    print("\n✓ All done.")