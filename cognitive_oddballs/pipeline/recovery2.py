"""
Model recovery on oddball tasks
MLE-BIC vs Bayesian Inference (Laplace Approximation)

Based on Marković & Kiebel (2016)
"""
# Imports
from collections.abc import Callable

import numpy as np
from environments.change_point_oddball import generate_change_point_environment
from environments.random_walk_oddball import generate_random_walk_environment
from models.change_point_model_variational import ChangePointModelVariational
from models.hgf.hgf2_gaussian import HGFPaper2Gaussian


# TODO: What’s left to do for real model recovery

# Run many simulations

# Loop over n_sims (e.g. 100–500), not just one dataset.

# Pick a decision rule

# Either BIC (robust, already works)

# Or Laplace log-evidence (needs Hessian fixes)

# Select winning model per simulation

# winner = argmin(BIC) or argmax(log_evidence)

# Build confusion matrices

# Rows = true model

# Columns = recovered model

# Values = proportions

# Do parameter recovery

# Plot true vs recovered params (only when model is correctly identified)

# Vary true parameters

# Don’t use one fixed set

# Sample across regimes (low/high hazard, low/high volatility)

# Fix Laplace (optional but needed if used)

# Add priors and/or Hessian regularization

# Otherwise stick to BIC


# Utilities and helper functions
def model_n_params(model_cls) -> int:
    # Prefer class attribute if present, else fall back to constructor args count
    if model_cls.__name__ == "ChangePointModelVariational":
        return 3  # w1, w2, h
    # HGFPaper2Gaussian has 2 core params: eta, s
    if model_cls.__name__ == "HGFPaper2Gaussian":
        return 2
    raise AttributeError(f"Unknown n_params for {model_cls.__name__}")

def safe_log_prior(model, params) -> float:
    """
    If model has log_prior(): use it.
    Otherwise return 0 (flat prior), turning MAP into ML.
    """
    lp_fn = getattr(model, "log_prior", None)
    if callable(lp_fn):
        return float(lp_fn(params))
    return 0.0

def set_seed(seed: int = 42):
    np.random.seed(seed)

# Model constructor function 
def make_model(model_cls, params, observations, *, obs_noise=25.0, sigma0=25.0, add_second_level=True):
    if model_cls.__name__ == "ChangePointModelVariational":
        params = np.asarray(params, dtype=float).ravel()
        if params.size != 3:
            raise ValueError(f"CPM params must be (w1,w2,h). Got {params}")

        w1, w2, h = map(float, params)

        mu0 = float(observations[0])  # common choice for sanity checks
        return model_cls(
            observations,
            mu0=mu0,
            sigma0=float(sigma0),
            obs_noise=float(obs_noise),
            w1=w1,
            w2=w2,
            h=h,
            add_second_level=add_second_level,
        )
    # HGF (eta, s)
    return model_cls(*params)

# Environments output normalization
def get_observations(env_out):
    """
    Normalize environment output to a 1D numpy array of observations (shape (T,)).
    Supports:
      - pandas DataFrame with column 'x' or 'o' (preferred)
      - dict with key 'x' or 'o' or 'observations'
      - tuple/list where first element is the observation array
      - array-like already 1D
    """
    # pandas DataFrame
    if hasattr(env_out, "columns"):
        if "x" in env_out.columns:
            return np.asarray(env_out["x"].values, dtype=float).ravel()
        if "o" in env_out.columns:
            return np.asarray(env_out["o"].values, dtype=float).ravel()
        # fall back: first column
        return np.asarray(env_out.iloc[:, 0].values, dtype=float).ravel()

    # dict
    if isinstance(env_out, dict):
        for key in ("x", "o", "observations"):
            if key in env_out:
                return np.asarray(env_out[key], dtype=float).ravel()
        raise KeyError(f"Env dict keys: {list(env_out.keys())}")

    # tuple/list
    if isinstance(env_out, (tuple, list)):
        return np.asarray(env_out[0], dtype=float).ravel()

    # array-like
    arr = np.asarray(env_out, dtype=float)
    if arr.ndim != 1:
        # if it's 2D, try first column
        arr = arr[:, 0]
    return arr.ravel()


# Response likelihood

def response_log_likelihood(responses, beliefs, sigma_r):
    """
    log p(r_t | mu_t, sigma_r) with r_t ~ Normal(mu_t, sigma_r^2)
    sigma_r can be scalar or shape (T,)
    """
    responses = np.asarray(responses)
    beliefs = np.asarray(beliefs)
    sigma_r = np.asarray(sigma_r)

    if np.any(sigma_r<=0):
        return -np.inf
    
    residuals = responses - beliefs
    var = sigma_r**2

    return -0.5 * np.sum((residuals**2) / var + np.log(2 * np.pi * var))

# Core simulation loop

def run_model_on_environment(model, observations, sigma_r, generate_responses=True, fixed_responses=None):
    if (not generate_responses) and (fixed_responses is None):
        raise ValueError("fixed_responses must be provided when generate_responses=False")

    observations = np.asarray(observations)

    beliefs = np.zeros(len(observations), dtype=float)
    responses = np.zeros(len(observations), dtype=float)
    prediction_errors = np.zeros(len(observations), dtype=float)
    updates = np.full(len(observations), np.nan, dtype=float)

    # ---- Case 1: CPM-style model (uses self.x and update(t)) ----
    # ChangePointModelVariational has attributes x, mu and update(t) reads self.x[t]. :contentReference[oaicite:4]{index=4}
    if model.__class__.__name__ == "ChangePointModelVariational":
        # attach the sequence
        model.n_trials = len(observations)

        # "reset" to initial state like run() does :contentReference[oaicite:5]{index=5}
        model.mu = model.mu0
        model.sigma = model.sigma0
        if getattr(model, "add_second_level", False):
            model.mu2 = 0.0
            model.sigma2 = 1.0

        # trial 0: no update in their run(); belief is initial mu0 :contentReference[oaicite:6]{index=6}
        mu = float(model.mu)
        beliefs[0] = mu
        prediction_errors[0] = observations[0] - mu
        responses[0] = (mu + np.random.randn() * sigma_r) if generate_responses else float(fixed_responses[0])

        # trials 1..T-1: predict = current mu (from previous posterior), then update(t)
        for t in range(1, len(observations)):
            mu = float(model.mu)  # prior/prediction for this trial
            beliefs[t] = mu
            prediction_errors[t] = observations[t] - mu

            model.update(t)  # CPM update uses index t :contentReference[oaicite:7]{index=7}
            # you can optionally expose "last_update" later; for now keep NaN

            responses[t] = (mu + np.random.randn() * sigma_r) if generate_responses else float(fixed_responses[t])

        return {
            "beliefs": beliefs,
            "responses": responses,
            "prediction_errors": prediction_errors,
            "updates": updates,
        }

    # ---- Case 2: HGF-style model (has mu1 and update(obs)) ----
    if hasattr(model, "mu1") and callable(getattr(model, "update", None)):
        # no reset available => rely on fresh instantiation in grid-search
        for t, obs in enumerate(observations):
            mu = float(model.mu1)  # predicted mean before seeing obs
            beliefs[t] = mu
            prediction_errors[t] = float(obs) - mu

            model.update(obs)

            responses[t] = (mu + np.random.randn() * sigma_r) if generate_responses else float(fixed_responses[t])

        return {
            "beliefs": beliefs,
            "responses": responses,
            "prediction_errors": prediction_errors,
            "updates": updates,
        }

    raise TypeError(f"Unsupported model interface: {type(model)}")


# MLE grid search + BIC


def calculate_bic(k, n_trials, ll):
    return k * np.log(n_trials) - 2 * ll


def grid_search_mle(model_cls, param_grid, observations, responses, sigma_r):
    best_ll = -np.inf
    best_params = None

    for params in param_grid:
        model = make_model(model_cls, params, observations)

        outputs = run_model_on_environment(
            model, observations, sigma_r, generate_responses=False, fixed_responses=responses
        )

        ll = response_log_likelihood(responses, outputs["beliefs"], sigma_r)

        if not np.isfinite(ll):
            continue

        if ll > best_ll:
            best_ll = ll
            best_params = params

    return best_params, best_ll


# MAP estimation (Bayesian inference)


def grid_search_map(model_cls, param_grid, observations, responses, sigma_r):
    best_logpost = -np.inf
    best_params = None

    for params in param_grid:
        model = make_model(model_cls, params, observations)

        outputs = run_model_on_environment(
            model, observations, sigma_r, generate_responses=False, fixed_responses=responses
        )

        ll = response_log_likelihood(responses, outputs["beliefs"], sigma_r)

        if not np.isfinite(ll):
            continue

        lp = safe_log_prior(model, params)

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
        model = make_model(model_cls, params, observations)
        outputs = run_model_on_environment(
            model, observations, sigma_r, generate_responses=False, fixed_responses=responses
        )
        ll = response_log_likelihood(responses, outputs["beliefs"], sigma_r)
        return -(ll + safe_log_prior(model, params))

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
    sign, logdet = np.linalg.slogdet(hessian)
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf
    return logpost_map + 0.5 * k * np.log(2 * np.pi) - 0.5 * logdet


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
        env_out = environment_fn(n_trials=n_trials)
        observations = get_observations(env_out)
        true_model = make_model(true_model_cls, true_params[true_name], observations)

        synth = run_model_on_environment(true_model, observations, sigma_r, generate_responses=True)

        responses = synth["responses"]

        results[true_name] = {}

        for fit_name, fit_model_cls in models.items():
            # ----- MLE + BIC -----
            best_params, best_ll = grid_search_mle(
                fit_model_cls, param_grids[fit_name], observations, responses, sigma_r
            )

            k = model_n_params(fit_model_cls)
            bic = calculate_bic(k, n_trials, best_ll)

            # ----- Bayesian (Laplace) -----
            map_params, logpost = grid_search_map(
                fit_model_cls, param_grids[fit_name], observations, responses, sigma_r
            )

            hessian = estimate_hessian(
                fit_model_cls, np.asarray(map_params), observations, responses, sigma_r
            )

            log_evidence = laplace_model_evidence(logpost, hessian, k)

            results[true_name][fit_name] = {
                "MLE": {"best_params": best_params, "loglik": best_ll, "BIC": bic},
                "Bayesian": {"MAP": map_params, "log_evidence": log_evidence},
            }

    return results

# Experiment Wrappers

def modelrec_changepoint():
    models = {"CPM": ChangePointModelVariational, "HGF": HGFPaper2Gaussian}

    # ---- True params must match make_model() ordering ----
    # CPM expects (w1, w2, h) after x (and possibly others depending on your __init__)
    true_params = {
        "CPM": np.array([0.01, 1000.0, 0.1]),  # w1 small, w2 large, hazard moderate,          # w1, w2, h  (example)
        "HGF": np.array([0.05, 15.0**2]),          # eta, s     (example)
    }

    # ---- Param grids ----
    w1_grid = np.linspace(0.05, 0.5, 6)
    w2_grid = np.array([50.0, 200.0, 1000.0])
    h_grid  = np.linspace(0.01, 0.3, 6)
    param_grids = {
        "CPM": np.array([(w1, w2, h) for w1 in w1_grid for w2 in w2_grid for h in h_grid], dtype=float)
    }

    eta_grid = np.logspace(-4, -1, 8)
    s_grid   = np.logspace(np.log10(1.0), np.log10(50.0**2), 8)
    param_grids["HGF"] = np.array([(e, s) for e in eta_grid for s in s_grid], dtype=float)

    return model_recovery_per_env(
        models=models,
        true_params=true_params,
        param_grids=param_grids,
        environment_fn=generate_change_point_environment,
        n_trials=300,
        sigma_r=5.0,
    )


def modelrec_randomwalk():
    models = {"CPM": ChangePointModelVariational}

    true_params = {"CPM": np.array([0.01, 1000.0, 0.1])}

    w1_grid = np.linspace(0.05, 0.5, 6)
    w2_grid = np.array([50.0, 200.0, 1000.0])
    h_grid  = np.linspace(0.01, 0.3, 6)
    param_grids = {
        "CPM": np.array([(w1, w2, h) for w1 in w1_grid for w2 in w2_grid for h in h_grid], dtype=float)
    }

    return model_recovery_per_env(
        models=models,
        true_params=true_params,
        param_grids=param_grids,
        environment_fn=generate_random_walk_environment,
        n_trials=300,
        sigma_r=5.0,
    )

# Fast sanity check
# what this sanity check is for?
# This run is not meant to prove model recovery yet.
# It only answers four yes/no questions:
# Can each model generate data?
# Can each model fit its own data?
# Can each model fail badly on the other model’s data?
# Do the model comparison scores reflect that?

# all answers are yes according to the last run
def fast_sanity_check():
    """
    Quick end-to-end check:
    - small n_trials
    - tiny param grids
    - runs both envs
    - runs MLE+BIC and (optionally) Laplace
    """

    set_seed(0)

    # Keep it small so it runs fast
    n_trials = 50
    sigma_r = 2.0

    models = {"CPM": ChangePointModelVariational, "HGF": HGFPaper2Gaussian}

    # True params (just placeholders for sanity)
    true_params = {
        "CPM": np.array([0.01, 1000.0, 0.1]),  # w1 small, w2 large, hazard moderate
        "HGF": np.array([0.05, 15.0**2]),       # eta, s (variance)
    }

    # Tiny grids (2–3 values each)
    w1_grid = np.array([0.01, 0.1])
    w2_grid = np.array([100.0, 1000.0])
    h_grid  = np.array([0.05, 0.15])
    cpm_grid = np.array([(w1, w2, h) for w1 in w1_grid for w2 in w2_grid for h in h_grid], dtype=float)

    eta_grid = np.array([1e-3, 1e-2])
    s_grid   = np.array([5.0**2, 15.0**2])
    hgf_grid = np.array([(e, s) for e in eta_grid for s in s_grid], dtype=float)

    param_grids = {"CPM": cpm_grid, "HGF": hgf_grid}

    envs = {
        "changepoint": generate_change_point_environment,
        "randomwalk": generate_random_walk_environment,
    }

    # Optional: skip Hessian for speed on first test
    do_laplace = True

    for env_name, env_fn in envs.items():
        print("\n" + "=" * 60)
        print(f"SANITY CHECK ENV: {env_name} | n_trials={n_trials} | sigma_r={sigma_r}")

        # Generate one dataset once (shared across generator models for speed)
        env_out = env_fn(n_trials=n_trials, sigma=25, seed=0)
        observations = get_observations(env_out)

        for true_name, true_model_cls in models.items():
            print(f"\n--- Generating with {true_name} ---")

            true_model = make_model(true_model_cls, true_params[true_name], observations)
            synth = run_model_on_environment(true_model, observations, sigma_r, generate_responses=True)
            responses = synth["responses"]

            for fit_name, fit_model_cls in models.items():
                # MLE + BIC
                best_params, best_ll = grid_search_mle(
                    fit_model_cls, param_grids[fit_name], observations, responses, sigma_r
                )
                k = model_n_params(fit_model_cls)
                bic = calculate_bic(k, n_trials, best_ll)

                msg = f"fit={fit_name:3s} | ll={best_ll: .2f} | bic={bic: .2f} | params={best_params}"
                if not do_laplace:
                    print(msg)
                    continue

                # MAP + Laplace (optional)
                map_params, logpost = grid_search_map(
                    fit_model_cls, param_grids[fit_name], observations, responses, sigma_r
                )
                hessian = estimate_hessian(
                    fit_model_cls, np.asarray(map_params), observations, responses, sigma_r, eps=1e-4
                )
                log_evidence = laplace_model_evidence(logpost, hessian, k)
                print(msg + f" | logev={log_evidence: .2f}")

    print("\nSanity check finished.")


# Main


if __name__ == "__main__":
    fast_sanity_check()
