"""
Model recovery on oddball tasks
MLE-BIC vs Bayesian Inference (Laplace Approximation)

Based on Marković & Kiebel (2016)
"""
#no special libraries needed as in recovery-optimized.py

# Imports
from collections.abc import Callable

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from cognitive_oddballs.environments.change_point_oddball import generate_change_point_environment
from cognitive_oddballs.environments.random_walk_oddball import generate_random_walk_environment
from cognitive_oddballs.models.change_point_model_variational import ChangePointModelVariational
from cognitive_oddballs.models.hgf.hgf2_gaussian import HGFPaper2Gaussian, HGF2Config, exp_clip


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
class PatchedHGF(HGFPaper2Gaussian):
    """
    A patched version of HGFPaper2Gaussian that avoids inefficient concatenation
    and correctly handles initialization to prevent AttributeErrors.
    """
    def __init__(self,
                 eta: float,
                 s: float,
                 mu1_init: float = 0.0,
                 sig1_init: float = 1.0,
                 mu2_init: float = 0.0,
                 sig2_init: float = 1.0,
                 min_var: float = 1e-8,
                 exp_clip_value: float = 60.0,
                 **kwargs): # Added **kwargs for robustness
        
        # We are taking over initialization completely, so we DO NOT call super().__init__()
        
        # 1. Copy the setup logic from the original HGFPaper2Gaussian.__init__
        self.cfg = HGF2Config(
            mu1_0=float(mu1_init), sig1_0=float(sig1_init),
            mu2_0=float(mu2_init), sig2_0=float(sig2_init),
            eta=float(eta), s=float(s),
            min_var=float(min_var), exp_clip_value=float(exp_clip_value),
        )

        if self.cfg.eta <= 0:
            raise ValueError("eta must be > 0")
        if self.cfg.s <= 0:
            raise ValueError("s (observation variance) must > 0")

        self.mu1 = self.cfg.mu1_0
        self.sig1 = max(self.cfg.sig1_0, self.cfg.min_var)
        self.mu2 = self.cfg.mu2_0
        self.sig2 = max(self.cfg.sig2_0, self.cfg.min_var)
        self.trial = 0

        # 2. This is the crucial change: instead of creating a DataFrame, we create our list.
        self.history_list = []

    @property
    def history(self):
        """
        This read-only property makes our class compatible with inherited methods
        (like plot_results) that expect `self.history` to be a DataFrame.
        It builds the DataFrame on-the-fly from our efficient list.
        """
        # Caching the result for performance is a good idea
        if not hasattr(self, '_history_cache') or len(self._history_cache) != len(self.history_list):
            self._history_cache = pd.DataFrame(self.history_list)
        return self._history_cache

    def update(self, o: float) -> None:
        """
        This overridden method is the same as before, populating the history_list.
        """
        o = float(o)
        minvar = self.cfg.min_var

        # --- Prediction (Copied directly from original) ---
        mu2_hat = self.mu2
        sig2_hat = max(self.sig2 + self.cfg.eta, minvar)
        omega = exp_clip(mu2_hat, self.cfg.exp_clip_value)
        mu1_prev = self.mu1
        sig1_prev = self.sig1
        den1 = max(sig1_prev + omega, minvar)

        # --- Update Level 1 (Copied directly from original) ---
        delta1 = o - mu1_prev
        sig1_new = 1.0 / max((1.0 / self.cfg.s) + (1.0 / den1), minvar)
        alpha1 = sig1_new / max(self.cfg.s, minvar)
        eps1 = alpha1 * delta1
        mu1_new = mu1_prev + eps1

        # --- Update Level 2 (Copied directly from original) ---
        delta2 = (sig1_new + eps1 * eps1) / den1 - 1.0
        k = omega / den1
        r = (omega - sig1_prev) / den1
        pi2 = (1.0 / sig2_hat) + 0.5 * k * (k + r * delta2)
        sig2_new = 1.0 / max(pi2, minvar)
        mu2_new = mu2_hat + 0.5 * max(sig2_new, minvar) * k * delta2

        # --- State update of the model (Copied directly from original) ---
        self.mu1, self.mu2 = mu1_new, mu2_new
        self.sig1, self.sig2 = sig1_new, sig2_new
        
        # --- EFFICIENT HISTORY LOGGING ---
        row_dict = {
            "o": o, "mu1_hat": mu1_prev, "sig1_hat": den1, "mu1": self.mu1,
            "sig1": self.sig1, "mu2_hat": mu2_hat, "sig2_hat": sig2_hat,
            "mu2": self.mu2, "sig2": self.sig2, "omega": omega, "delta1": delta1,
            "alpha1": alpha1, "delta2": delta2, "k": k, "r": r,
        }
        self.history_list.append(row_dict)

# Utilities and helper functions
def model_n_params(model_cls) -> int:
    # Prefer class attribute if present, else fall back to constructor args count
    if model_cls.__name__ == "ChangePointModelVariational":
        return 3  # w1, w2, h
    # HGFPaper2Gaussian has 2 core params: eta, s
    if model_cls.__name__ in ("HGFPaper2Gaussian", "PatchedHGF"):
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
def make_model(
    model_cls, params, observations, *, obs_noise_std=25.0, sigma0=25.0, add_second_level=True
):
    if model_cls.__name__ == "ChangePointModelVariational":
        params = np.asarray(params, dtype=float).ravel()
        if params.size != 3:
            raise ValueError(f"CPM params must be (w1,w2,h). Got {params}")

        w1_std, w2_std, h = map(float, params)

        mu0 = float(observations[0])  # common choice for sanity checks
        return model_cls(
            mu0=mu0,
            sigma0=float(sigma0),
            obs_noise_std=float(obs_noise_std),
            w1_std=w1_std,
            w2_std=w2_std,
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
    log p(r_t | mu_t, sigma_r) with r_t ~ Normal(mu_t, sigma_r^2)
    sigma_r can be scalar or shape (T,)
    """
    responses = np.asarray(responses)
    beliefs = np.asarray(beliefs)
    sigma_r = np.asarray(sigma_r)

    if np.any(sigma_r <= 0):
        return -np.inf

    residuals = responses - beliefs
    var = sigma_r**2

    return -0.5 * np.sum((residuals**2) / var + np.log(2 * np.pi * var))
   

# Core simulation loop


def run_model_on_environment(
    model, observations, sigma_r, generate_responses=True, fixed_responses=None
):
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
        responses[0] = (
            (mu + np.random.randn() * sigma_r) if generate_responses else float(fixed_responses[0])
        )

        # trials 1..T-1: predict = current mu (from previous posterior), then update(t)
        for t in range(1, len(observations)):
            mu = float(model.mu)  # prior/prediction for this trial
            beliefs[t] = mu
            prediction_errors[t] = observations[t] - mu

            model.update(t)  # CPM update uses index t :contentReference[oaicite:7]{index=7}
            # you can optionally expose "last_update" later; for now keep NaN

            responses[t] = (
                (mu + np.random.randn() * sigma_r)
                if generate_responses
                else float(fixed_responses[t])
            )

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


            responses[t] = (
                (mu + np.random.randn() * sigma_r)
                if generate_responses
                else float(fixed_responses[t])
            )

        return {
            "beliefs": beliefs,
            "responses": responses,
            "prediction_errors": prediction_errors,
            "updates": updates,
        }
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
    # --- move MAP to unconstrained space ---
    theta_u_map = np.log(map_params)

    k = len(map_params)
    hessian = np.zeros((k, k))

    def neg_log_post_unconstrained(theta_u):
        # back to constrained space
        params = np.exp(theta_u)

        model = make_model(model_cls, params, observations)

        outputs = run_model_on_environment(
            model,
            observations,
            sigma_r,
            generate_responses=False,
            fixed_responses=responses,
        )

        ll = response_log_likelihood(responses, outputs["beliefs"], sigma_r)

        if not np.isfinite(ll):
            return np.inf

        return -(ll + safe_log_prior(model, params))

    # --- finite differences in unconstrained space ---
    for i in range(k):
        for j in range(k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i] = eps
            ej[j] = eps

            hessian[i, j] = (
                neg_log_post_unconstrained(theta_u_map + ei + ej)
                - neg_log_post_unconstrained(theta_u_map + ei - ej)
                - neg_log_post_unconstrained(theta_u_map - ei + ej)
                + neg_log_post_unconstrained(theta_u_map - ei - ej)
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
    models = {"CPM": ChangePointModelVariational, "HGF": PatchedHGF}
    
    # ---- True params must match make_model() ordering ----
    # CPM expects (w1, w2, h) after x (and possibly others depending on your __init__)
    true_params = {
        "CPM": np.array(
            [0.01, 1000.0, 0.1]
        ),  # w1 small, w2 large, hazard moderate,          # w1, w2, h  (example)
        "HGF": np.array([0.05, 15.0**2]),  # eta, s     (example)
    }

    # ---- Param grids ----
    w1_grid = np.linspace(0.05, 0.5, 6)
    w2_grid = np.array([50.0, 200.0, 1000.0])
    h_grid = np.linspace(0.01, 0.3, 6)
    param_grids = {
        "CPM": np.array(
            [(w1, w2, h) for w1 in w1_grid for w2 in w2_grid for h in h_grid], dtype=float
        )
    }

    eta_grid = np.logspace(-4, -1, 8)
    s_grid = np.logspace(np.log10(1.0), np.log10(50.0**2), 8)
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
    h_grid = np.linspace(0.01, 0.3, 6)
    param_grids = {
        "CPM": np.array(
            [(w1, w2, h) for w1 in w1_grid for w2 in w2_grid for h in h_grid], dtype=float
        )
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

    models = {"CPM": ChangePointModelVariational, "HGF": PatchedHGF}

    # True params (just placeholders for sanity)
    true_params = {
        "CPM": np.array([0.01, 1000.0, 0.1]),  # w1 small, w2 large, hazard moderate
        "HGF": np.array([0.05, 15.0**2]),  # eta, s (variance)
    }

    # Tiny grids (2–3 values each)
    w1_grid = np.array([0.01, 0.1])
    w2_grid = np.array([100.0, 1000.0])
    h_grid = np.array([0.05, 0.15])
    cpm_grid = np.array(
        [(w1, w2, h) for w1 in w1_grid for w2 in w2_grid for h in h_grid], dtype=float
    )

    eta_grid = np.array([1e-3, 1e-2])
    s_grid = np.array([5.0**2, 15.0**2])
    hgf_grid = np.array([(e, s) for e in eta_grid for s in s_grid], dtype=float)

    param_grids = {"CPM": cpm_grid, "HGF": hgf_grid}

    envs = {
        "changepoint": generate_change_point_environment,
        "randomwalk": generate_random_walk_environment,
    }

    

    for env_name, env_fn in envs.items():
        print("\n" + "=" * 60)
        print(f"SANITY CHECK ENV: {env_name} | n_trials={n_trials} | sigma_r={sigma_r}")

        # generate one dataset once (shared across models for speed)
        env_out = env_fn(n_trials=n_trials, sigma=25, seed=0)
        observations = get_observations(env_out)

        for true_name, true_model_cls in models.items():
            print(f"\n--- Generating with {true_name} ---")

            true_model = make_model(true_model_cls, true_params[true_name], observations)
            synth = run_model_on_environment(
                true_model, observations, sigma_r, generate_responses=True
            )
            responses = synth["responses"]

            for fit_name, fit_model_cls in models.items():
                # MLE + BIC
                best_params, best_ll = grid_search_mle(
                    fit_model_cls, param_grids[fit_name], observations, responses, sigma_r
                )
                k = model_n_params(fit_model_cls)
                bic = calculate_bic(k, n_trials, best_ll)

                # MAP + Laplace (always computed)
                map_params, logpost = grid_search_map(
                    fit_model_cls, param_grids[fit_name], observations, responses, sigma_r
                )
                hessian = estimate_hessian(
                    fit_model_cls, np.asarray(map_params), observations, responses, sigma_r
                )
                log_evidence = laplace_model_evidence(logpost, hessian, k)

                # ---------------------------------------------------------
                # print BOTH scores together
                # ---------------------------------------------------------
                print(
                    f"fit={fit_name:3s} | ll={best_ll: .2f} | BIC={bic: .2f} | "
                    f"params={best_params} | logEvidence={log_evidence: .2f}"
                )
    print("\nSanity check finished.")


# Main


if __name__ == "__main__":
    fast_sanity_check()




def true_param_sampler(model_name: str):
    """Sample true parameters across regimes."""
    if model_name == "CPM":
        w1 = np.random.uniform(0.01, 0.3)
        w2 = np.random.choice([50.0, 200.0, 1000.0])
        h = np.random.uniform(0.01, 0.3)
        return np.array([w1, w2, h])

    if model_name == "HGF":
        eta = 10 ** np.random.uniform(-4, -1)
        s = 10 ** np.random.uniform(np.log10(1.0), np.log10(50.0**2))
        return np.array([eta, s])

    raise KeyError(model_name)




def run_many_simulations(
    n_sims,
    models,
    true_param_sampler,
    param_grids,
    environment_fn,
    n_trials,
    sigma_r,
    decision_rule="BIC",
    seed=0,
):
    set_seed(seed)

    model_names = list(models.keys())
    
    simulation_results = []

    for sim in range(n_sims):
        print(f"Running simulation {sim + 1}/{n_sims}...")
        true_params = {m: true_param_sampler(m) for m in model_names}

        results_per_env = model_recovery_per_env(
            models=models,
            true_params=true_params,
            param_grids=param_grids,
            environment_fn=environment_fn,
            n_trials=n_trials,
            sigma_r=sigma_r,
        )

        for true_m in model_names:
            scores = {}
            for fit_m in model_names:
                scores[fit_m] = {
                    "BIC": results_per_env[true_m][fit_m]["MLE"]["BIC"],
                    # Laplace log‑evidence (always computed)
                    "LogEvidence": results_per_env[true_m][fit_m]["Bayesian"]["log_evidence"],
                }
            # decide winners for *each* criterion separately
            winner_bic = min(
                scores, key=lambda k: scores[k]["BIC"]
            )  # lowest BIC
            winner_logevi = max(
                scores, key=lambda k: scores[k]["LogEvidence"]
            )  # highest evidence

            # store the MAP/MLE parameters for each winner
            recovered_params_bic = results_per_env[true_m][winner_bic]["MLE"]["best_params"]
            recovered_params_logevi = results_per_env[true_m][winner_logevi]["Bayesian"]["MAP"]
            # -----------------------------------------------------------------
            # 4) record the full result – no filtering
            simulation_results.append(
                {
                    "sim_id": sim,
                    "true_model": true_m,
                    # winners for the two criteria
                    "winner_BIC": winner_bic,
                    "winner_LogEvidence": winner_logevi,
                    # recovered parameter vectors (MLE for BIC, MAP for evidence)
                    "recovered_BIC": recovered_params_bic,
                    "recovered_LogEvidence": recovered_params_logevi,
                    # full scores (both numbers) – useful for later analysis
                    "scores": scores,
                    # the true parameters (kept for later correlation plots)
                    "true_params": true_params[true_m],
                })

            
   
    return simulation_results



# -----------------------------------------------------------------
#  1️.  CONFUSION MATRICES (one for each criterion)
# -----------------------------------------------------------------
def plot_two_confusion_matrices(simulation_results, model_names):
    """
    Build two side‑by‑side heat‑maps:
      • left  – confusion matrix for the winner selected by BIC,
      • right – confusion matrix for the winner selected by Log‑Evidence.
    The matrices are normalised row‑wise (proportions of each true model).
    """
    # --- build count matrices -------------------------------------------------
    n_models = len(model_names)
    conf_bic = pd.DataFrame(np.zeros((n_models, n_models)), index=model_names, columns=model_names)
    conf_logevi = pd.DataFrame(np.zeros((n_models, n_models)), index=model_names, columns=model_names)

    for res in simulation_results:
        true_m = res["true_model"]
        # BIC winner
        conf_bic.loc[true_m, res["winner_BIC"]] += 1
        # Log‑Evidence winner
        conf_logevi.loc[true_m, res["winner_LogEvidence"]] += 1

    # --- convert to proportions ------------------------------------------------
    conf_bic = conf_bic.div(conf_bic.sum(axis=1), axis=0).fillna(0)
    conf_logevi = conf_logevi.div(conf_logevi.sum(axis=1), axis=0).fillna(0)

    # --- plot in a single figure ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(conf_bic, annot=True, fmt=".2f", cmap="Blues", linewidths=.5,
                ax=axes[0], cbar=False)
    axes[0].set_title("Confusion (BIC winner)", fontsize=14)
    axes[0].set_xlabel("Recovered model", fontsize=12)
    axes[0].set_ylabel("True model", fontsize=12)

    sns.heatmap(conf_logevi, annot=True, fmt=".2f", cmap="Oranges", linewidths=.5,
                ax=axes[1], cbar=False)
    axes[1].set_title("Confusion (Log‑Evidence winner)", fontsize=14)
    axes[1].set_xlabel("Recovered model", fontsize=12)
    axes[1].set_ylabel("")  # shared y‑label already on left

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------
#  2️.  SCORE DISTRIBUTIONS (BIC vs. Log‑Evidence)
# -----------------------------------------------------------------
def plot_score_distributions(simulation_results, model_names):
    """
    Violin plot that shows, for each *true* model, the distribution of the
    raw BIC values (lower = better) and the raw Log‑Evidence values
    (higher = better) obtained for every fitted model.
    """
    # --- build a tidy DataFrame ------------------------------------------------
    rows = []
    for res in simulation_results:
        true_m = res["true_model"]
        for fit_m in model_names:
            rows.append({
                "TrueModel": true_m,
                "FitModel": fit_m,
                "Criterion": "BIC",
                "Score": res["scores"][fit_m]["BIC"]
            })
            rows.append({
                "TrueModel": true_m,
                "FitModel": fit_m,
                "Criterion": "LogEvidence",
                "Score": res["scores"][fit_m]["LogEvidence"]
            })
    df = pd.DataFrame(rows)

    # --- plot ---------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="TrueModel", y="Score",
                   hue="Criterion", split=True, inner="quartile",
                   palette={"BIC": "lightblue", "LogEvidence": "lightcoral"},
                   cut=0)
    plt.title("Score distributions per true model", fontsize=16)
    plt.xlabel("True generative model", fontsize=12)
    plt.ylabel("Score (BIC ↓ , Log‑Evidence ↑)", fontsize=12)
    plt.legend(title="Criterion")
    plt.grid(True, axis="y", alpha=0.3)
    plt.show()


# -----------------------------------------------------------------
#  3️.  PARAMETER RECOVERY (both MLE‑based & MAP‑based on same axes)
# -----------------------------------------------------------------
def plot_parameter_recovery(simulation_results, param_names_dict):
    """
    Scatter plots of *true* vs. *recovered* parameters.
    For each model we plot two series on the same axes:
      • recovered by the BIC‑winner (MLE, blue circles)
      • recovered by the Log‑Evidence‑winner (MAP, orange triangles)
    """
    # --- keep *all* simulations (both correctly and incorrectly identified) ---
    df = pd.DataFrame(simulation_results)

    for model_name, param_names in param_names_dict.items():
        # select rows belonging to this true model
        sub = df[df["true_model"] == model_name]

        # true parameter matrix (N × d)
        true_mat = np.vstack(sub["true_params"].values)

        # recovered matrices
        rec_bic_mat = np.vstack(sub["recovered_BIC"].values)
        rec_logevi_mat = np.vstack(sub["recovered_LogEvidence"].values)

        n_params = len(param_names)
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5), squeeze=False)

        for i, pname in enumerate(param_names):
            ax = axes[0, i]

            # true vs. BIC‑recovered (MLE)
            ax.scatter(true_mat[:, i], rec_bic_mat[:, i],
                       alpha=0.6, edgecolor="k", facecolor="steelblue",
                       label="BIC (MLE)", marker="o")

            # true vs. LogEvidence‑recovered (MAP)
            ax.scatter(true_mat[:, i], rec_logevi_mat[:, i],
                       alpha=0.6, edgecolor="k", facecolor="darkorange",
                       label="LogEvidence (MAP)", marker="^")

            # identity line
            lim_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
            lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1)

            ax.set_xlabel(f"True {pname}", fontsize=12)
            ax.set_ylabel(f"Recovered {pname}", fontsize=12)
            ax.set_title(f"{pname}", fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Parameter recovery for {model_name} (n={len(sub)})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


# -----------------------------------------------------------------
#  4️.  OPTIONAL: ONE‑SHOT CONFUSION + SCORE SUMMARY (for quick sanity checks)
# -----------------------------------------------------------------
def plot_overall_summary(simulation_results, model_names):
    """
    Convenience wrapper that calls the three visualisers above.
    Use it after the main experiment finishes:
        plot_overall_summary(simulation_results, model_names)
    """
    plot_two_confusion_matrices(simulation_results, model_names)
    plot_score_distributions(simulation_results, model_names)

    # define the mapping from model name → parameter names (same as before)
    param_names_dict = {
        "CPM": ["w1_std", "w2_std", "h"],
        "HGF": ["eta", "s"]
    }
    plot_parameter_recovery(simulation_results, param_names_dict)


# -----------------------------------------------------------------
#  5️.  Replace the old calls in the __main__ block
# -----------------------------------------------------------------
if __name__ == "__main__":
    set_seed(1)

    models = {
        "CPM": ChangePointModelVariational,
        "HGF": PatchedHGF,
    }

    # Parameter names for the scatter‑plots
    param_names_dict = {
        "CPM": ["w1_std", "w2_std", "h"],
        "HGF": ["eta", "s"]
    }

    # ----- parameter grids -------------------------------------------------
    w1_grid = np.linspace(0.05, 0.5, 6)
    w2_grid = np.array([50.0, 200.0, 1000.0])
    h_grid = np.linspace(0.01, 0.3, 6)
    cpm_grid = np.array([(w1, w2, h) for w1 in w1_grid
                         for w2 in w2_grid for h in h_grid])

    eta_grid = np.logspace(-4, -1, 8)
    s_grid = np.logspace(np.log10(1.0), np.log10(50.0**2), 8)
    hgf_grid = np.array([(e, s) for e in eta_grid for s in s_grid])

    param_grids = {
        "CPM": cpm_grid,
        "HGF": hgf_grid,
    }

    # ----- run the simulations -------------------------------------------
    simulation_results = run_many_simulations(
        n_sims=50,
        models=models,
        true_param_sampler=true_param_sampler,
        param_grids=param_grids,
        environment_fn=generate_change_point_environment,
        n_trials=200,
        sigma_r=5.0,
        decision_rule="BIC",      # kept for compatibility; both scores are stored anyway
    )

    model_names = list(models.keys())

    # ----- NEW unified visualisation ---------------------------------------
    plot_overall_summary(simulation_results, model_names)

    print("\nModel recovery and plotting finished.")

