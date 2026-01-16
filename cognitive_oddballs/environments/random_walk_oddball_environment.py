import numpy as np
import pandas as pd

BAG_MIN_POS = 0
BAG_MAX_POS = 500


def generate_random_walk_environment(
    n_trials=400,
    oddball_hazard_rate=0.1,
    sigma=25,
    drift_sigma=10,  # standard deviation of random-walk drift, 5 because medium step sizes -> smooth continuous drift
    seed=555,
) -> pd.DataFrame:
    """
    Generate true helicopter and observed bag positions for an oddball environment with
    RANDOM-WALK.
        - Helicopter moves according to a Gaussian random walk
            each trial (drift_sigma)
        - Occasional oddball bag drops (oddball_hazard_rate)
        - Static noise level (sigma) throughout all trials

    The helicopter is positioned x in [3 * sigma, 500 - 3 * sigma]
        to avoid edge effects (The rest is clipped).
    The bag can be dropped anywhere in [0, 500] with respect to oddballs and noise.

    Helicopter:
        mu_t = mu_{t-1} + epsilon,   epsilon ~ N(0, drift_sigma)

    Bag:
        - Normal trials: x ~ N(mu_t, sigma)
        - Oddball trials: x ~ Uniform(0, 500)

    Args:
        n_trials (int): Number of trials to simulate
        oddball_hazard_rate (float): Probability of an oddball bag drop on each trial
        sigma (float): Standard deviation of noise for bag drops from helicopter
        drift_sigma (float): Standard deviation of random-walk drift
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: DataFrame with columns:
            - trial: Trial number (0 to n_trials-1)
            - mu: True helicopter positions for each trial
            - x: Observed bag drop positions for each trial
            - sigma: Noise levels for each trial (constant in this case)
            - is_oddball: Boolean indicating which trials are oddballs
            - drift: Random-walk drift step for each trial

    Usage:
        Read from 'x' column for each trial and check 'is_oddball' to see if it's an oddball trial.
    """
    helicopter_min = 3 * sigma
    helicopter_max = BAG_MAX_POS - 3 * sigma

    np.random.seed(seed)  # reproducibility

    # Initialize DataFrame with trial numbers
    df = pd.DataFrame({"trial": range(n_trials)})

    # Initialize columns
    df["mu"] = 0.0  # true helicopter position
    df["x"] = 0.0  # observed bag drops
    df["sigma"] = sigma  # noise level (constant)
    df["is_oddball"] = False  # track oddball trials
    df["drift"] = 0.0  # track random-walk drift steps

    # Initialize helicopter in center of range
    df.loc[0, "mu"] = BAG_MAX_POS / 2  # Start at center
    df.loc[0, "x"] = np.random.normal(df.loc[0, "mu"], sigma)  # initial bag drop location

    # Generate trials
    for t in range(1, n_trials):
        # Random walk drift
        drift_step = np.random.normal(0, drift_sigma)
        df.loc[t, "drift"] = drift_step

        # New helicopter position with clipping to avoid edges
        new_mu = df.loc[t - 1, "mu"] + drift_step
        df.loc[t, "mu"] = np.clip(new_mu, helicopter_min, helicopter_max)

        # Oddball decision: bag from random location or helicopter location?
        if np.random.rand() < oddball_hazard_rate:
            # Oddball: bag from uniform distribution (anywhere on screen)
            new_x = np.random.uniform(BAG_MIN_POS, BAG_MAX_POS + 1)
            df.loc[t, "x"] = np.clip(new_x, BAG_MIN_POS, BAG_MAX_POS)
            df.loc[t, "is_oddball"] = True
        else:
            # Normal: bag from helicopter location with noise
            df.loc[t, "x"] = np.random.normal(df.loc[t, "mu"], sigma)
            df.loc[t, "x"] = np.clip(df.loc[t, "x"], BAG_MIN_POS, BAG_MAX_POS)

    return df
