import numpy as np
import pandas as pd

BAG_MIN_POS = 0
BAG_MAX_POS = 500


def generate_oddball_environment(
    n_trials=400,
    oddball_hazard_rate=0.1,
    sigma=20,
    change_point_hazard_rate=0.1,
    seed=555,
) -> pd.DataFrame:
    """
    Generate true helicopter and observed bag positions for oddball condition:
        - Helicopter changes position with a certain probability
            each trial (change_point_hazard_rate)
        - Occasional oddball bag drops (oddball_hazard_rate)
        - Static noise level (sigma) throughout all trials

    The helicopter is positioned x in [3 * sigma, 500 - 3 * sigma]
        to avoid edge effects (The rest is clipped).
    The bag can be dropped anywhere in [0, 500] with respect to oddballs and noise.

    Args:
        n_trials (int): Number of trials to simulate
        oddball_hazard_rate (float): Probability of an oddball bag drop on each trial
        sigma (float): Standard deviation of noise for bag drops from helicopter
        change_point_hazard_rate (float): Probability of helicopter changing position on each trial
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: DataFrame with columns:
            - trial: Trial number (0 to n_trials-1)
            - mu: True helicopter positions for each trial
            - x: Observed bag drop positions for each trial
            - sigma: Noise levels for each trial (constant in this case)
            - is_oddball: Boolean indicating which trials are oddballs
            - is_change_point: Boolean indicating which trials are change points

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
    df["is_change_point"] = False  # track change point trials

    # Initialize helicopter in center of range
    df.loc[0, "mu"] = BAG_MAX_POS / 2  # Start at center
    df.loc[0, "x"] = np.random.normal(df.loc[0, "mu"], sigma)  # initial bag drop location

    # Generate trials
    for t in range(1, n_trials):
        # Change point decision: does helicopter change position?
        # Allow change point only after 5 trials without one
        recent_cp = [df.loc[i, "is_change_point"] for i in range(max(0, t - 5), t)]

        if np.random.rand() < change_point_hazard_rate and not any(recent_cp):
            # Change Point: change to a new helicopter position within bounds
            new_mu = np.random.uniform(helicopter_min, helicopter_max + 1)
            df.loc[t, "mu"] = np.clip(new_mu, helicopter_min, helicopter_max)
            df.loc[t, "is_change_point"] = True
        else:
            # No Change Point: maintain previous position
            df.loc[t, "mu"] = df.loc[t - 1, "mu"]

        # Oddball decision: bag from random location or helicopter location?
        if np.random.rand() < oddball_hazard_rate:
            # Oddball: bag from uniform distribution (anywhere on screen)
            new_x = np.random.uniform(
                BAG_MIN_POS, BAG_MAX_POS + 1
            )  # drawing float up to 500.99[...]
            df.loc[t, "x"] = np.clip(new_x, BAG_MIN_POS, BAG_MAX_POS)
            df.loc[t, "is_oddball"] = True
        else:
            # Normal: bag from helicopter location with noise
            df.loc[t, "x"] = np.random.normal(df.loc[t, "mu"], sigma)
            df.loc[t, "x"] = np.clip(df.loc[t, "x"], BAG_MIN_POS, BAG_MAX_POS)

    return df
