from cognitive_oddballs.environments.change_point_oddball import generate_change_point_environment
from cognitive_oddballs.environments.random_walk_oddball import generate_random_walk_environment


def main():
    _ = generate_change_point_environment(
        n_trials=400, change_point_hazard_rate=0.1, oddball_hazard_rate=0.1, sigma=25, seed=555
    )

    _ = generate_random_walk_environment(
        n_trials=400, drift_sigma=10, oddball_hazard_rate=0.1, sigma=25, seed=555
    )


if __name__ == "__main__":
    main()
