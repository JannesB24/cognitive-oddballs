from cognitive_oddballs.environments.change_point_oddball import generate_change_point_environment
from cognitive_oddballs.environments.random_walk_oddball import generate_random_walk_environment


def main():
    _ = generate_change_point_environment(
        n_trials=100, change_point_hazard_rate=0.1, oddball_hazard_rate=0.2, sigma=25, seed=42
    )

    _ = generate_random_walk_environment(
        n_trials=100, drift_sigma=5, oddball_hazard_rate=0.02, sigma=25, seed=42
    )

    nassar_model = ChangePointNassarModel(X=df["x"], sigma_sequence=df["sigma"])
    normative_model_results = nassar_model.run()
    print(normative_model_results)


if __name__ == "__main__":
    main()
