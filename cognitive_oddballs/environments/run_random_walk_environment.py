from random_walk_oddball_environment import generate_random_walk_environment
from visualizer_for_both_environments import visualize_environment, visualize_summary


def main():
    """Main entry point."""
    # Generate environment
    df = generate_random_walk_environment(
        n_trials=50,
        oddball_hazard_rate=0.15,
        sigma=20,
        drift_sigma=5,
        seed=42,
    )

    # Visualize
    visualize_environment(df, delay=2.0)
    
    # Show summary
    visualize_summary(df)


if __name__ == "__main__":
    main()
