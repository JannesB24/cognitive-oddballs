from environments.change_point_oddball_environment import generate_oddball_environment
from visualizer import visualize_environment, visualize_summary


def main():
    """Main entry point."""
    # Generate environment
    df = generate_oddball_environment(
        n_trials=50, oddball_hazard_rate=0.15, sigma=20, change_point_hazard_rate=0.1, seed=42
    )

    # Visualize
    visualize_environment(df, delay=2.0)

    # Show summary
    visualize_summary(df)


if __name__ == "__main__":
    main()
