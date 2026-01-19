# THIS CODE WAS AUTOMATICALLY GENERATED WITH THE HELP OF AI LANGUAGE MODEL.

import time

import pandas as pd
from environments.change_point_oddball_environment import BAG_MAX_POS


def visualize_environment(df: pd.DataFrame, delay: float = 2.0, width: int = 80):
    """
    Visualize the helicopter environment in terminal.

    Args:
        df: DataFrame from generate_oddball_environment
        delay: Delay between frames in seconds (0 for no delay)
        width: Width of the visualization bar
    """
    max_pos = BAG_MAX_POS

    for _, row in df.iterrows():
        trial = int(row["trial"])
        mu = row["mu"]
        x = row["x"]
        is_oddball = row["is_oddball"]
        is_change_point = row["is_change_point"]

        # Clear screen and show header
        print("\033[2J\033[H", end="")  # Clear screen, move cursor to top

        print("=" * width)
        print("🎮 HELICOPTER BAG DROP SIMULATION")
        print("=" * width)
        print(f"Trial: {trial:3d} | Helicopter: {mu:6.1f} | Bag: {x:6.1f}")

        # Status indicators
        status = []
        if is_change_point:
            status.append("🔄 CHANGE POINT")
        if is_oddball:
            status.append("⚠️  ODDBALL")
        if not status:
            status.append("✓ Normal drop")
        print(f"Status: {' | '.join(status)}")
        print("-" * width)

        # Create visualization bar
        # Calculate positions on the bar
        heli_pos = int((mu / max_pos) * (width - 1))
        bag_pos = int((x / max_pos) * (width - 1))

        # Build the bar
        bar = [" "] * width
        bar[0] = "|"
        bar[-1] = "|"

        # Add markers (bag first, then helicopter to overlay if same position)
        if 0 <= bag_pos < width:
            bar[bag_pos] = "💰" if not is_oddball else "⚡"
        if 0 <= heli_pos < width:
            if heli_pos == bag_pos:
                bar[heli_pos] = "🎯"  # Perfect drop
            else:
                bar[heli_pos] = "🚁"

        # Print the bar
        print("".join(bar))

        # Scale
        scale = [" "] * width
        scale[0] = "0"
        scale[width // 2] = "250"
        scale[-3:] = ["5", "0", "0"]
        print("".join(scale))

        print()
        print("Legend: 🚁=Helicopter 💰=Bag ⚡=Oddball 🎯=Perfect drop")
        print("        🔄=Position changed ⚠️=Oddball trial")
        print("=" * width)

        if delay > 0:
            time.sleep(delay)

    print("\n✅ Simulation complete!")


def visualize_summary(df: pd.DataFrame):
    """Print summary statistics of the environment."""
    print("\n" + "=" * 60)
    print("📊 SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total trials:        {len(df)}")
    print(
        f"Change points:       {df['is_change_point'].sum()} ({df['is_change_point'].mean() * 100:.1f}%)"
    )
    print(f"Oddball trials:      {df['is_oddball'].sum()} ({df['is_oddball'].mean() * 100:.1f}%)")
    print(f"\nHelicopter position: {df['mu'].min():.1f} - {df['mu'].max():.1f}")
    print(f"Bag drop position:   {df['x'].min():.1f} - {df['x'].max():.1f}")
    print(f"Average distance:    {abs(df['x'] - df['mu']).mean():.1f}")
    print("=" * 60)
