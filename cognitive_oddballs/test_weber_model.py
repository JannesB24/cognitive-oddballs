
from models.weber_model import Weber_model
from environments.change_point_oddball_environment import generate_oddball_environment
from environments.random_walk_oddball_environment import generate_random_walk_environment

oddball_data = generate_oddball_environment(n_trials=500, oddball_hazard_rate=0.15, sigma=20, change_point_hazard_rate=0.1, seed=42)
#print(oddball_data.head())
random_walk_data = generate_random_walk_environment(n_trials=500, oddball_hazard_rate=0.15, sigma=20, seed=42)
oddball_model = Weber_model()
random_walk_model = Weber_model()

oddball_model.fit_to_change_point_oddball_environment(oddball_data)
#oddball_model.show_graph()
oddball_model.plot_trajectories()

#oddball_model.print_predictions()
#oddball_model.network.plot_nodes(0)
random_walk_model.fit_to_random_walk_oddball_environment(random_walk_data)
random_walk_model.plot_trajectories()