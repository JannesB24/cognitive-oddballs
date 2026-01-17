import pyhgf
from pyhgf.model import Network

class Weber_model(Network):
    """An example of a generalized Hierarchichal Gaussian Filter, using a set structure"""
    network = 0
    def __init__(self):
        self.network = (
        Network()
        .add_nodes(node_parameters={"mean":250})
        .add_nodes(value_children=0, node_parameters={"mean":250})
        .add_nodes(volatility_children=1)
        .add_nodes(volatility_children=0)
        .add_nodes(volatility_children=3, node_parameters={"precision":3}) # jump from precison 3 to 4 made it give up earlier in random walk environment
        )
  # precision 3 seems to be the sweet spot so far, such that the first 500 trials can be predicted and are shown in graph (more trials still not working)
  # -> dicotomy between two environments with random walk environment giving up earlier with higher preciscion


    def plot_network(self):
        """Plots the network structure of the model"""
        self.network.plot_network()
        #currently gives no output and I'm not sure why <- run in interactive window -> still no graph
        


    def plot_trajectories(self):
        """plotting the trajectories of the different network nodes"""
        self.network.plot_trajectories() 
        #plotting in the interactive window 

# both fitting functions currently the same, but should maybe stay separate for usability
    def fit_to_change_point_oddball_environment(self, df):
        """Fitting the gHGF to a given dataset, produced in a change-point oddball environment"""
       
        input = df["x"].to_numpy()
        self.network.input_data(input)

    def fit_to_random_walk_oddball_environment(self, df):
        """Fits the gHGF onto data generated in a random walk oddball environment"""
        input = df["x"].to_numpy()

        self.network.input_data(input)
