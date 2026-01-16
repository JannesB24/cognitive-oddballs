import pyhgf
from pyhgf.model import Network
from pyhgf.plots.matplotlib import plot_nodes
from pyhgf.plots.matplotlib import plot_trajectories
import pandas as pd


class Weber_model():
    """An example of a generalized Hierarchichal Gaussian Filter, using a set structure"""
    network = 0
    def __init__(self):
        self.network = (
        Network()
        .add_nodes(node_parameters={"mean":250})
        .add_nodes(value_children=0, node_parameters={"mean":250})
        .add_nodes(volatility_children=1)
        .add_nodes(volatility_children=0)
        .add_nodes(volatility_children=3)
        )
  
#currently gives no output and I'm not sure why <- run in interactive window -> still no graph
    def show_graph(self):
        """Plots the network structure of the model"""
        #plot_nodes(self.network,[0,1,2,3,4])
        #testing why it might not work
        print("test") # call in test file works, but graph is not shown
        print(self.network) # there is an actual network object created within the test file

#plotting in the interactive window -> now unsure why plot_network isn't working
# model just seems to give up after the third input 
    def plot_trajectories(self):
        """plotting the trajectories of the different network nodes"""
        pyhgf.plots.matplotlib.plot_trajectories(self.network)    

# both fitting functions currently the same, but should maybe stay separate for usability
    def fit_to_change_point_oddball_environment(self, df):
        """Fitting the gHGF to a given dataset, produced in a change-point oddball environment"""
       #seems to be working so far, at least no error message, but not clear as long as the graph doesn't work
        input = df["x"].to_numpy()
        self.network.input_data(input)

    def fit_to_random_walk_oddball_environment(self, df):
        """Fits the gHGF onto data generated in a random walk oddball environment"""
        input = df["x"].to_numpy()

        self.network.input_data(input)

    