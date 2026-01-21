from pyhgf.model import Network
import numpy as np

class Weber_model(Network):

    def __init__(self,  input, node4 = True, n4_p = 3.0):
        """A subclass of the pyhfg Network class.
        Each Instance is a set network of either 4 or 5 Nodes (node 4 can be left out for experimental purposes):
            Node 0: Continous input Node
            Node 1: Value Parent of Node 0
            Node 2: Volatility Parent of Node 1
            Node 3: Volatility Parent of Node 0
            Node 4: Volatility Parent of Node 3

            Input:
            - input: a Dataframe containing observations (in a column named 'x'), to which the model is fit.
            - node4: A Boolean indicating whether the model contains node 4
            - n4_p: precision of node 4, if used

        """
        super().__init__()
        self.add_nodes( mean=250, tonic_volatility=5)
        self.add_nodes( mean=250, value_children=0)
        self.add_nodes( volatility_children=1)
        self.add_nodes( volatility_children=0)
        if node4:
            self.add_nodes( precision = n4_p, volatility_children=3 ) # jump from precison 3 to 4 made it give up earlier in random walk environment
        self.input_data(input["x"].to_numpy())
        
  # precision 3 seems to be the sweet spot so far, such that the first 500 trials can be predicted and are shown in graph (more trials still not working)
  # -> dicotomy between two environments with random walk environment giving up earlier with higher preciscion
  # -> works fine if node 4 is removed


# both fitting functions currently the same, but should maybe stay separate for usability
    def fit_to_change_point_oddball_environment(self, df):
        """Fitting the gHGF to a given dataset, produced in a change-point oddball environment"""
       
        input = df["x"].to_numpy()
        self.input_data(input)

    def fit_to_random_walk_oddball_environment(self, df):
        """Fits the gHGF onto data generated in a random walk oddball environment"""
        input = df["x"].to_numpy()
        self.input_data(input)
    
    
    def largest_jump(self):
        """Returns the largest jump in the observations and its index (for the entire observation span or until one of the nodes drops out)"""
        trajectories = self.to_pandas()
        x0_means = trajectories["x_0_mean"]
        number_nan = np.count_nonzero(np.isnan(trajectories.iloc[0]))
        largest_jump = 0
        i = 1
        at = 1
        while i < len(x0_means) and ( number_nan == 0 ):
            diff = abs(x0_means[i-1]-x0_means[i])
            number_nan = np.count_nonzero(np.isnan(trajectories.iloc[i]))
            if diff > largest_jump:
                largest_jump = diff
                at = i
            i = i+1
        return largest_jump, at
    
    def max_total_surprise(self):
        """Returns the biggest total surprise in the observations and its index (for the entire observation span or until one of the nodes drops out)"""
        trajectories = self.to_pandas()
        total_surprises = trajectories["total_surprise"]
        number_nan = np.count_nonzero(np.isnan(trajectories.iloc[0]))
        max_total_surprise = 0
        i = 1
        at = 1
        while i < len(trajectories) and ( number_nan == 0 ):
            number_nan = np.count_nonzero(np.isnan(trajectories.iloc[i]))
            if total_surprises[i] > max_total_surprise:
                max_total_surprise = total_surprises[i]
                at = i
            i = i+1
        return max_total_surprise, at
