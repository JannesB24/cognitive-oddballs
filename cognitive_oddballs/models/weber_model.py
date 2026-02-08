import numpy as np
import pandas as pd
from pyhgf.model import Network

from cognitive_oddballs.models.model import Model


class WeberModel(Network, Model):
    def __init__(
        self, n_nodes=4, x_0_tv=5, x_2_mu=15, x_3_mu=40, x_4_p=1e1, update_type="standard"
    ):  # paramteres to maybe add: node precisions, tonic volatilities, initial means for node 3 and 4 (if 4 is kept)
        """A subclass of the pyhfg Network class.
        Each Instance is a set network of 3 to 5 Nodes (4th and 5th node can be left out for
        experimental purposes):

            Node 0: Continous input Node | represents the expected bag drop location
            Node 1: Value Parent of Node 0 | represents the expected location of the helicopter
            Node 2: Volatility Parent of Node 1 | represents the volatility of helicopter movement (spikes in trajectory noting changepoints)
            Node 3: Volatility Parent of Node 0  | represents the volatility of the bag drop locations (spikes in trajectory noting oddballs?)
            Node 4: Volatility Parent of Node 3

            Input:
            - n_nodes: An int denominating the number of nodes ( 3 to 5 nodes (default 4))
            - x_0_tv: tonic volatility of node 0 (default 5)
            - x_2_mu: initial mean of node 2 (default 15)
            - x_3_mu: initial mean of node 3 (default 40)
            - x_4_p: precision of node 4 (default 1e1; only needed if n_nodes == 5)
            - ....
            - update_type: the update type forwarded to the Network class (default currently "eHGF")


        """
        # if n_nodes is not in specified range, a value error is raised
        if n_nodes < 3 or n_nodes > 5:
            raise ValueError("n_nodes must be between 3 and 5 (inclusive)")

        # passing the update type to the Init function of the Network class
        super().__init__(update_type=update_type)
        # Node 0: Observation node/ Continuous input node
        self.add_nodes(
            mean=250, tonic_volatility=x_0_tv, autoconnection_strength=0
        )  # initial mean set at 250, as that is always the middle of the possible environmental values
        # Node 1: Value parent of Node 0
        self.add_nodes(
            mean=250, value_children=0
        )  # initial mean set at 250, as that is always the middle of the possible environmental values
        # Node 2: Volatility parent of node 1
        self.add_nodes(mean=x_2_mu, volatility_children=1)
        # if given number of nodes is at least 4, more nodes are added
        if n_nodes >= 4:
            self.add_nodes(mean=x_3_mu, volatility_children=0)
            # if number of nodes is 5 one more node is added
            if n_nodes == 5:
                self.add_nodes(volatility_children=3, precision=x_4_p)

    def run(self, observations: np.ndarray) -> pd.DataFrame:
        self.input_data(observations)

        output = self.to_pandas()[["x_0_expected_mean"]]

        rename_dict = {"x_0_expected_mean": "raw_responses"}

        return output.rename(columns=rename_dict)

    def to_pandas(self):
        """Returns the trajectories of the nodes. Extended with the prediction error of node 0"""
        output = super().to_pandas()
        output["x_0_prediction_error"] = output["x_0_mean"] - output["x_0_expected_mean"]
        return output

    # functions used in testing
    def drops_out(self) -> bool:
        """Checks whether the Model drops out at any point and returns the corresponding bool"""
        trajectories = self.to_pandas()

        return np.count_nonzero(np.isnan(trajectories)) > 0

    def value_prediction_errors(self):
        """Returns the prediction error of Node 0 for each observation"""
        return self.to_pandas()["x_0_prediction_error"]

    def largest_jump(self):
        """Returns the largest jump in the observations and its index
        (for the entire observation span or until one of the nodes drops out)"""
        trajectories = self.to_pandas()
        x0_means = trajectories["x_0_mean"]
        number_nan = np.count_nonzero(np.isnan(trajectories.iloc[0]))
        largest_jump = 0
        i = 1
        at = 1
        while i < len(x0_means) and (number_nan == 0):
            diff = abs(x0_means[i - 1] - x0_means[i])
            number_nan = np.count_nonzero(np.isnan(trajectories.iloc[i]))
            if diff > largest_jump:
                largest_jump = diff
                at = i
            i = i + 1
        return largest_jump, at

    def max_total_surprise(self):
        """Returns the biggest total surprise in the observations and its index
        (for the entire observation span or until one of the nodes drops out)"""
        trajectories = self.to_pandas()
        total_surprises = trajectories["total_surprise"]
        number_nan = np.count_nonzero(np.isnan(trajectories.iloc[0]))
        max_total_surprise = 0
        i = 1
        at = 1
        while i < len(trajectories) and (number_nan == 0):
            number_nan = np.count_nonzero(np.isnan(trajectories.iloc[i]))
            if total_surprises[i] > max_total_surprise:
                max_total_surprise = total_surprises[i]
                at = i
            i = i + 1
        return max_total_surprise, at
