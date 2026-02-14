import numpy as np
import pandas as pd
import logging
from pyhgf.model import Network

from cognitive_oddballs.models.model import Model

logging.getLogger("jax").setLevel(logging.INFO)
logging.getLogger("jax._src").setLevel(logging.INFO)

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
        self.initial_means = [250, 250, x_2_mu, x_3_mu]

        # if n_nodes is not in specified range, a value error is raised
        if n_nodes < 3 or n_nodes > 5:
            raise ValueError("n_nodes must be between 3 and 5 (inclusive)")

        # passing the update type to the Init function of the Network class
        super().__init__(update_type=update_type)
        # Node 0: Observation node/ Continuous input node
        self.add_nodes(
            mean=self.initial_means[0], tonic_volatility=x_0_tv, autoconnection_strength=0
        )  # initial mean set at 250, as that is always the middle of the possible environmental values
        self.idx_0 = self.n_nodes - 1
        # Node 1: Value parent of Node 0
        self.add_nodes(
            mean=self.initial_means[1], value_children=0
        )  # initial mean set at 250, as that is always the middle of the possible environmental values
        self.idx_1 = self.n_nodes - 1
        # Node 2: Volatility parent of node 1
        self.add_nodes(mean=self.initial_means[2], volatility_children=1)
        self.idx_2 = self.n_nodes - 1
        # if given number of nodes is at least 4, more nodes are added
        if n_nodes >= 4:
            self.add_nodes(mean=self.initial_means[3], volatility_children=0)
            self.idx_3 = self.n_nodes - 1
            # if number of nodes is 5 one more node is added
            if n_nodes == 5:
                self.add_nodes(volatility_children=3, precision=x_4_p)
                self.idx_4 = self.n_nodes - 1

    # -------------- Model interface implentation -----------------    
    def run(self, observations: np.ndarray) -> pd.DataFrame:
        self.input_data(observations)

        cols = [
            "x_0_expected_mean",
            "x_0_prediction_error",
            # "x_0_surprise",
            "total_surprise",
        ]

        output = self.to_pandas()[cols]

        initial_row = pd.DataFrame(
            {
                "x_0_expected_mean": self.initial_means[0],
                "x_0_prediction_error": 0.0,
                # "x_0_surprise": 0.0,
                "total_surprise": 0.0,
            },
            index=[0],
        )

        output = pd.concat([initial_row, output], ignore_index=True)

        # output["total_free_energy"] = -output["x_0_surprise"]
        output["variational_free_energy"] = -output["total_surprise"]

        rename_dict = {
            "x_0_expected_mean": "beliefs",
            "x_0_prediction_error": "prediction_error",
            #"total_free_energy": "variational_free_energy",
        }

        return output.rename(columns=rename_dict)
    
    def set_parameters_cma(self, theta):
        """
        CMA-ES parameterization:
            theta[0] = log_x0_tv (tonic volatility of node 0)
            theta[1] = x2_mu (initial mean of node 2)
            theta[2] = x3_mu (initial mean of node 3)
        """
        log_x0_tv, x2_mu, x3_mu = map(float, theta)

        x0_tv = float(np.exp(log_x0_tv))
        x2_mu = float(x2_mu)
        x3_mu = float(x3_mu)

        attrs = self.attributes

        if "tonic_volatility" in attrs[self.idx_0]:
            attrs[self.idx_0]["tonic_volatility"] = x0_tv
        if "mean" in attrs[self.idx_2]:
            attrs[self.idx_2]["mean"] = x2_mu
        if self.idx_3 is not None and "mean" in attrs[self.idx_3]:
            attrs[self.idx_3]["mean"] = x3_mu

        self.initial_means[2] = x2_mu
        if self.idx_3 is not None:
            self.initial_means[3] = x3_mu
        
        # reset dynamic state
        self.node_trajectories = {}
        self.predictions = {}
        self.last_attributes = None

    
    def objective_cma(self, observations):
        """
        CMA-ES objective: sum of per-trial 'total_surprise'

        pyHGF's 'total_surprise' is the negative log probability under the model.
        minimizing its sum is equivalent to maximizing the log-likelihood / free energy of the observed data.
        """
        # run model and get trajectories
        self.input_data(observations)
        trajectories = self.to_pandas()

        if "total_surprise" not in trajectories.columns:
            raise ValueError("total_surprise column not found in trajectories. Check if model is run correctly.")
        s = trajectories["total_surprise"].to_numpy(dtype=np.float64)

        if not np.all(np.isfinite(s)):
            raise ValueError("total_surprise contains non-finite values. Check if model is run correctly and if parameters are in a valid range.")
        
        return float(np.sum(s))
    
    @staticmethod
    def decode_cma_theta(theta: np.ndarray) -> dict:
        """
        Decodes the CMA-ES parameter vector into a human-readable dictionary.
        """
        log_x0_tv, x2_mu, x3_mu = map(float, theta)

        return {
            "x0_tonic_vol": float(np.exp(log_x0_tv)),
            "x2_mu": float(x2_mu),
            "x3_mu": float(x3_mu),
            "log_x0_tonic_vol": log_x0_tv,
        }


    # -------------- End of Model interface implentation -----------------
    def to_pandas(self):
        """Returns the trajectories of the nodes. Extended with the prediction error of node 0"""
        output = super().to_pandas()
        output["x_0_prediction_error"] = output["x_0_mean"] - output["x_0_expected_mean"]
        return output

    # --------------- Functions used in testing ----------------
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