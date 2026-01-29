import numpy as np
import pandas as pd
from pyhgf.model import Network
from pyhgf.response import total_gaussian_surprise

from cognitive_oddballs.models.model import Model
import logging


logging.getLogger("jax").setLevel(logging.INFO)
logging.getLogger("jax._src").setLevel(logging.INFO)


class WeberModel(Network, Model):
    def __init__(self, node4=True, node_4_type="volatility_parent", n4_p=3.0):
        """A subclass of the pyhfg Network class.
        Each Instance is a set network of either 4 or 5 Nodes (node 4 can be left out for
        experimental purposes):
            Node 0: Continous input Node
            Node 1: Value Parent of Node 0
            Node 2: Volatility Parent of Node 1
            Node 3: Volatility Parent of Node 0
            Node 4: Volatility Parent of Node 3

            Input:
            - input: (optional) a Dataframe containing observations (in a column named 'x'),
            to which the model is fit immediately.
            - node4: A Boolean indicating whether the model contains node 4 (default true)
            - node_4_type: a string indicating whether node 4 is supposed to be a value parent or
              a volatility parent (default volatility parent)
            - n4_p: precision of node 4, if used (default 3, as that was the sweetspot for model
              performance in both environments)

        """
        super().__init__()

        # Node 0: continuous input node
        self.add_nodes(mean=250, tonic_volatility=5, autoconnection_strength=0.5)
        self.idx_0 = self.n_nodes - 1

        # Node 1: value parent of node 0
        self.add_nodes(mean=250, value_children=0)
        self.idx_1 = self.n_nodes - 1

        # Node 2: volatility parent of node 1
        self.add_nodes(volatility_children=1)
        self.idx_2 = self.n_nodes - 1

        # Node 3: volatility parent of node 0
        self.add_nodes(volatility_children=0)
        self.idx_3 = self.n_nodes - 1

        self._has_node4 = bool(node4)
        self._node4_type = node_4_type
        self.idx_4 = None

        if node4:
            if node_4_type not in ["volatility_parent", "value_parent"]:
                raise ValueError(
                    "node_4_type has to be either 'volatility_parent' or 'value_parent'."
                )
            elif node_4_type == "volatility_parent":
                self.add_nodes(precision=n4_p, volatility_children=3)
            else:
                self.add_nodes(precision=n4_p, value_children=3)
            self.idx_4 = self.n_nodes - 1

        self._initial_n4_p = float(n4_p)

        # if node4:
        #     if node_4_type not in ["volatility_parent", "value_parent"]:
        #         raise ValueError(
        #             "node_4_type has to be either 'volatility_parent' or 'value_parent'."
        #         )
        #     elif node_4_type == "volatility_parent":
        #         self.add_nodes(
        #             precision=n4_p, volatility_children=3
        #         )  # jump from precison 3 to 4 made it give up earlier in random walk environment
        #         self._has_node4 = True
        #     else:
        #         self.add_nodes(precision=n4_p, value_children=3)
        #         self._has_node4 = True

    # precision 3 seems to be the sweet spot so far, such that the first 500 trials can be predicted
    # and are shown in graph (more trials still not working)
    # -> dicotomy between two environments with random walk environment giving up earlier with
    # higher preciscion
    # -> works fine if node 4 is removed

    # ---------- Model interface implementation ----------
    def run(self, observations: np.ndarray) -> pd.DataFrame:
        self.input_data(observations)

        output = self.to_pandas()[["x_0_expected_mean"]]

        rename_dict = {"x_0_expected_mean": "raw_responses"}

        return output.rename(columns=rename_dict)
    
    # TODO: IMPORTANT!!! this is unfinished -- I'm pretty sure I remember there being a specific way to re-assign node parameters
    # and i did not use that here yet
    # i need to look up how to do that properly
    # ALSO ALSO ALSO this does not include all parameters yet
    # def set_parameters_cma(self, theta: np.ndarray) -> None:
    #     """
    #     theta layout (example):
    #         theta[0] = log_n4_p           (precision of node 4, if present)
    #         theta[1] = log_tonic_vol_3    (tonic volatility of node 3)
    #         theta[2] = log_tonic_vol_1    (tonic volatility of node 1)
    #     """
    #     log_n4_p, log_tv_3, log_tv_1 = map(float, theta)

    #     n4_p = float(np.exp(log_n4_p))
    #     tv_3 = float(np.exp(log_tv_3))
    #     tv_1 = float(np.exp(log_tv_1))

    #     attrs = self.attributes

    #     # 1) node 4 precision
    #     if self._has_node4 and self.idx_4 is not None:
    #         if "precision" in attrs[self.idx_4]:
    #             attrs[self.idx_4]["precision"] = n4_p
    #         else:
    #             # optional: warn here if needed
    #             pass

    #     # 2) tonic volatility of node 3
    #     if "tonic_volatility" in attrs[self.idx_3]:
    #         attrs[self.idx_3]["tonic_volatility"] = tv_3

    #     # 3) tonic volatility of node 1
    #     if "tonic_volatility" in attrs[self.idx_1]:
    #         attrs[self.idx_1]["tonic_volatility"] = tv_1

    #     # Should I be doing a "reset" to clear history after parameter change? 
    #     # Reset network state so a new input_data() run uses the new parameters
    #     self.node_trajectories = {}
    #     self.predictions = {}
    #     self.last_attributes = None

    def set_parameters_cma(self, theta: np.ndarray) -> None:
        """
        theta[0] = log_n4_p        (precision of node 4, if present)
        theta[1] = log_tv_3        (tonic volatility of node 3)
        theta[2] = log_tv_1        (tonic volatility of node 1)
        """
        log_n4_p, log_tv_3, log_tv_1 = map(float, theta)

        n4_p = float(np.exp(log_n4_p))
        tv_3 = float(np.exp(log_tv_3))
        tv_1 = float(np.exp(log_tv_1))

        attrs = self.attributes

        # 1) node 4 precision
        if self._has_node4 and self.idx_4 is not None:
            if "precision" in attrs[self.idx_4]:
                attrs[self.idx_4]["precision"] = n4_p

        # 2) tonic volatility of node 3
        if "tonic_volatility" in attrs[self.idx_3]:
            attrs[self.idx_3]["tonic_volatility"] = tv_3

        # 3) tonic volatility of node 1
        if "tonic_volatility" in attrs[self.idx_1]:
            attrs[self.idx_1]["tonic_volatility"] = tv_1

        # reset dynamic state so new parameters are used fresh
        self.node_trajectories = {}
        self.predictions = {}
        self.last_attributes = None

    # def set_parameters_cma(self, theta: np.ndarray) -> None:
    #     """
    #     Set model parameters from CMA-ES parameter vector.

    #     theta[0] = log_sigma_obs
    #     theta[1] = log_sigma_mu
    #     """

    #     log_n4_p, log_tv_3, log_tv_1 = map(float, theta)

    #     # Maybe this way instead?
    #     #self.nodes[0].obs_noise = float(np.exp(theta[0]))
    #     #self.nodes[1].tonic_volatility = float(np.exp(theta[1]))

    #     n4_p = float(np.exp(log_n4_p))
    #     tv_3 = float(np.exp(log_tv_3))
    #     tv_1 = float(np.exp(log_tv_1))

    #     # IMPORTANT: still need to check how pyHGF exposes nodes.
    #     # Often sth like self.nodes[0], self.nodes[1], ...
    #     # Here assuming: node0=input, node1=value parent, node2=vol parent of 1,
    #     # node3=vol parent of 0, node4=vol parent of 3 (if present).

    #     # 1) set node4 precision (if present)
    #     if self._has_node4:
    #         node4 = self.nodes[-1]
    #         if hasattr(node4, "precision"):
    #             node4.precision = n4_p

    #     # 2) set tonic volatility of node3
    #     node3 = self.nodes[3]   # adjust index to your Network API
    #     if hasattr(node3, "tonic_volatility"):
    #         node3.tonic_volatility = tv_3

    #     # 3) set tonic volatility of node1 (volatility parent of node0)
    #     node1 = self.nodes[1]   # again, adjust index if needed
    #     if hasattr(node1, "tonic_volatility"):
    #         node1.tonic_volatility = tv_1

        

    # TODO: LLM-generated -- verify correctness
    # Again, does not include all params
    def objective_cma(self, observations: np.ndarray) -> float:
        """
        CMA-ES objective: model 'surprise' (negative log probability) for the
        given sequence of observations, using pyHGF's Gaussian surprise.

        CMA-ES minimizes this directly.
        """
        # Feed new data; this should trigger a fresh run under current parameters
        self.input_data(observations)

        # vector of per-trial surprises (JAX array)
        surprise_vec = self.surprise(
            response_function=total_gaussian_surprise, # TODO: is that the right one to use? I feel like I had this problem when I implemented stuff for the presentation
            response_function_inputs=(),          # no extra inputs
            response_function_parameters=None,    # no extra params -- possibly 1.0?
        )
        print(f"Per trial surprise in objective_cma: {surprise_vec}")

        # convert to NumPy and sum
        surprise_arr = np.asarray(surprise_vec, dtype=float)
        if not np.all(np.isfinite(surprise_arr)):
            # Bad param combo, let CMA wrapper handle this
            raise FloatingPointError(
                f"Non-finite surprise values in WeberModel: {surprise_arr}"
            )

        print(f"Objective (total surprise) in objective_cma: {total_surprise}")
        total_surprise = surprise_arr.sum()
        return float(total_surprise)

            
    # TODO: LLM-generated -- verify correctness
    @staticmethod
    def decode_cma_theta(theta: np.ndarray) -> dict:
        """
        Map CMA parameter vector back to named, interpretable parameters.

        Current parameterization:
            ...
        """
        log_n4_p, log_tv_3, log_tv_1 = map(float, theta)
        return {
            "n4_precision": float(np.exp(log_n4_p)),
            "tonic_vol_3": float(np.exp(log_tv_3)),
            "tonic_vol_1": float(np.exp(log_tv_1)),
            "log_n4_precision": log_n4_p,
            "log_tonic_vol_3": log_tv_3,
            "log_tonic_vol_1": log_tv_1,
        }

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

    def value_prediction_errors(self):
        """Returns the prediction error of Node 0 for each observation"""
        return self.to_pandas()["x_0_prediction_error"]

    def to_pandas(self):
        output = super().to_pandas()
        output["x_0_prediction_error"] = output["x_0_mean"] - output["x_0_expected_mean"]
        return output

    def get_outputs(self):
        trajectories = self.to_pandas()
        outputs = {"prediction_errors": trajectories["x_0_prediction_error"], "updates": []}
        # not sure for the last one, which index to use. I don't know if a new prediction is
        # made before a new input would be given, so for now it simply appends the last known
        # prediction even tho it pertains to the last observation
        for i in range(len(trajectories) - 1):
            outputs["updates"].append(trajectories.loc[i + 1, "x_0_expected_mean"])
        outputs["updates"].append(trajectories.loc[len(trajectories), "x_0_expected_mean"])
        return outputs
