import numpy as np
from pyhgf.model import Network


class WeberModel(Network):
    def __init__(self, input=None, node4=True, node_4_type="volatility_parent", n4_p=3.0):
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
        self.add_nodes(mean=250, tonic_volatility=5, autoconnection_strength=0.5)
        self.add_nodes(mean=250, value_children=0)
        self.add_nodes(volatility_children=1)
        self.add_nodes(volatility_children=0)
        if node4:
            if node_4_type not in ["volatility_parent", "value_parent"]:
                raise ValueError(
                    "node_4_type has to be either 'volatility_parent' or 'value_parent'."
                )
            elif node_4_type == "volatility_parent":
                self.add_nodes(
                    precision=n4_p, volatility_children=3
                )  # jump from precison 3 to 4 made it give up earlier in random walk environment
            else:
                self.add_nodes(precision=n4_p, value_children=3)
        if input is not None:
            self.input_data(input["x"].to_numpy())

    # precision 3 seems to be the sweet spot so far, such that the first 500 trials can be predicted
    # and are shown in graph (more trials still not working)
    # -> dicotomy between two environments with random walk environment giving up earlier with
    # higher preciscion
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
