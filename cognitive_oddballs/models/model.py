from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Model(ABC):
    """Abstract class to define the mode interface."""

    @abstractmethod
    def run(self, observations: np.ndarray) -> pd.DataFrame:
        """
        Run the model on each observation in sequence.

        Args:
            observations (np.ndarray): Input observations for the model.

        Returns:
            pd.DataFrame: Relevant model outputs as a pandas DataFrame.

            The first line corresponds to the initial belief before seeing any observation,
            therefore the entry at index 1 in "beliefs" corresponds to the belief after seeing
            the first observation.
        """
        pass
