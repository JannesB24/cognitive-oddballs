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
        """
        pass
