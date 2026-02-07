from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

class Model(ABC):
    """Abstract class to define the mode interface."""

    @abstractmethod
    def run(self, observations: np.ndarray) -> pd.DataFrame:
    def run(self, observations: np.ndarray) -> pd.DataFrame:
        """
        Run the model on each observation in sequence.

        Args:
            observations (np.ndarray): Input observations for the model.

        Returns:
            pd.DataFrame: Relevant model outputs as a pandas DataFrame.
        """
        pass

    @abstractmethod
    def set_parameters_cma(self, theta: np.ndarray) -> None:
        """
        Set model parameters from CMA-ES parameter vector.

        Args:
            theta (np.ndarray): Parameter vector used by CMA-ES.
        """
        pass

    @abstractmethod
    def objective_cma(self, observations: np.ndarray) -> float:
        """
        Compute the scalar objective value for CMA-ES optimization.

        This method should:
          - run the model on `observations` (internally calling `run`)
          - compute and return a scalar loss (NLL, VFE, etc.)

        Args:
            observations (np.ndarray): Input observations.

        Returns:
            float: Scalar objective to minimize.
        """
        pass

    @staticmethod
    @abstractmethod
    def decode_cma_theta(theta: np.ndarray) -> dict:
        """Map theta to a dict of named parameters for reporting."""
        pass