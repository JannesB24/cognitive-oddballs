from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """Abstract class to define the mode interface."""

    @abstractmethod
    def run(self, observations: np.ndarray):
        """
        Run the model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Model output.
        """
        pass
