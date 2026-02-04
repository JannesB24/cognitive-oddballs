from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Minimal protocol required by the inference utilities."""
    @abstractmethod
    def reset(self, initial_obs: float) -> None: ...
    @abstractmethod
    def predict(self) -> float: ...
    @abstractmethod
    def update(self, observation: float) -> None: ...
    @property
    @abstractmethod
    def n_params(self) -> int: ...