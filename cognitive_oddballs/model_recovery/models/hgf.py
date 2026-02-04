from __future__ import annotations
import numpy as np
from .base import BaseModel
from cognitive_oddballs.models.hgf.hgf2_gaussian import HGFPaper2Gaussian

class HGF(BaseModel):
    def __init__(self, eta: float, s2: float):
        self._model = HGFPaper2Gaussian(eta, s2)
        self.n_params = 2

    def reset(self, initial_obs: float) -> None:
        # HGF objects are stateless after construction, so we just re‑instantiate.
        self._model = HGFPaper2Gaussian(*self._model.params)

    def predict(self) -> float:
        return float(self._model.mu1)

    def update(self, observation: float) -> None:
        self._model.update(observation)

    @property
    def n_params(self) -> int:
        return 2