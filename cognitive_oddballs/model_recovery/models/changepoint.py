from __future__ import annotations
import numpy as np
from .base import BaseModel
from cognitive_oddballs.models.change_point_model_variational import ChangePointModelVariational

class CPM(BaseModel):
    """
    Thin wrapper around the library's ChangePointModelVariational so that
    the inference code sees a uniform API.
    """
    def __init__(
        self,
        w1_std: float,
        w2_std: float,
        h: float,
        sigma0: float = 25.0,
        obs_noise_std: float = 25.0,
        add_second_level: bool = True,
        mu0: float | None = None,
    ):
        # `mu0` will be set later from the first observation
        self._mu0 = mu0
        self._model = ChangePointModelVariational(
            mu0=0.0,                     # placeholder – overwritten in reset()
            sigma0=sigma0,
            obs_noise_std=obs_noise_std,
            w1_std=w1_std,
            w2_std=w2_std,
            h=h,
            add_second_level=add_second_level,
        )
        self.n_params = 3

    # ------------------------------------------------------------------ #
    # Uniform API implementation
    # ------------------------------------------------------------------ #
    def reset(self, initial_obs: float) -> None:
        self._model.mu0 = self._model.mu = initial_obs if self._mu0 is None else self._mu0
        self._model.sigma = self._model.sigma0
        if getattr(self._model, "add_second_level", False):
            self._model.mu2 = 0.0
            self._model.sigma2 = 1.0
        self._model.n_trials = 0   # will be increased automatically in update()

    def predict(self) -> float:
        return float(self._model.mu)

    def update(self, observation: float) -> None:
        # `self._model.update(t)` expects the *trial index* – we cheat by using `n_trials`
        self._model.x = np.asarray([observation])
        self._model.update(self._model.n_trials)
        self._model.n_trials += 1

    @property
    def n_params(self) -> int:
        return 3