from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Iterable, Mapping, Union

ObsType = Union[np.ndarray, pd.DataFrame, Mapping[str, Any], Iterable[Any]]

def get_observations(env_out: ObsType) -> np.ndarray:
    """
    Convert any supported environment output to a 1‑D float array.
    """
    # DataFrame → column “x” or “o”
    if isinstance(env_out, pd.DataFrame):
        for col in ("x", "o"):
            if col in env_out.columns:
                return env_out[col].to_numpy(dtype=float).ravel()
        # fall back to the first column
        return env_out.iloc[:, 0].to_numpy(dtype=float).ravel()

    # Mapping (dict‑like)
    if isinstance(env_out, Mapping):
        for key in ("x", "o", "observations"):
            if key in env_out:
                return np.asarray(env_out[key], dtype=float).ravel()
        raise KeyError(f"Missing observation key in dict {list(env_out)}")

    # Iterable / tuple / list → first element is the array
    if isinstance(env_out, (list, tuple)):
        return np.asarray(env_out[0], dtype=float).ravel()

    # Anything else → numpy conversion
    arr = np.asarray(env_out, dtype=float)
    if arr.ndim != 1:
        arr = arr[:, 0]          # take first column if 2‑D
    return arr.ravel()