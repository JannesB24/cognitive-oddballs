from __future__ import annotations
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

def confusion_matrix(winners: Dict[str, Counter]) -> None:
    """Print a normalized confusion matrix (rows = true, columns = recovered)."""
    model_names = list(winners.keys())
    header = " " * 12 + " ".join(f"{m:>8}" for m in model_names)
    print("\nCONFUSION MATRIX (rows = true, cols = recovered)")
    print(header)

    for true_m in model_names:
        row = winners[true_m]
        total = sum(row.values())
        probs = [row[m] / total if total else 0 for m in model_names]
        print(f"{true_m:>10} | " + " ".join(f"{p:8.2f}" for p in probs))


def param_recovery_stats(
    param_pairs: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]
) -> None:
    """Correlation of true ↔ recovered params for the correctly identified fits."""
    print("\nPARAMETER RECOVERY (only correctly identified fits)")
    for model, pairs in param_pairs.items():
        if len(pairs) == 0:
            print(f"{model}: No correctly recovered simulations")
            continue

        true_arr = np.vstack([p[0] for p in pairs])
        rec_arr  = np.vstack([p[1] for p in pairs])

        print(f"\n{model}:")
        for i in range(true_arr.shape[1]):
            r = np.corrcoef(true_arr[:, i], rec_arr[:, i])[0, 1]
            print(f"  param {i+1:>2}: corr(true, rec) = {r: .2f}")