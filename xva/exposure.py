"""Exposure profiles and a simple variation-margin model."""
import numpy as np


def collateralise(V, grid, threshold=0.0, mpor=0.0):
    """Net exposure after variation margin.

    Collateral held at t is based on the mark-to-market observed `mpor`
    years earlier (margin period of risk), above a symmetric `threshold`:

        C(t) = max(V(t - mpor) - H, 0) - max(-V(t - mpor) - H, 0)

    Returns V(t) - C(t). With mpor = 0 and H = 0 the exposure is zero.
    """
    grid = np.asarray(grid, dtype=float)
    lag = int(round(mpor / (grid[1] - grid[0])))
    V_lag = np.zeros_like(V)
    if lag == 0:
        V_lag = V.copy()
    else:
        V_lag[:, lag:] = V[:, :-lag]
    C = np.maximum(V_lag - threshold, 0.0) - np.maximum(-V_lag - threshold, 0.0)
    return V - C


def profiles(V, D, quantile=0.975):
    """Exposure metrics from mark-to-market paths V and discount factors D.

    EE / ENE  : expected positive / negative exposure (undiscounted)
    PFE       : `quantile` of the positive exposure
    dEE / dENE: discounted expected exposures, the inputs of CVA / DVA
    """
    pos, neg = np.maximum(V, 0.0), np.maximum(-V, 0.0)
    return {
        "EE": pos.mean(axis=0),
        "ENE": neg.mean(axis=0),
        "PFE": np.quantile(pos, quantile, axis=0),
        "dEE": (D * pos).mean(axis=0),
        "dENE": (D * neg).mean(axis=0),
    }


def epe(ee, grid):
    """Time-averaged expected exposure."""
    grid = np.asarray(grid, dtype=float)
    return np.sum(0.5 * (ee[1:] + ee[:-1]) * np.diff(grid)) / (grid[-1] - grid[0])
