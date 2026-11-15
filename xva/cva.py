"""Unilateral CVA / DVA with a flat hazard rate, and credit sensitivity."""
import numpy as np


def survival(grid, spread, recovery):
    """Survival curve from a flat CDS spread (credit triangle: lambda = s / (1 - R))."""
    hazard = spread / (1.0 - recovery)
    return np.exp(-hazard * np.asarray(grid, dtype=float))


def cva(discounted_ee, grid, spread, recovery=0.4):
    """CVA = (1 - R) * sum_i dEE(t_i) * [S(t_{i-1}) - S(t_i)].

    Exposure and default are assumed independent (no wrong-way risk).
    The same function gives DVA when fed the discounted ENE and our own spread.
    """
    S = survival(grid, spread, recovery)
    default_prob = S[:-1] - S[1:]
    mid_exposure = 0.5 * (discounted_ee[1:] + discounted_ee[:-1])
    return (1.0 - recovery) * np.sum(mid_exposure * default_prob)


def cs01(discounted_ee, grid, spread, recovery=0.4, bump=1e-4):
    """CVA change for a +1bp parallel shift of the counterparty spread."""
    return cva(discounted_ee, grid, spread + bump, recovery) - cva(
        discounted_ee, grid, spread, recovery
    )
