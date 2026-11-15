"""One-factor Hull-White short-rate model fitted to a flat initial curve.

    dr(t) = (theta(t) - a r(t)) dt + sigma dW(t)

The short rate is written r(t) = x(t) + alpha(t), where x is a zero-mean
Ornstein-Uhlenbeck process simulated exactly (no Euler bias) and alpha(t)
is the deterministic shift that reproduces the initial curve.
Zero-coupon bond prices are analytic, so the swap can be revalued on every
path and date without nested Monte Carlo.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HullWhite:
    a: float      # mean-reversion speed
    sigma: float  # short-rate volatility
    r0: float     # flat continuously-compounded initial curve

    def discount(self, T):
        """Initial discount factor P(0, T)."""
        return np.exp(-self.r0 * np.asarray(T, dtype=float))

    def B(self, t, T):
        return (1.0 - np.exp(-self.a * (T - t))) / self.a

    def alpha(self, t):
        """Deterministic shift: r(t) = x(t) + alpha(t)."""
        t = np.asarray(t, dtype=float)
        return self.r0 + self.sigma**2 / (2 * self.a**2) * (1 - np.exp(-self.a * t)) ** 2

    def bond(self, t, T, r):
        """Zero-coupon price P(t, T) given the short rate r at time t."""
        B = self.B(t, T)
        ln_A = (
            -self.r0 * (T - t)
            + B * self.r0
            - self.sigma**2 / (4 * self.a) * (1 - np.exp(-2 * self.a * t)) * B**2
        )
        return np.exp(ln_A - B * r)

    def simulate(self, grid, n_paths, seed=0):
        """Simulate the short rate on `grid` (antithetic variates).

        Returns
        -------
        r : (n_paths, len(grid)) short-rate paths
        D : (n_paths, len(grid)) stochastic discount factors exp(-int_0^t r ds)
        """
        grid = np.asarray(grid, dtype=float)
        if n_paths % 2:
            raise ValueError("n_paths must be even (antithetic variates)")
        rng = np.random.default_rng(seed)
        half = rng.standard_normal((n_paths // 2, len(grid) - 1))
        z = np.vstack([half, -half])

        x = np.zeros((n_paths, len(grid)))
        for i, dt in enumerate(np.diff(grid)):
            decay = np.exp(-self.a * dt)
            std = self.sigma * np.sqrt((1 - decay**2) / (2 * self.a))
            x[:, i + 1] = x[:, i] * decay + std * z[:, i]
        r = x + self.alpha(grid)

        # Trapezoidal integration of the short rate along each path.
        integral = np.zeros_like(r)
        integral[:, 1:] = np.cumsum(0.5 * (r[:, 1:] + r[:, :-1]) * np.diff(grid), axis=1)
        return r, np.exp(-integral)
