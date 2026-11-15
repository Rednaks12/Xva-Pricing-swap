"""Vanilla fixed-for-floating interest rate swap, single-curve."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Swap:
    notional: float
    fixed_rate: float
    maturity: float      # years
    tau: float = 1.0     # accrual / payment frequency in years (both legs)
    payer: bool = True   # True: we pay fixed, receive floating

    @property
    def pay_dates(self):
        return np.arange(self.tau, self.maturity + 1e-9, self.tau)

    def par_rate(self, model):
        P = model.discount(self.pay_dates)
        return (1.0 - P[-1]) / (self.tau * P.sum())

    def value(self, model, grid, r):
        """Mark-to-market on every path and date (ex-coupon on payment dates).

        `grid` must contain every reset date, i.e. its step must divide `tau`.
        Returns an array of shape (n_paths, len(grid)).
        """
        grid = np.asarray(grid, dtype=float)
        V = np.zeros_like(r)
        for j, t in enumerate(grid):
            remaining = self.pay_dates[self.pay_dates > t + 1e-9]
            if remaining.size == 0:
                continue
            nxt, last = remaining[0], remaining[-1]
            k = int(np.argmin(np.abs(grid - (nxt - self.tau))))  # last reset date
            if abs(grid[k] - (nxt - self.tau)) > 1e-9:
                raise ValueError("grid must contain every reset date")

            # Floating coupon fixed at the last reset, paid at `nxt`.
            libor = (1.0 / model.bond(grid[k], nxt, r[:, k]) - 1.0) / self.tau
            floating = model.bond(t, nxt, r[:, j]) * (1.0 + self.tau * libor) - model.bond(
                t, last, r[:, j]
            )
            fixed = self.fixed_rate * self.tau * sum(model.bond(t, T, r[:, j]) for T in remaining)
            V[:, j] = self.notional * (floating - fixed)
        return V if self.payer else -V
