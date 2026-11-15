"""CVA / DVA of a 10-year payer swap, with and without variation margin.

Run from the repository root:  python examples/run_cva.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xva import HullWhite, Swap, collateralise, cs01, cva, epe, profiles

# --- Inputs -----------------------------------------------------------------
model = HullWhite(a=0.05, sigma=0.01, r0=0.03)
grid = np.linspace(0.0, 10.0, 121)            # monthly time steps
notional = 100e6
cpty_spread, own_spread, recovery = 0.015, 0.008, 0.40
mpor = 1 / 12                                  # one-month margin period of risk

# --- Simulation -------------------------------------------------------------
r, D = model.simulate(grid, n_paths=100_000, seed=7)
par = Swap(notional, 0.0, 10.0).par_rate(model)
swap = Swap(notional, par, 10.0, payer=True)
V = swap.value(model, grid, r)

cases = {
    "Uncollateralised": V,
    "Variation margin (MPOR 1m)": collateralise(V, grid, threshold=0.0, mpor=mpor),
}

print(f"Par rate: {par:.4%}\n")
print(f"{'Case':<28}{'EPE':>12}{'Peak PFE':>14}{'CVA':>12}{'DVA':>12}{'CS01':>10}")
results = {}
for name, values in cases.items():
    p = profiles(values, D)
    results[name] = p
    print(
        f"{name:<28}"
        f"{epe(p['EE'], grid):>12,.0f}"
        f"{p['PFE'].max():>14,.0f}"
        f"{cva(p['dEE'], grid, cpty_spread, recovery):>12,.0f}"
        f"{cva(p['dENE'], grid, own_spread, recovery):>12,.0f}"
        f"{cs01(p['dEE'], grid, cpty_spread, recovery):>10,.0f}"
    )

# --- Figure -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
naked, margined = results.values()
ax.plot(grid, naked["PFE"] / 1e6, label="PFE 97.5% (uncollateralised)", color="#b23b3b")
ax.plot(grid, naked["EE"] / 1e6, label="EE (uncollateralised)", color="#1f4e79")
ax.plot(grid, margined["EE"] / 1e6, label="EE (variation margin, MPOR 1m)", color="#1f4e79", ls="--")
ax.plot(grid, -naked["ENE"] / 1e6, label="ENE (uncollateralised)", color="#7f7f7f")
ax.axhline(0, color="black", lw=0.6)
ax.set(xlabel="Time (years)", ylabel="Exposure (EUR m)",
       title="10y payer swap, EUR 100m notional: exposure profile")
ax.legend(frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs" / "exposure_profile.png"
fig.savefig(out, dpi=150)
print(f"\nFigure saved to {out}")
