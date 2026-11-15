import numpy as np
import pytest

from xva import HullWhite, Swap, collateralise, cs01, cva, profiles

MODEL = HullWhite(a=0.05, sigma=0.01, r0=0.03)
GRID = np.linspace(0.0, 10.0, 121)  # monthly
N_PATHS = 40_000


@pytest.fixture(scope="module")
def paths():
    return MODEL.simulate(GRID, N_PATHS, seed=42)


@pytest.fixture(scope="module")
def par_swap():
    swap = Swap(notional=100e6, fixed_rate=0.0, maturity=10.0)
    return Swap(notional=100e6, fixed_rate=swap.par_rate(MODEL), maturity=10.0)


def test_discount_factors_reprice_initial_curve(paths):
    _, D = paths
    for j in (12, 60, 120):
        assert D[:, j].mean() == pytest.approx(MODEL.discount(GRID[j]), rel=2e-3)


def test_bond_prices_are_martingales(paths):
    r, D = paths
    j = 60  # t = 5y
    mc = (D[:, j] * MODEL.bond(GRID[j], 10.0, r[:, j])).mean()
    assert mc == pytest.approx(MODEL.discount(10.0), rel=2e-3)


def test_par_swap_is_worth_zero_at_inception(paths, par_swap):
    r, _ = paths
    V = par_swap.value(MODEL, GRID, r)
    assert np.allclose(V[:, 0], 0.0, atol=1e-6 * par_swap.notional)


def test_exposure_vanishes_at_maturity(paths, par_swap):
    r, _ = paths
    V = par_swap.value(MODEL, GRID, r)
    assert np.all(V[:, -1] == 0.0)


def test_payer_and_receiver_are_mirror_images(paths, par_swap):
    r, _ = paths
    receiver = Swap(par_swap.notional, par_swap.fixed_rate, par_swap.maturity, payer=False)
    assert np.allclose(par_swap.value(MODEL, GRID, r), -receiver.value(MODEL, GRID, r))


def test_cva_is_zero_without_credit_risk_and_increases_with_spread(paths, par_swap):
    r, D = paths
    dee = profiles(par_swap.value(MODEL, GRID, r), D)["dEE"]
    assert cva(dee, GRID, spread=0.0) == 0.0
    assert 0.0 < cva(dee, GRID, spread=0.01) < cva(dee, GRID, spread=0.02)
    assert cs01(dee, GRID, spread=0.015) > 0.0


def test_collateral_reduces_cva(paths, par_swap):
    r, D = paths
    V = par_swap.value(MODEL, GRID, r)
    naked = cva(profiles(V, D)["dEE"], GRID, 0.015)
    margined = cva(profiles(collateralise(V, GRID, mpor=1 / 12), D)["dEE"], GRID, 0.015)
    perfect = cva(profiles(collateralise(V, GRID), D)["dEE"], GRID, 0.015)
    assert perfect == 0.0
    assert 0.0 < margined < naked
