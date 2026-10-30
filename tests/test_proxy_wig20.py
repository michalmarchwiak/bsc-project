import numpy as np
import pytest

from pf.options import proxy_wig20
from pf.options.surface import SkewParams

P = SkewParams(a=0.02, b=0.85, c=-0.15, d=-1.2)


def test_atm_formula():
    # σ_WIG20 = σ_GARCH,WIG20 · VIX / σ_GARCH,SPX
    assert proxy_wig20.atm_vol_30d(0.22, 0.16, 0.20) == pytest.approx(0.22 * 0.20 / 0.16)


def test_same_skew_slope_as_spx():
    k = np.array([-0.1, 0.0])
    s = proxy_wig20.implied_vol(k, 0.2, P, 0.22, 0.16)
    assert (s[0] - s[1]) / -0.1 == pytest.approx(P.slope(0.2))
    assert s[1] == pytest.approx(0.22 / 0.16 * P.atm(0.2))


def test_sensitivity_shifts_three_pp():
    k = np.linspace(-0.2, 0.05, 6)
    grid = proxy_wig20.sensitivity(k, 0.25, P, 0.24, 0.18)
    np.testing.assert_allclose(grid["+3pp"] - grid["base"], 0.03)
    np.testing.assert_allclose(grid["base"] - grid["-3pp"], 0.03)
