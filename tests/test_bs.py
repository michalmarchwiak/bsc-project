import numpy as np
import pytest

from pf.options import bs


def test_reference_hull():
    # Hull, „Options, Futures and Other Derivatives”, przykład 15.6: S=42, K=40, r=10%, σ=20%, T=0,5.
    assert bs.price(42, 40, 0.5, 0.10, 0.0, 0.20, True) == pytest.approx(4.7594, abs=1e-4)
    assert bs.price(42, 40, 0.5, 0.10, 0.0, 0.20, False) == pytest.approx(0.8086, abs=1e-4)


def test_reference_haug_with_dividend_yield():
    # Haug, „The Complete Guide to Option Pricing Formulas”, uogólniony BS: S=75, K=70, T=0,5,
    # r=10%, b=r−q=5%, σ=35% → put = 4,0870.
    assert bs.price(75, 70, 0.5, 0.10, 0.05, 0.35, False) == pytest.approx(4.0870, abs=1e-4)


@pytest.mark.parametrize("q", [0.0, 0.02])
def test_put_call_parity(q):
    S, K = 100.0, np.linspace(60, 160, 51)
    T, r, sigma = 0.75, 0.04, 0.3
    c = bs.price(S, K, T, r, q, sigma, True)
    p = bs.price(S, K, T, r, q, sigma, False)
    np.testing.assert_allclose(c - p, S * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-10)


def test_vectorised_over_scenarios():
    rng = np.random.default_rng(0)
    S = 100 * np.exp(rng.normal(0, 0.2, size=10_000))
    p = bs.price(S, 95.0, 0.25, 0.03, 0.01, 0.2, False)
    assert p.shape == S.shape
    assert np.all(p >= np.maximum(95.0 * np.exp(-0.03 * 0.25) - S * np.exp(-0.01 * 0.25), 0) - 1e-12)


def test_expiry_equals_payoff():
    S = np.linspace(50, 150, 101)
    np.testing.assert_allclose(bs.price(S, 100, 0.0, 0.05, 0.0, 0.2, False), np.maximum(100 - S, 0))
    np.testing.assert_allclose(bs.price(S, 100, 0.0, 0.05, 0.0, 0.2, True), np.maximum(S - 100, 0))


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("K", [80.0, 100.0, 125.0])
def test_greeks_vs_finite_differences(is_call, K):
    S, T, r, q, sigma = 100.0, 0.6, 0.03, 0.015, 0.25
    g = bs.greeks(S, K, T, r, q, sigma, is_call)
    hS, hs, hT = 1e-3, 1e-5, 1e-6
    f = lambda **kw: bs.price(**{**dict(S=S, K=K, T=T, r=r, q=q, sigma=sigma, is_call=is_call), **kw})
    assert g["delta"] == pytest.approx((f(S=S + hS) - f(S=S - hS)) / (2 * hS), abs=1e-7)
    assert g["gamma"] == pytest.approx((f(S=S + hS) - 2 * f() + f(S=S - hS)) / hS ** 2, abs=1e-5)
    assert g["vega"] == pytest.approx((f(sigma=sigma + hs) - f(sigma=sigma - hs)) / (2 * hs), abs=1e-5)
    # θ = ∂V/∂t = −∂V/∂T
    assert g["theta"] == pytest.approx(-(f(T=T + hT) - f(T=T - hT)) / (2 * hT), abs=1e-4)
