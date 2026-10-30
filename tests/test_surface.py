import numpy as np
import pandas as pd
import pytest

from pf.options import bs, surface
from pf.options.surface import SkewParams, VolSurface

TENORS = np.array([9, 30, 93, 186, 365]) / 365.0
FILTERS = {"min_bid": 0.0, "max_rel_spread": 0.5, "require_open_interest": False, "max_stale_days": 3,
           "min_days_to_expiry": 7, "max_days_to_expiry": 400}
CFG = {"tenors_days": dict(surface.DEFAULT_TENORS_DAYS), "default_rate": 0.04, "default_div_yield": 0.013}
TRUE = SkewParams(a=0.02, b=0.85, c=-0.15, d=-1.2)


def test_term_structure_hits_nodes_and_extrapolates_flat():
    vols = np.array([0.14, 0.16, 0.18, 0.19, 0.20])
    np.testing.assert_allclose(surface.interp_total_variance(TENORS, vols, TENORS), vols)
    assert surface.interp_total_variance(TENORS, vols, 2 / 365) == pytest.approx(0.14)
    assert surface.interp_total_variance(TENORS, vols, 3.0) == pytest.approx(0.20)


@pytest.mark.parametrize("vols", [[0.14, 0.16, 0.18, 0.19, 0.20],   # contango
                                  [0.60, 0.45, 0.35, 0.30, 0.28],   # backwardation (kryzys)
                                  [0.30, 0.10, 0.12, 0.11, 0.10]])  # odwrócona – arbitraż kalendarzowy
def test_total_variance_monotone(vols):
    T = np.linspace(1 / 365, 2.0, 500)
    w = surface.total_variance(TENORS, np.array(vols), T)
    assert np.all(np.diff(w) >= -1e-15)


def test_term_structure_ignores_missing_tenors():
    vols = np.array([np.nan, 0.16, 0.18, np.nan, 0.20])
    assert surface.interp_total_variance(TENORS, vols, 30 / 365) == pytest.approx(0.16)
    assert surface.interp_total_variance(TENORS, vols, 5 / 365) == pytest.approx(0.16)


def test_vix_at_tenor_per_date():
    vol = pd.DataFrame({"VIX9D": [12, 40], "VIX": [14, 35], "VIX3M": [16, 30], "VIX6M": [17, 28],
                        "VIX1Y": [18, 27]}, index=pd.to_datetime(["2024-01-02", "2020-03-16"]))
    out = surface.vix_at_tenor(vol, pd.to_datetime(["2024-01-02", "2020-03-16", "2024-01-02"]),
                               np.array([30, 30, 93]) / 365)
    np.testing.assert_allclose(out, [0.14, 0.35, 0.16])


def synthetic_obs(params: SkewParams, n: int = 4000, noise: float = 0.0, outliers: float = 0.0, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.uniform(-0.25, 0.1, n)
    vix_T = rng.uniform(0.1, 0.6, n)
    ivs = params.sigma(k, vix_T) + noise * rng.standard_normal(n)
    bad = rng.random(n) < outliers
    ivs[bad] += rng.uniform(0.1, 0.3, bad.sum())
    return pd.DataFrame({"iv": ivs, "k": k, "vix_T": vix_T})


@pytest.mark.parametrize("robust", [False, True])
def test_fit_recovers_parameters(robust):
    p = surface.fit_skew(synthetic_obs(TRUE, noise=0.002), robust=robust)
    np.testing.assert_allclose([p.a, p.b, p.c, p.d], [TRUE.a, TRUE.b, TRUE.c, TRUE.d], atol=0.01)


def test_robust_fit_resists_outliers():
    obs = synthetic_obs(TRUE, noise=0.002, outliers=0.05, seed=3)
    ols = surface.fit_skew(obs, robust=False)
    hub = surface.fit_skew(obs, robust=True)
    err = lambda p: abs(p.a - TRUE.a) + abs(p.b - TRUE.b)
    assert err(hub) < err(ols)


def test_atm_correction_and_slope():
    assert TRUE.atm(0.2) == pytest.approx(0.02 + 0.85 * 0.2)
    assert TRUE.slope(0.2) == pytest.approx(-0.15 - 1.2 * 0.2)
    # Ujemna skośność: put OTM droższy (w σ) od ATM.
    assert TRUE.sigma(-0.1, 0.2) > TRUE.sigma(0.0, 0.2)


def make_chain(params: SkewParams, dates, vol: pd.DataFrame, r=0.045, q=0.013):
    rows = []
    for d in dates:
        S = 500.0 * (1 + 0.001 * (d - dates[0]).days)
        surf = VolSurface(vol.loc[d].to_dict(), params)
        for days in (14, 30, 60, 120):
            T = days / 365
            F = S * np.exp((r - q) * T)
            for K in np.arange(0.8, 1.11, 0.025) * S:
                sigma = float(surf.implied_vol(K, T, F))
                for cp in ("C", "P"):
                    px = float(bs.price(S, K, T, r, q, sigma, cp == "C"))
                    rows.append({"quote_date": d, "expiration": d + pd.Timedelta(days=days), "strike": K,
                                 "call_put": cp, "bid": px * 0.99, "ask": px * 1.01, "underlying_price": S})
    return pd.DataFrame(rows)


def vol_frame(dates):
    base = np.linspace(14, 30, len(dates))
    return pd.DataFrame({"VIX9D": base * 0.95, "VIX": base, "VIX3M": base * 1.05, "VIX6M": base * 1.08,
                         "VIX1Y": base * 1.1}, index=dates)


def test_implied_forwards_from_parity():
    dates = pd.bdate_range("2024-01-02", periods=2)
    chain = make_chain(TRUE, dates, vol_frame(dates))
    chain["T"] = (chain["expiration"] - chain["quote_date"]).dt.days / 365
    chain["r"] = 0.045
    F = surface.implied_forwards(chain)
    expected = chain.groupby(["quote_date", "expiration"]).apply(
        lambda g: g["underlying_price"].iloc[0] * np.exp((0.045 - 0.013) * g["T"].iloc[0]))
    np.testing.assert_allclose(F.sort_index().to_numpy(), expected.sort_index().to_numpy(), rtol=1e-9)


def test_calibration_end_to_end_and_oos():
    dates = pd.bdate_range("2023-11-01", periods=30)
    vol = vol_frame(dates)
    chain = make_chain(TRUE, dates, vol)
    irx = pd.Series(100 * np.expm1(0.045), index=dates)
    obs = surface.prepare_quotes(chain, vol, irx, CFG, FILTERS)
    assert (obs["reject"] == "").mean() > 0.95
    np.testing.assert_allclose(obs["q"], 0.013, atol=1e-8)
    train, test = surface.time_split(obs, str(dates[20].date()))
    assert train["quote_date"].max() < test["quote_date"].min()
    p = surface.fit_skew(train, robust=True)
    np.testing.assert_allclose([p.a, p.b, p.c, p.d], [TRUE.a, TRUE.b, TRUE.c, TRUE.d], atol=1e-6)
    rep = surface.oos_report(test, p, {"ATM": [0.975, 1.025], "OTM5": [0.925, 0.975], "OTM10": [0.875, 0.925]})
    assert set(rep["bucket"]) == {"ATM", "OTM5", "OTM10"}
    assert (rep["n"] > 0).all()
    assert rep["rmse_price_pct"].max() < 1e-4


def test_flat_vol_underprices_otm_puts_without_skew():
    """Wycena bez korekty skośności (σ = σ_ATM) zaniża ceny putów OTM."""
    flat = SkewParams(a=TRUE.a, b=TRUE.b, c=0.0, d=0.0)
    levels = {"VIX9D": 18, "VIX": 20, "VIX3M": 21, "VIX6M": 22, "VIX1Y": 23}
    S, K, T = 100.0, 90.0, 30 / 365
    p_skew = VolSurface(levels, TRUE).price(S, K, T, 0.04, 0.01, False)
    p_flat = VolSurface(levels, flat).price(S, K, T, 0.04, 0.01, False)
    assert p_flat < p_skew
