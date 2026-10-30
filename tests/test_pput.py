import numpy as np
import pandas as pd
import pytest

from pf.options import pput
from pf.options.surface import SkewParams, VolSurface

P = SkewParams(a=0.02, b=0.85, c=-0.15, d=-1.2)


@pytest.mark.parametrize("y,m,expected", [(2008, 10, "2008-10-17"), (2020, 3, "2020-03-20"),
                                          (2022, 6, "2022-06-17"), (2024, 3, "2024-03-15")])
def test_third_friday(y, m, expected):
    assert pput.third_friday(y, m) == pd.Timestamp(expected)


def test_roll_dates_shift_to_next_session():
    idx = pd.bdate_range("2024-03-01", "2024-04-30").drop(pd.Timestamp("2024-03-15"))
    assert list(pput.roll_dates(idx)) == [pd.Timestamp("2024-03-18"), pd.Timestamp("2024-04-19")]


def market(seed=0, n=600, crash_at=None):
    idx = pd.bdate_range("2019-01-02", periods=n)
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.012, n)
    if crash_at is not None:
        r[crash_at:crash_at + 15] = -0.03
    S = pd.Series(3000 * np.exp(np.cumsum(r)), index=idx)
    vix = pd.DataFrame({"VIX": 15 + 300 * pd.Series(r, index=idx).rolling(10, min_periods=1).std()}, index=idx)
    return S, vix


def test_put_protection_floor_per_roll():
    """Bez dywidend i przy r = q = 0: V_{t1}/V_{t0} = max(S_{t1}, K)/(S_{t0} + P₀) ≥ (K/S₀)·S₀/(S₀ + P₀)."""
    S, vix = market(crash_at=300)
    zero = pd.Series(0.0, index=S.index)
    rep = pput.replicate(S, S, vix, P, zero, zero)
    rolls = pput.roll_dates(rep.index)
    checked = 0
    for t0, t1 in zip(rolls[:-1], rolls[1:]):
        expiry = pput.third_friday(t1.year, t1.month)
        if t1 != expiry:
            continue  # rolowanie przesunięte na sesję po wygaśnięciu
        K = 5.0 * np.floor(0.95 * S[t0] / 5.0)
        T0 = (expiry - t0).days / 365
        P0 = float(VolSurface({"VIX": vix.loc[t0, "VIX"]}, P).price(S[t0], K, T0, 0, 0, False))
        ratio = rep[t1] / rep[t0]
        assert ratio == pytest.approx(max(S[t1], K) / (S[t0] + P0), rel=1e-12)
        assert ratio >= (K / S[t0]) * S[t0] / (S[t0] + P0) - 1e-12
        checked += 1
    assert checked > 10


def test_replica_beats_index_in_crash():
    S, vix = market(crash_at=300)
    zero = pd.Series(0.0, index=S.index)
    rep = pput.replicate(S, S, vix, P, zero, zero)
    window = rep.index[(rep.index >= S.index[295]) & (rep.index <= S.index[318])]
    assert rep[window[-1]] / rep[window[0]] > S[window[-1]] / S[window[0]]


def test_tracking_report_zero_for_identical_series():
    S, _ = market()
    rep = pput.tracking_report(S, S)
    assert rep.loc[rep["period"] == "all", "te_ann_pct"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert {"all", "2019", "2020"} <= set(rep["period"])


def test_trailing_div_yield():
    idx = pd.bdate_range("2019-01-02", periods=400)
    S = pd.Series(3000.0, index=idx)
    TR = pd.Series(3000.0 * np.exp(0.02 * np.arange(400) / 252), index=idx)
    q = pput.trailing_div_yield(S, TR).dropna()
    np.testing.assert_allclose(q, 0.02, atol=1e-12)
