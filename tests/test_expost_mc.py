"""Etap 5: MC ex post – FHS, bootstrap stacjonarny, wielowymiarowe MC, adaptacyjne N."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pf import config
from pf.backtest.expost_mc import FHS, MultiMC, adaptive, block_length, stationary_monthly_losses
from pf.opt.cvar import empirical_es


def iid_normal(n=3000, sd=0.01, seed=0):
    return np.random.default_rng(seed).normal(0.0003, sd, n)


def test_adaptive_stops_and_reaches_target():
    rng = np.random.default_rng(1)
    res = adaptive(lambda b: rng.standard_normal(b), 0.975, 5000, 5000, 200000, 0.01)
    assert res["converged"] and res["es_se"] / res["es"] <= 0.01
    q = stats.norm.ppf(0.975)
    assert res["es"] == pytest.approx(stats.norm.pdf(q) / 0.025, abs=4 * res["es_se"])
    capped = adaptive(lambda b: rng.standard_normal(b), 0.975, 1000, 1000, 3000, 1e-6)
    assert capped["n"] == 3000 and not capped["converged"]


def test_fhs_and_bootstrap_on_iid_series():
    """Dla iid N(μ, σ²) miesięczny ES ≈ ES rozkładu N(21μ, 21σ²) (w przybliżeniu stóp prostych)."""
    r = iid_normal()
    sd_m = 0.01 * np.sqrt(21)
    es_true = -21 * 0.0003 + sd_m * stats.norm.pdf(stats.norm.ppf(0.975)) / 0.025
    fhs = FHS(r)
    rng = np.random.default_rng(2)
    res = adaptive(lambda b: fhs.monthly_losses(b, 21, rng), 0.975, 20000, 20000, 100000, 0.01)
    assert res["es"] == pytest.approx(es_true, rel=0.08)
    b = block_length(r)
    assert 1.0 <= b < 5.0
    L = stationary_monthly_losses(r, 60000, 21, b, seed=3)
    assert empirical_es(L, 0.975) == pytest.approx(es_true, rel=0.08)


def test_stationary_bootstrap_preserves_clustering():
    rng = np.random.default_rng(4)
    n = 4000
    s2, e, r = 1e-4, 0.0, np.empty(n)
    for t in range(n):                                  # GARCH(1,1)
        s2 = 2e-6 + 0.1 * e * e + 0.88 * s2
        e = np.sqrt(s2) * rng.standard_normal()
        r[t] = e
    b = block_length(r)
    assert b > 3.0                                     # długość bloku z r² > 1 przy skupianiu zmienności
    L = stationary_monthly_losses(r, 40000, 21, b, seed=5)
    assert np.isfinite(L).all() and len(L) == 40000


def test_multivariate_mc_shapes():
    cfg = config.load("models")
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2015-01-01", periods=900)
    f = rng.standard_normal((900, 1)) * 0.006
    X = pd.DataFrame(0.7 * f + rng.standard_normal((900, 3)) * 0.005, index=idx, columns=["SF1", "SF2", "SF3"])
    mm = MultiMC(np.expm1(X), cfg)
    R = mm.monthly_returns(3000, 21, np.random.default_rng(7))
    assert R.shape == (3000, 3) and np.isfinite(R).all()
    # zależność między SF zachowana w stopach miesięcznych
    assert np.corrcoef(R, rowvar=False)[0, 1] > 0.3
