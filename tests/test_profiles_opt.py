"""Etap 3: ekwiwalent pewności, miara zależności, optymalizacja dwupoziomowa profili, benchmark MV."""

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from pf import config
from pf.opt import dependence_penalty as dp
from pf.opt.cvar import empirical_es, min_es
from pf.opt.mv import ledoit_wolf, mv_two_level, solve_mv
from pf.opt.profiles import (Profile, equal_weight, load_profiles, optimize_two_level, relaxed_upper_bounds,
                             solve_ce, subfund_data)
from pf.opt.utility import ce_and_grad, certainty_equivalent

ALPHA = 0.975
OPT = {"maxiter": 500, "ftol": 1e-12}


def scenarios(S=4000, n=5, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_t(5, size=(S, 1))
    mu = np.linspace(0.002, 0.012, n)
    sd = np.linspace(0.015, 0.07, n)
    Z = 0.6 * f + 0.8 * rng.standard_t(5, size=(S, n))
    return mu + sd * Z / np.sqrt(5 / 3)


# --------------------------------------------------------------------------- CE

def test_ce_properties():
    R = scenarios()
    w = np.full(5, 0.2)
    r = R @ w
    assert certainty_equivalent(R, w, 1.0) == pytest.approx(np.exp(np.mean(np.log1p(r))) - 1, rel=1e-12)
    for eps in (1e-5, -1e-5):
        assert certainty_equivalent(R, w, 1.0 + eps) == pytest.approx(certainty_equivalent(R, w, 1.0), abs=1e-8)
    for g in (0.5, 1.0, 3.0, 10.0):
        assert certainty_equivalent(R, w, g) <= r.mean() + 1e-15
    assert certainty_equivalent(np.full((100, 1), 0.013), np.ones(1), 7.0) == pytest.approx(0.013, rel=1e-12)
    # CE maleje z γ
    ces = [certainty_equivalent(R, w, g) for g in (1.5, 3, 5, 7, 10)]
    assert all(np.diff(ces) < 0)


@pytest.mark.parametrize("gamma", [1.0, 1.5, 5.0, 10.0])
def test_ce_gradient_matches_finite_differences(gamma):
    R = scenarios(S=2000)
    w = np.array([0.1, 0.3, 0.2, 0.25, 0.15])
    _, g = ce_and_grad(R, w, gamma)
    num = optimize.approx_fprime(w, lambda x: ce_and_grad(R, x, gamma)[0], 1e-7)
    np.testing.assert_allclose(g, num, rtol=1e-4, atol=1e-7)
    M = dp.lower_tail_matrix(R, 0.05)
    _, gd = dp.penalty_and_grad(M, w)
    np.testing.assert_allclose(gd, optimize.approx_fprime(w, lambda x: dp.penalty_and_grad(M, x)[0], 1e-7),
                               rtol=1e-5, atol=1e-8)


# --------------------------------------------------------------------------- miara zależności

def test_tail_dependence_matrix_limits_and_psd():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40000, 3))
    L = dp.lower_tail_matrix(X, 0.05)
    off = L[~np.eye(3, dtype=bool)]
    np.testing.assert_allclose(off, 0.05, atol=0.01)            # niezależność: Λ_ij ≈ q
    np.testing.assert_allclose(np.diag(L), 1.0, atol=1e-3)
    x = rng.standard_normal(5000)
    C = dp.lower_tail_matrix(np.column_stack([x, np.exp(x), 3 * x + 1]), 0.05)
    np.testing.assert_allclose(C, 1.0, atol=1e-3)                # komonotoniczność: Λ_ij = 1
    for M in (L, dp.spearman_matrix(scenarios()), dp.lower_tail_matrix(scenarios(), 0.1)):
        assert np.linalg.eigvalsh(M).min() >= -1e-10
    np.testing.assert_allclose(np.diag(dp.spearman_matrix(scenarios())), 1.0)


# --------------------------------------------------------------------------- problem CE − κD

def test_large_kappa_gives_min_dependence():
    R = scenarios()
    M = dp.lower_tail_matrix(R, 0.05)
    w, info = solve_ce(R, 5.0, 1e6, 1.0, M, 1.0, OPT)
    ref = optimize.minimize(lambda x: x @ M @ x, np.full(5, 0.2), jac=lambda x: 2 * M @ x, method="SLSQP",
                            bounds=[(0, 1)] * 5, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
                            options={"ftol": 1e-15, "maxiter": 500})
    assert w @ M @ w == pytest.approx(ref.fun, rel=1e-4)


def test_large_gamma_approaches_min_risk():
    rng = np.random.default_rng(2)
    S, n = 5000, 4
    C = np.array([[1, .3, .2, .1], [.3, 1, .3, .2], [.2, .3, 1, .4], [.1, .2, .4, 1]])
    sd = np.array([0.02, 0.04, 0.06, 0.08])
    R = 0.002 + rng.multivariate_normal(np.zeros(n), C * np.outer(sd, sd), S)
    # umiarkowanie duże γ: CE ≈ μ − (γ/2)σ² → portfel minimalnej wariancji; γ → ∞: CE → min_s R_s w
    # (najgorszy scenariusz) → portfel minimalnego ryzyka ogona
    var = [np.var(R @ solve_ce(R, g, 0.0, 1.0, opt=OPT)[0]) for g in (1.0, 5.0, 50.0)]
    assert all(np.diff(var) < 1e-12)
    Sig = np.cov(R, rowvar=False)
    w_mv = optimize.minimize(lambda x: x @ Sig @ x, np.full(n, 0.25), jac=lambda x: 2 * Sig @ x, method="SLSQP",
                             bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
                             options={"ftol": 1e-16, "maxiter": 1000}).x
    assert var[-1] == pytest.approx(w_mv @ Sig @ w_mv, rel=0.03)
    w_es = min_es(R, ALPHA).weights
    assert empirical_es(-R @ solve_ce(R, 400.0, 0.0, 1.0, opt=OPT)[0], ALPHA) == pytest.approx(
        empirical_es(-R @ w_es, ALPHA), rel=0.05)


def test_relaxed_upper_bounds():
    ub, flags = relaxed_upper_bounds(np.array([0.2, 0.25, 1.0]), 0.4, 0.05)
    assert flags["lt_relaxed"] and ub.sum() == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(ub, np.minimum(0.4, flags["lt_factor"] * 0.05 / np.array([0.2, 0.25, 1.0])))
    ub2, f2 = relaxed_upper_bounds(np.array([0.2] * 9), 0.4, 0.05)
    assert not f2["lt_relaxed"] and np.allclose(ub2, 0.25)
    ub3, f3 = relaxed_upper_bounds(np.array([0.5, 0.5]), 0.4, 0.25)
    assert f3["wmax_relaxed"] and ub3.sum() >= 1 - 1e-12


SUBFUNDS = {"SF1": {"tickers": ["A1", "A2", "A3", "A4", "A5"]}, "SF2": {"tickers": ["B1", "B2", "B3"]},
            "SF3": {"tickers": ["C1", "C2", "C3", "C4", "C5", "C6"]}, "SF4": {"tickers": ["D1"]}}


@pytest.fixture(scope="module")
def sfs():
    rng = np.random.default_rng(3)
    names = [t for spec in SUBFUNDS.values() for t in spec["tickers"]]
    R = np.column_stack([scenarios(3000, 5, 4), scenarios(3000, 3, 5) * 0.5, scenarios(3000, 6, 6),
                         0.003 + 0.01 * rng.standard_normal((3000, 1))])
    return subfund_data(R, names, SUBFUNDS), names


@pytest.mark.parametrize("strategy", ["ce_dep", "ce", "maxret_es", "min_es"])
@pytest.mark.parametrize("pid", ["P1", "P3", "P5"])
def test_two_level_limits(sfs, strategy, pid):
    data, names = sfs
    prof = load_profiles(config.load("profiles"))[pid]
    res = optimize_two_level(data, prof, strategy, ALPHA, OPT)
    assert res.W.sum() == pytest.approx(1.0) and (res.W >= -1e-12).all()
    for k, w in res.w.items():
        assert w.sum() == pytest.approx(1.0)
        assert (w <= max(prof.c_sf, 1 / len(w)) + 1e-9).all()
    factor = res.flags["lt_factor"]
    assert (res.W <= prof.w_max + 1e-9).all() or res.flags["wmax_relaxed"]
    for k, w in res.w.items():
        assert res.W[k] * w.max() <= factor * prof.c_lt + 1e-9
    x = res.x
    assert x.sum() == pytest.approx(1.0) and x.max() <= factor * prof.c_lt + 1e-9
    # SF4 ma jedno aktywo → W_4 ≤ c_LT
    assert res.W["SF4"] <= factor * prof.c_lt + 1e-9


def test_ce_dep_vs_ce_and_ew(sfs):
    data, _ = sfs
    prof = Profile("PX", "test", 5.0, 0.35, 0.3, 1.0, 1.0)
    a = optimize_two_level(data, prof, "ce", ALPHA, OPT)
    b = optimize_two_level(data, prof, "ce_dep", ALPHA, OPT)
    sf = data[0]
    M = sf.matrix("lambda_L", 0.05)
    wa, wb = a.w["SF1"].to_numpy(), b.w["SF1"].to_numpy()
    assert wb @ M @ wb <= wa @ M @ wa + 1e-12                            # kara obniża zależność w ogonie
    assert certainty_equivalent(sf.R, wa, 5.0) >= certainty_equivalent(sf.R, wb, 5.0) - 1e-12
    ew = equal_weight({s.name: s.tickers for s in data})
    assert ew.W.to_numpy() == pytest.approx(0.25) and ew.x.sum() == pytest.approx(1.0)


def test_ledoit_wolf_and_mv():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((60, 40)) * 0.01
    Sig, delta = ledoit_wolf(X)
    assert 0 <= delta <= 1 and np.linalg.eigvalsh(Sig).min() > 0
    # zgodność z wzorem zamkniętym: β = (Σ‖x‖⁴/T − ‖S‖²)/(nT), δ = min(β, d²)/d²
    Xc = X - X.mean(0)
    T, n = Xc.shape
    S = Xc.T @ Xc / T
    m = np.trace(S) / n
    d2 = np.sum((S - m * np.eye(n)) ** 2) / n
    beta = (np.sum(np.sum(Xc ** 2, axis=1) ** 2) / T - np.sum(S ** 2)) / (n * T)
    assert delta == pytest.approx(min(beta, d2) / d2, rel=1e-10)
    big = rng.standard_normal((20000, 3)) @ np.array([[1, .5, 0], [0, 1, .3], [0, 0, 1]])
    assert ledoit_wolf(big)[1] < 0.01                                     # T ≫ n: słaby shrinkage
    mu = np.array([0.01, 0.02, 0.005])
    Sg = np.diag([0.002, 0.01, 0.0005])
    w = solve_mv(mu, Sg, 5.0, 0.6)
    assert w.sum() == pytest.approx(1.0) and (w <= 0.6 + 1e-9).all()
    # KKT dla aktywnych dodatnich wag: μ_i − γ(Σw)_i równe
    grad = mu - 5.0 * Sg @ w
    inner = (w > 1e-6) & (w < 0.6 - 1e-6)
    assert np.ptp(grad[inner]) < 1e-6 if inner.sum() > 1 else True


def test_mv_two_level_limits():
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2015-01-01", periods=800)
    cols = ["A1", "A2", "A3", "B1", "B2"]
    window = pd.DataFrame(rng.standard_normal((800, 5)) * 0.01 + 0.0003, index=idx, columns=cols)
    window.iloc[:50, 4] = np.nan
    prof = load_profiles(config.load("profiles"))["P2"]
    res = mv_two_level(window, {"SF1": ["A1", "A2", "A3"], "SF2": ["B1", "B2"]}, prof)
    assert res.W.sum() == pytest.approx(1.0)
    assert res.x.max() <= res.flags["lt_factor"] * prof.c_lt + 1e-9
    assert res.flags["lt_relaxed"]   # 2 SF × limity P2 są niedopuszczalne bez luzowania
