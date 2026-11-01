"""Etap 1: kopuła Franka, MNW krawędzi, modele zależności (rvine, cvine, dvine, gauss, t, indep, empirical),
PIT poza próbą i ocena modeli."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

from pf.models import dependence as dep
from pf.models.copulas import PairCopula, fit_mle, frank_tau, kendall_tau, param_from_tau, select
from pf.models.gof import compare_with_reference, oos_log_score, pit_panel, vuong_test
from pf.models.margins import MarginFit, fit_margin, simulate_daily
from pf.models.rvine import Edge, RVine, kendall_matrix, max_hamiltonian_path, path_weight, pseudo_observations

CFG = {"rvine": {"truncation": 3, "families": ["gauss", "t", "clayton", "gumbel", "frank"], "rotations": True,
                 "criterion": "aic", "estimation": "mle", "indep_test_level": 0.05, "t_df_bounds": [2.05, 30.0]},
       "dependence": {"psd_eig_min": 1e-6, "student_t": {"df_grid": [2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]}}}


# --------------------------------------------------------------------------- Frank, Gumbel, MNW

@pytest.mark.parametrize("tau", [0.4, -0.4, 0.85])
def test_frank_h_density_and_inverse(tau):
    c = PairCopula("frank", 0, param_from_tau("frank", tau))
    assert c.tau() == pytest.approx(tau, abs=1e-8)
    for u, v in [(0.3, 0.7), (0.05, 0.9), (0.8, 0.2)]:
        hq = integrate.quad(lambda s: np.exp(c.logpdf(s, v)), 0, u, limit=200)[0]
        assert float(c.h(u, v)) == pytest.approx(hq, abs=1e-6)
    if abs(tau) < 0.8:
        total = integrate.dblquad(lambda v, u: np.exp(c.logpdf(u, v)), 0, 1, 0, 1, epsabs=1e-6)[0]
        assert total == pytest.approx(1.0, abs=1e-3)
    rng = np.random.default_rng(0)
    w, v = rng.random(20_000), rng.random(20_000)
    u = c.hinv(w, v)
    np.testing.assert_allclose(c.h(u, v), w, atol=1e-9)
    assert kendall_tau(u, v)[0] == pytest.approx(tau, abs=0.02)
    assert c.tail_dependence() == (0.0, 0.0)


def test_frank_tau_is_odd_and_monotone():
    th = np.array([-20, -5, -1, 1, 5, 20.0])
    taus = np.array([frank_tau(x) for x in th])
    np.testing.assert_allclose(taus[:3], -taus[3:][::-1])
    assert np.all(np.diff(taus) > 0)


@pytest.mark.parametrize("theta", [4.0, -6.0])
def test_frank_mle_recovers_parameter(theta):
    true = PairCopula("frank", 0, (theta,))
    rng = np.random.default_rng(3)
    v = rng.random(5000)
    u = true.hinv(rng.random(5000), v)
    start = param_from_tau("frank", kendall_tau(u, v)[0])
    p, ll = fit_mle(u, v, "frank", 0, start)
    assert p[0] == pytest.approx(theta, rel=0.08)
    assert ll >= np.sum(PairCopula("frank", 0, start).logpdf(u, v)) - 1e-9
    cop, _ = select(u, v, families=("gauss", "t", "clayton", "gumbel", "frank"), estimation="mle")
    assert cop.family == "frank"


@pytest.mark.parametrize("family,rotation", [("clayton", 0), ("gumbel", 180), ("gauss", 0)])
def test_mle_not_worse_than_itau(family, rotation):
    true = PairCopula(family, rotation, param_from_tau(family, 0.5))
    rng = np.random.default_rng(4)
    v = rng.random(3000)
    u = true.hinv(rng.random(3000), v)
    c_itau, i_itau = select(u, v, families=(family,), estimation="itau")
    c_mle, i_mle = select(u, v, families=(family,), estimation="mle")
    assert i_mle["loglik"] >= i_itau["loglik"] - 1e-9
    assert c_mle.tau() == pytest.approx(0.5, abs=0.03)


def test_gumbel_inverse_extremes():
    for th in (1.0 + 1e-6, 1.5, 10.0, 30.0):
        c = PairCopula("gumbel", 0, (th,))
        w = np.array([1e-10, 1e-4, 0.5, 1 - 1e-4, 1 - 1e-10] * 3)
        v = np.repeat([1e-8, 0.5, 1 - 1e-8], 5)
        u = c.hinv(w, v)
        assert np.isfinite(u).all() and ((u > 0) & (u < 1)).all()
        mid = (w > 1e-6) & (w < 1 - 1e-6) & (v > 1e-6) & (v < 1 - 1e-6)
        np.testing.assert_allclose(c.h(u[mid], v[mid]), w[mid], atol=1e-8)


# --------------------------------------------------------------------------- macierze korelacji

def test_nearest_correlation():
    A = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])   # nie jest PSD
    assert np.linalg.eigvalsh(A).min() < 0
    R = dep.nearest_correlation(A, eig_min=1e-6)
    np.testing.assert_allclose(np.diag(R), 1.0)
    assert np.linalg.eigvalsh(R).min() >= 1e-7 and np.allclose(R, R.T)
    B = np.array([[1.0, 0.3], [0.3, 1.0]])
    np.testing.assert_allclose(dep.nearest_correlation(B), B)


def random_corr(d, seed):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(d, 2))
    S = F @ F.T + np.diag(rng.uniform(0.3, 1.0, d))
    s = np.sqrt(np.diag(S))
    return S / np.outer(s, s)


# --------------------------------------------------------------------------- Gauss, t

def test_gauss_copula_recovery_and_density():
    R = random_corr(5, 0)
    true = dep.GaussCopula([str(i) for i in range(5)], R, n_obs=0)
    U = true.simulate(6000, np.random.default_rng(1))
    m = dep.fit_model("gauss", pseudo_observations(U), list("abcde"), CFG)
    np.testing.assert_allclose(m.R, R, atol=0.04)
    tau = np.array([[kendall_tau(U[:, i], U[:, j])[0] for j in range(5)] for i in range(5)])
    np.testing.assert_allclose(tau, 2 / np.pi * np.arcsin(R), atol=0.03)
    x = special.ndtri(U[:50])
    ref = stats.multivariate_normal(np.zeros(5), R).logpdf(x) - stats.norm.logpdf(x).sum(axis=1)
    np.testing.assert_allclose(true.logpdf(U[:50]), ref, rtol=1e-8, atol=1e-8)
    assert m.n_params == 10 and m.aic() == pytest.approx(20 - 2 * m.loglik)


def test_student_t_copula_recovery_and_density():
    R = random_corr(5, 2)
    true = dep.StudentTCopula([str(i) for i in range(5)], R, nu=5.0, n_obs=0)
    U = true.simulate(6000, np.random.default_rng(3))
    m = dep.fit_model("student_t", pseudo_observations(U), list("abcde"), CFG)
    np.testing.assert_allclose(m.R, R, atol=0.05)
    assert 3.5 < m.nu < 7.5
    x = special.stdtrit(5.0, U[:50])
    ref = stats.multivariate_t(np.zeros(5), R, df=5.0).logpdf(x) - stats.t.logpdf(x, 5.0).sum(axis=1)
    np.testing.assert_allclose(true.logpdf(U[:50]), ref, rtol=1e-8, atol=1e-8)
    g = dep.fit_model("gauss", pseudo_observations(U), list("abcde"), CFG)
    assert m.aic() < g.aic()  # ogony t: kopuła t lepsza od Gaussa


# --------------------------------------------------------------------------- C-vine, D-vine

def true_cvine():
    tau = {(0, 1): 0.6, (0, 2): 0.5, (0, 3): 0.6}
    trees = [[Edge(1, 0, 1, frozenset(), PairCopula("clayton", 0, param_from_tau("clayton", 0.6))),
              Edge(1, 0, 2, frozenset(), PairCopula("gumbel", 0, param_from_tau("gumbel", 0.5))),
              Edge(1, 0, 3, frozenset(), PairCopula("frank", 0, param_from_tau("frank", 0.6)))],
             [Edge(2, 1, 2, frozenset({0}), PairCopula("gauss", 0, param_from_tau("gauss", 0.25))),
              Edge(2, 1, 3, frozenset({0}), PairCopula("gauss", 0, param_from_tau("gauss", 0.2)))],
             [Edge(3, 2, 3, frozenset({0, 1}), PairCopula("clayton", 0, param_from_tau("clayton", 0.05)))]]
    return RVine([str(i) for i in range(4)], trees, truncation=3, structure="cvine"), tau


def test_cvine_recovery():
    true, tau = true_cvine()
    U = true.simulate(8000, np.random.default_rng(5))
    m = dep.fit_model("cvine", pseudo_observations(U), [str(i) for i in range(4)], CFG)
    t1 = m.vine.trees[0]
    roots = set.intersection(*[{e.a, e.b} for e in t1])
    T = np.abs(kendall_matrix(pseudo_observations(U)))
    assert roots == {0} == {int(np.argmax(T.sum(axis=1)))}  # korzeń maksymalizuje Σ|τ̂|
    got = {frozenset((e.a, e.b)): e for e in t1}
    for e in true.trees[0]:
        f = got[frozenset((e.a, e.b))]
        assert f.copula.family == e.copula.family
        assert f.copula.tau() == pytest.approx(e.copula.tau(), abs=0.03)
    t2_root = set.intersection(*[{e.a, e.b} for e in m.vine.trees[1]])
    assert t2_root == {1}
    for e2 in m.vine.trees[1]:
        ref = [e for e in true.trees[1] if e.full == e2.full][0]
        assert e2.copula.tau() == pytest.approx(ref.copula.tau(), abs=0.03)
    # τ̂ symulacji z dopasowanego modelu ≈ τ modelu
    S = m.simulate(8000, np.random.default_rng(6))
    for e in t1:
        assert kendall_tau(S[:, e.a], S[:, e.b])[0] == pytest.approx(e.copula.tau(), abs=0.03)


def test_dvine_recovery_and_order():
    order = [2, 0, 3, 1]
    pairs = list(zip(order[:-1], order[1:]))
    fams = [("gumbel", 0.6), ("clayton", 0.5), ("frank", 0.55)]
    trees = [[Edge(1, a, b, frozenset(), PairCopula(f, 0, param_from_tau(f, t))) for (a, b), (f, t) in zip(pairs, fams)],
             [Edge(2, 2, 3, frozenset({0}), PairCopula("gauss", 0, (0.3,))),
              Edge(2, 0, 1, frozenset({3}), PairCopula("t", 0, (0.3, 5.0)))],
             [Edge(3, 2, 1, frozenset({0, 3}), PairCopula())]]
    true = RVine([str(i) for i in range(4)], trees, truncation=3, structure="dvine")
    U = true.simulate(8000, np.random.default_rng(7))
    m = dep.fit_model("dvine", pseudo_observations(U), [str(i) for i in range(4)], CFG)
    assert {frozenset((e.a, e.b)) for e in m.vine.trees[0]} == {frozenset(p) for p in pairs}
    got = {frozenset((e.a, e.b)): e for e in m.vine.trees[0]}
    for (a, b), (f, t) in zip(pairs, fams):
        assert got[frozenset((a, b))].copula.family == f
        assert got[frozenset((a, b))].copula.tau() == pytest.approx(t, abs=0.03)
    assert m.vine.trees[2][0].full == frozenset(range(4))
    # struktura poprawna: plan symulacji obejmuje wszystkie zmienne
    assert sorted(a for a, _ in m.vine.sampling_plan()) == [0, 1, 2, 3]


def test_hamiltonian_path_heuristic():
    rng = np.random.default_rng(8)
    d = 9
    W = rng.uniform(0, 0.3, (d, d))
    perm = rng.permutation(d)
    for a, b in zip(perm[:-1], perm[1:]):
        W[a, b] = 0.9
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0)
    path = max_hamiltonian_path(W)
    assert sorted(path) == list(range(d))
    assert path_weight(W, path) == pytest.approx(0.9 * (d - 1))


# --------------------------------------------------------------------------- interfejs wspólny

@pytest.fixture(scope="module")
def factor_U():
    rng = np.random.default_rng(9)
    f = rng.standard_t(4, size=(1500, 1))
    X = 0.8 * f + rng.standard_normal((1500, 6)) * np.linspace(0.5, 1.2, 6)
    return pseudo_observations(X)


@pytest.fixture(scope="module")
def fitted(factor_U):
    names = list("ABCDEF")
    return {n: dep.fit_model(n, factor_U, names, CFG) for n in dep.REGISTRY}


def test_registry_and_interface(fitted, factor_U, tmp_path):
    assert set(fitted) == {"rvine", "cvine", "dvine", "gauss", "student_t", "indep", "empirical"}
    W = np.random.default_rng(10).random((400, 6))
    for name, m in fitted.items():
        S = m.simulate(400, np.random.default_rng(1), W=W)
        assert S.shape == (400, 6) and ((S > 0) & (S < 1)).all()
        m2 = dep.load_model(dep.save_model(m, tmp_path / f"{name}.json.gz"))
        np.testing.assert_allclose(m2.simulate(400, np.random.default_rng(1), W=W), S)
        if m.has_density:
            assert m.logpdf(factor_U).sum() == pytest.approx(m.loglik, rel=1e-6, abs=1e-6)
            np.testing.assert_allclose(m2.logpdf(factor_U[:20]), m.logpdf(factor_U[:20]))
            assert np.isfinite(m.aic()) and m.bic() > m.aic() - 1e-9 or m.n_params == 0
        else:
            assert np.isnan(m.aic())
    # wspólne W: modele vine i Gauss deterministyczne względem W (niezależnie od rng)
    for name in ("rvine", "cvine", "dvine", "gauss", "indep", "empirical"):
        a = fitted[name].simulate(400, np.random.default_rng(1), W=W)
        b = fitted[name].simulate(400, np.random.default_rng(2), W=W)
        np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(fitted["indep"].simulate(400, None, W=W), np.clip(W, 1e-12, 1 - 1e-12))
    rows = fitted["empirical"].simulate(400, None, W=W)
    assert set(map(tuple, rows.round(12))) <= set(map(tuple, factor_U.round(12)))


def test_vine_ranking_and_vuong(fitted, factor_U):
    table = compare_with_reference(fitted, factor_U, "rvine")
    t = table.set_index("model")
    assert t.loc["rvine", "aic"] < t.loc["indep", "aic"]
    assert t.loc["indep", "vuong_winner"] == "rvine"
    assert np.isnan(t.loc["empirical", "aic"]) and pd.isna(t.loc["empirical", "vuong_stat"])
    same = vuong_test(np.ones(100), np.ones(100), 3, 3)
    assert same["winner"] == 0 and same["stat"] == 0.0
    rng = np.random.default_rng(0)
    ll = rng.normal(size=500)
    assert vuong_test(ll + 0.5, ll + rng.normal(0, 0.1, 500), 1, 1)["winner"] == 1
    # poprawka Schwarza karze model z większą liczbą parametrów
    v_pen = vuong_test(ll + 0.01, ll, 50, 1)
    assert v_pen["correction"] == pytest.approx(49 * np.log(500) / 2) and v_pen["stat"] < 0


# --------------------------------------------------------------------------- PIT poza próbą

def margin_fit(**kw):
    base = dict(name="X", mu=0.03, phi=0.05, omega=0.02, alpha=0.05, gamma=0.1, beta=0.85, eta=7.0, lam=-0.1,
                scale=100.0, last_r=0.4, last_eps=-1.0, last_sigma2=1.3, loglik=0.0, n_obs=1, converged=True)
    return MarginFit(**{**base, **kw})


def test_pit_forward_inverts_simulation():
    f = margin_fit()
    U = np.random.default_rng(11).random((1, 3000, 1))
    r = simulate_daily([f], U)[0, :, 0]
    u = f.pit_forward(r)
    np.testing.assert_allclose(u, U[0, :, 0], atol=1e-7)
    r2 = r.copy()
    r2[5] = np.nan                       # sesja bez notowania: u = NaN, stan filtra bez zmian
    u2 = f.pit_forward(r2)
    assert np.isnan(u2[5]) and u2[4] == pytest.approx(u[4])


def test_pit_forward_uniform_on_fitted_model():
    from tests.test_margins_opt import garch_series

    fit, _ = fit_margin(garch_series(n=2500, seed=1))
    U = np.random.default_rng(12).random((1, 2000, 1))
    r = simulate_daily([fit], U)[0, :, 0]
    u = fit.pit_forward(r)
    assert stats.kstest(u, "uniform").pvalue > 0.05
    np.testing.assert_allclose(u, U[0, :, 0], atol=1e-6)


def test_pit_panel_and_log_score(fitted):
    idx = pd.bdate_range("2024-01-01", periods=5)
    fits = {a: margin_fit(name=a) for a in ["P.WA", "S.WA", "Q", "R"]}
    r = pd.DataFrame({"P.WA": [0.01, np.nan, 0.0, 0.01, np.nan], "S.WA": [0.0, 0.01, 0.0, 0.0, np.nan],
                      "Q": [0.0, 0.01, np.nan, -0.01, 0.01], "R": [0.01, 0.0, np.nan, 0.02, 0.0]}, index=idx)
    U = pit_panel(fits, r)
    # 2024-01-03: rynek US zamknięty (Q, R puste) → usunięty; 2024-01-05: GPW zamknięta → usunięty
    assert list(U.index) == [idx[0], idx[1], idx[3]]
    assert U.loc[idx[1], "P.WA"] == pytest.approx(
        float(__import__("pf.models.margins", fromlist=["x"]).skewt_cdf(np.array([0.0]), 7.0, -0.1)[0]))
    assert U.notna().all().all()
    m = fitted["gauss"]
    Uo = pd.DataFrame(np.random.default_rng(13).random((30, 6)), columns=m.names)
    ls = oos_log_score(m, Uo)
    np.testing.assert_allclose(ls, m.logpdf(Uo.to_numpy()))
    with pytest.raises(ValueError):
        oos_log_score(fitted["empirical"], Uo)


def test_residual_panel_trims_pre_listing_span():
    from pf.models.margins import residual_panel

    idx = pd.bdate_range("2024-01-01", periods=6)
    z = pd.DataFrame({"A": [0.1, 0.2, -0.3, 0.4, np.nan, 0.5], "B": [np.nan, np.nan, 0.3, 0.1, 0.2, -0.1]}, index=idx)
    p = residual_panel(z)
    assert p.index[0] == idx[2] and p.loc[idx[4], "A"] == 0.0      # pojedynczy brak → 0, okres przed B – usunięty
    assert len(residual_panel(z, trim_leading=False)) == 6
