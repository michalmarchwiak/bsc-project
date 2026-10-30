import itertools

import numpy as np
import pandas as pd
import pytest

from pf.models.margins import MarginFit, fit_margin, residual_panel, simulate_paths
from pf.opt.cvar import empirical_es, max_return_es
from pf.portfolio import lookthrough_matrix, subfund_returns
from pipelines.monthly_rebalance import estimation_window, realized_returns, rebalance_dates


def garch_series(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    omega, alpha, gamma, beta, mu, phi = 0.02, 0.05, 0.1, 0.85, 0.03, 0.05
    s2, e, r_prev, out = omega / (1 - alpha - gamma / 2 - beta), 0.0, 0.0, []
    for _ in range(n):
        s2 = omega + (alpha + gamma * (e < 0)) * e ** 2 + beta * s2
        e = np.sqrt(s2) * rng.standard_t(6) / np.sqrt(1.5)
        r_prev = mu + phi * r_prev + e
        out.append(r_prev)
    return pd.Series(np.array(out) / 100, index=pd.bdate_range("2010-01-01", periods=n), name="X")


def test_fit_margin_recovers_garch():
    fit, z = fit_margin(garch_series())
    assert fit.converged
    assert fit.beta == pytest.approx(0.85, abs=0.06) and fit.gamma == pytest.approx(0.1, abs=0.06)
    assert abs(z.mean()) < 0.05 and z.std() == pytest.approx(1.0, abs=0.05)


def test_simulate_paths_matches_recursion():
    f = MarginFit("X", mu=0.03, phi=0.1, omega=0.02, alpha=0.05, gamma=0.1, beta=0.85, eta=8.0, lam=-0.1,
                  scale=100.0, last_r=-1.0, last_eps=-1.2, last_sigma2=1.5, loglik=0.0, n_obs=1, converged=True)
    U = np.full((1, 2, 1), 0.5)
    from pf.models.margins import skewt_ppf
    z = skewt_ppf(np.array([0.5]), 8.0, -0.1)[0]
    s2 = f.next_sigma2()
    assert s2 == pytest.approx(0.02 + 0.15 * 1.44 + 0.85 * 1.5)
    e1 = np.sqrt(s2) * z
    r1 = 0.03 + 0.1 * -1.0 + e1
    s2b = 0.02 + (0.05 + 0.1 * (e1 < 0)) * e1 ** 2 + 0.85 * s2
    r2 = 0.03 + 0.1 * r1 + np.sqrt(s2b) * z
    assert simulate_paths([f], U)[0, 0] == pytest.approx((r1 + r2) / 100)


def test_residual_panel_drops_market_holidays():
    idx = pd.bdate_range("2024-01-01", periods=4)
    z = pd.DataFrame({"PKO.WA": [1.0, np.nan, 0.5, 0.2], "PZU.WA": [0.1, np.nan, 0.3, np.nan],
                      "AAPL": [0.2, 0.3, np.nan, 0.1]}, index=idx)
    p = residual_panel(z)
    assert list(p.index) == [idx[0], idx[3]] and p.loc[idx[3], "PZU.WA"] == 0.0


def test_lp_matches_brute_force_three_subfunds():
    rng = np.random.default_rng(0)
    S = 2000
    R = np.column_stack([rng.normal(0.010, 0.06, S), rng.normal(0.004, 0.02, S),
                         np.where(rng.random(S) < 0.05, -0.15, 0.012) + rng.normal(0, 0.01, S)])
    alpha, L = 0.975, 0.06
    sol = max_return_es(R, alpha, L)
    assert sol.feasible and sol.es <= L + 1e-6
    best = -np.inf
    grid = np.linspace(0, 1, 101)
    for w1, w2 in itertools.product(grid, grid):
        if w1 + w2 > 1 + 1e-12:
            continue
        w = np.array([w1, w2, 1 - w1 - w2])
        if empirical_es(-R @ w, alpha) <= L:
            best = max(best, R.mean(axis=0) @ w)
    assert sol.expected_return >= best - 1e-9
    assert sol.expected_return <= best + 5e-4  # siatka 0,01 zaniża optimum nieznacznie


def test_lp_infeasible_falls_back_to_min_es():
    rng = np.random.default_rng(1)
    R = rng.normal(0.01, 0.1, size=(1000, 3))
    sol = max_return_es(R, 0.975, 0.001)
    assert not sol.feasible and sol.weights.sum() == pytest.approx(1.0)
    for w in np.eye(3):
        assert sol.es <= empirical_es(-R @ w, 0.975) + 1e-9


def test_empirical_es():
    losses = np.arange(1, 41, dtype=float)  # 40 obserwacji, α = 0,975 → ogon 1 obserwacji
    assert empirical_es(losses, 0.975) == pytest.approx(40.0)
    assert empirical_es(losses, 0.95) == pytest.approx(39.5)


def test_lookthrough_equal_weights_among_available():
    sf = {"SF1": {"tickers": ["A", "B", "C"]}, "SF2": {"tickers": ["D"]}}
    A = lookthrough_matrix(sf, ["A", "C", "D"])
    np.testing.assert_allclose(A.loc["SF1"].to_numpy(), [0.5, 0.5, 0.0])
    np.testing.assert_allclose(subfund_returns(np.array([[0.1, 0.3, -0.2]]), A), [[0.2, -0.2]])


def test_rebalance_dates_and_no_look_ahead():
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    dates = rebalance_dates(idx, min_history=40)
    assert list(dates.strftime("%Y-%m-%d")) == ["2020-03-02", "2020-04-01", "2020-05-01", "2020-06-01"]
    r = pd.DataFrame({"X": np.arange(len(idx), dtype=float)}, index=idx)
    w = estimation_window(r, dates[0], 30)
    assert len(w) == 30 and w.index.max() < dates[0]
    real = realized_returns(r * 0 + 0.01, dates[0], dates[1])
    assert real["X"] == pytest.approx(np.expm1(0.01 * len(idx[(idx >= dates[0]) & (idx < dates[1])])))


def test_rebalance_one_smoke():
    """Pełny krok: GARCH → R-vine → scenariusze → LP na syntetycznym panelu 6 aktywów / 2 SF."""
    from pf import config
    from pipelines.monthly_rebalance import rebalance_one, summary_report

    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-01", periods=900)
    f = rng.standard_t(5, size=(900, 1)) * 0.01
    data = 0.8 * f + rng.standard_t(5, size=(900, 6)) * np.array([0.01, 0.012, 0.008, 0.004, 0.003, 0.005])
    returns = pd.DataFrame(data, index=idx, columns=["A.WA", "B.WA", "C", "D", "E", "F"])
    subfunds = {"SF1": {"name": "akcje", "tickers": ["A.WA", "B.WA", "C"]},
                "SF2": {"name": "obligacje", "tickers": ["D", "E", "F"]}}
    cfg = config.load("models")
    cfg = {**cfg, "rebalance": {**cfg["rebalance"], "window_days": 800, "min_obs": 600, "es_limit": 0.05}}
    t = idx[850]
    row, model, margins, R = rebalance_one(t, returns, cfg, subfunds, n_scen=500, jobs=1)
    assert model.end_date < t and len(model.assets) == 6
    assert R.shape == (500, 2)
    assert row["w_SF1"] + row["w_SF2"] == pytest.approx(1.0)
    assert row["es"] <= 0.05 + 1e-6 or not row["feasible"]
    weights = pd.DataFrame([row])
    realized = pd.DataFrame([{"date": t, "end": idx[870], "portfolio": 0.01, "r_SF1": 0.02, "r_SF2": 0.0}])
    assert "portfel (ES-LP)" in summary_report(weights, realized, subfunds, cfg["rebalance"])
