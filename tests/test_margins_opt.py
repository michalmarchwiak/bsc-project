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


def test_optimize_subfund_constraints():
    from pipelines.monthly_rebalance import optimize_subfund

    rng = np.random.default_rng(4)
    S = 3000
    mu = np.array([0.02, 0.01, 0.005, 0.003, 0.0])
    sd = np.array([0.08, 0.05, 0.03, 0.02, 0.01])
    R = pd.DataFrame(rng.standard_t(5, size=(S, 5)) * sd * 0.8 + mu, columns=list("ABCDE"))
    w, st = optimize_subfund(R, 0.975, 0.30)
    assert w.sum() == pytest.approx(1.0) and (w <= 0.30 + 1e-9).all() and (w >= -1e-12).all()
    assert st["es"] <= st["es_ew"] + 1e-8
    assert st["exp_return"] >= st["exp_return_ew"] - 1e-12
    # mniej aktywów niż 1/limit → limit 1/n wymusza równe wagi
    w3, st3 = optimize_subfund(R[list("ABC")], 0.975, 0.30)
    np.testing.assert_allclose(w3.to_numpy(), 1 / 3, atol=1e-8)
    assert st3["cap"] == pytest.approx(1 / 3)
    w1, _ = optimize_subfund(R[["A"]], 0.975, 0.30)
    assert w1.iloc[0] == 1.0


def test_turnover_uses_drifted_weights():
    from pipelines.monthly_rebalance import drifted_weights, turnover

    d = pd.to_datetime(["2020-01-02", "2020-02-03", "2020-03-02"])
    # okres 1: A +10%, B −10%; okres 2: brak zmian cen
    returns = pd.DataFrame({"A": [np.log1p(0.1), 0.0, 0.0], "B": [np.log1p(-0.1), 0.0, 0.0]},
                           index=pd.to_datetime(["2020-01-15", "2020-02-14", "2020-03-02"]))
    comp = pd.DataFrame({"date": [d[0], d[0], d[1], d[1], d[2]], "subfund": "SF1",
                         "ticker": ["A", "B", "A", "B", "C"],
                         "weight": [0.5, 0.5, 0.5, 0.5, 1.0], "weight_ew": [0.5, 0.5, 0.5, 0.5, 1.0]})
    np.testing.assert_allclose(drifted_weights(pd.Series([0.5, 0.5], index=["A", "B"]),
                                               pd.Series({"A": 0.1, "B": -0.1})), [0.55, 0.45])
    to = turnover(comp, returns)
    # niezmienione wagi docelowe 50/50, ale po dryfie 55/45 → obrót 5%
    assert to["turnover_opt"].tolist() == pytest.approx([0.05, 1.0])
    assert to["turnover_ew"].tolist() == pytest.approx([0.05, 1.0])  # wymiana A, B → C: obrót 100%
    assert list(to["date"]) == [d[1], d[2]]


def test_apply_costs():
    from pipelines.monthly_rebalance import apply_costs

    d = pd.to_datetime(["2020-01-02", "2020-02-03"])
    realized = pd.DataFrame({"date": d, "end": d + pd.Timedelta(days=28), "subfund": "SF1",
                             "r_opt": [0.02, -0.01], "r_ew": [0.01, 0.0]})
    to = pd.DataFrame({"date": [d[1]], "subfund": ["SF1"], "turnover_opt": [0.25], "turnover_ew": [0.05]})
    out = apply_costs(realized, to, 0.001)
    # zakup początkowy: koszt c; później c·Σ|Δw| = 2c·TO
    np.testing.assert_allclose(out["cost_opt"], [0.001, 0.0005])
    np.testing.assert_allclose(out["cost_ew"], [0.001, 0.0001])
    np.testing.assert_allclose(out["r_opt_net"], [(1 - 0.001) * 1.02 - 1, (1 - 0.0005) * 0.99 - 1])
    zero = apply_costs(realized, to, 0.0)
    np.testing.assert_allclose(zero["r_opt_net"], realized["r_opt"])


def test_rebalance_one_smoke():
    """Pełny krok: GARCH → R-vine → scenariusze → LP w każdym sub-funduszu (6 aktywów / 2 SF)."""
    from pf import config
    from pipelines.monthly_rebalance import apply_costs, realized_table, rebalance_one, summary_report, turnover

    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-01", periods=900)
    f = rng.standard_t(5, size=(900, 1)) * 0.01
    data = 0.8 * f + rng.standard_t(5, size=(900, 8)) * np.array([0.01, 0.012, 0.008, 0.009, 0.004, 0.003,
                                                                   0.005, 0.006])
    returns = pd.DataFrame(data, index=idx, columns=["A.WA", "B.WA", "C", "D", "E", "F", "G", "H"])
    subfunds = {"SF1": {"name": "akcje", "tickers": ["A.WA", "B.WA", "C", "D"]},
                "SF2": {"name": "obligacje", "tickers": ["E", "F", "G", "H"]}}
    cfg = config.load("models")
    cfg = {**cfg, "rebalance": {**cfg["rebalance"], "window_days": 800, "min_obs": 600}}
    t1, t2 = idx[820], idx[850]
    comp1, stats1, row, model, sf_scen = rebalance_one(t1, returns, cfg, subfunds, n_scen=500, jobs=1)
    assert model.end_date < t1 and len(model.assets) == 8
    assert sf_scen.shape == (500, 2)
    assert comp1.groupby("subfund")["weight"].sum().to_numpy() == pytest.approx([1.0, 1.0])
    assert (comp1["weight"] <= 0.30 + 1e-9).all()
    assert (stats1["es"] <= stats1["es_ew"] + 1e-8).all()
    # ten sam model wczytany ponownie daje te same scenariusze (ziarno zależne od daty)
    comp1b, *_ = rebalance_one(t1, returns, cfg, subfunds, n_scen=500, jobs=1, model=model)
    np.testing.assert_allclose(comp1b["weight"], comp1["weight"])
    comp2, stats2, *_ = rebalance_one(t2, returns, cfg, subfunds, n_scen=500, jobs=1, model=model)
    comp, stats = pd.concat([comp1, comp2]), pd.concat([stats1, stats2])
    realized = realized_table(returns, comp, stats)
    assert len(realized) == 2 and set(realized["subfund"]) == {"SF1", "SF2"}
    to = turnover(comp, returns)
    assert len(to) == 2 and (to[["turnover_opt", "turnover_ew"]] >= 0).all().all()
    report = summary_report(comp, stats, apply_costs(realized, to, 0.001), to, subfunds,
                            {**cfg["rebalance"], "transaction_cost": 0.001})
    assert "SF1 akcje" in report and "Po kosztach transakcyjnych (0.10%" in report


def test_simulation_winsorizes_innovations():
    """|ε| ≤ k·max|ε̂|: przy η bliskim 2 i silnym ARCH skrajne kwantyle nie powodują eksplozji ścieżek."""
    f = MarginFit("X", mu=0.1, phi=0.0, omega=1.5, alpha=0.15, gamma=0.18, beta=0.58, eta=2.2, lam=0.0,
                  scale=100.0, last_r=0.0, last_eps=0.0, last_sigma2=4.0, loglik=0.0, n_obs=1, converged=True,
                  eps_absmax=10.0)
    U = np.full((1, 21, 1), 1 - 1e-12)
    capped = simulate_paths([f], U, eps_cap=2.0)[0, 0]
    assert capped <= 21 * (0.1 + 20.0) / 100 + 1e-9                  # każdy dzień ≤ μ + 2·max|ε̂|
    assert simulate_paths([f], U, eps_cap=None)[0, 0] > 100 * capped   # bez ograniczenia – eksplozja
