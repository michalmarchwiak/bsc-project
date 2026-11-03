"""Obligacje oszczędnościowe (SF10): indeksy wartości, margines deterministyczny, kara, koszty wykupu, backtest."""

import numpy as np
import pandas as pd
import pytest

from pf import config
from pf.backtest import engine
from pf.backtest.data import eligible_assets
from pf.backtest.realized import realize
from pf.data import savings_bonds as sb
from pf.models.dependence import fit_model
from pf.models.gof import pit_panel
from pf.models.margins import fit_margins, simulate_daily
from pf.opt.dependence_penalty import lower_tail_matrix, spearman_matrix
from pf.portfolio import deterministic_subfunds, deterministic_tickers
from pf.sim.scenarios import control_portfolios, simulate_adaptive
from tests.test_backtest import smoke_configs, synthetic_panel

EMISJE = """seria,rodzina,isin,sprzedaz_od,sprzedaz_do,miesiace_do_wykupu,okres_odsetkowy,podstawa_oprocentowania,kapitalizacja,oprocentowanie_1_okres_pct,marza_pct,marza_1_okres_pct,oplata_wykup_zl,oplata_wykup_rodzaj,cena_zamiany_zl,sprzedaz_mln_zl,w_tym_zamiana_mln_zl,lista_emisyjna,zrodlo
CAP0122,CAP,X,2020-01-01,2020-01-31,24,roczny,inflacja,true,5.0,1.0,,2.0,kwota z listu,,1,,1/2020,list
CAP0124,CAP,X,2022-01-03,2022-01-31,24,roczny,inflacja,true,3.0,1.0,,2.0,kwota z listu,99.9,1,,1/2022,list
CPN0221,CPN,X,2020-01-01,2020-01-31,12,miesięczny,stopa referencyjna NBP,false,6.0,0.0,,0.5,kwota z listu,,1,,2/2020,list
OTX0420,OTX,X,2020-01-01,2020-01-31,3,jednorazowo przy wykupie,stałe,false,2.0,,,,bez odsetek,,1,,3/2020,list
"""
OPROC = """seria,okres,jednostka_okresu,oprocentowanie_pct
CAP0122,1,rok,5.0
CAP0122,2,rok,7.0
CAP0124,1,rok,3.0
CPN0221,1,miesiąc,6.0
CPN0221,2,miesiąc,12.0
OTX0420,1,cały okres,2.0
"""


@pytest.fixture()
def src():
    return sb.read_issues(EMISJE), sb.read_rates(OPROC)


def test_capitalized_chain_value_roll_and_end(src):
    issues, rates = src
    V, F, hs = sb.chain(issues, rates, "CAP", "2020-01-01", "2024-06-30")
    assert [h.series for h in hs] == ["CAP0122", "CAP0124"]
    assert V["2020-01-01"] == pytest.approx(100.0)
    # pół roku pierwszego okresu: odsetki liniowe 5%·182/366
    assert V["2020-07-01"] == pytest.approx(100 * (1 + 0.05 * 182 / 366))
    # wykup 2022-01-01: 100·1,05·1,07; brak serii w sprzedaży do 2022-01-03 – środki bez odsetek
    redemption = 100 * 1.05 * 1.07
    assert V["2022-01-01"] == pytest.approx(redemption) and V["2022-01-02"] == pytest.approx(redemption)
    # zakup 2022-01-03 po cenie zamiany 99,9 zł
    assert V["2022-01-03"] == pytest.approx(redemption * 100 / 99.9)
    # drugi okres CAP0124 bez stopy w danych → stopa okresu 1 (3%); po wykupie 2024-01-03 brak serii – koniec
    assert V["2024-01-03"] == pytest.approx(redemption * 100 / 99.9 * 1.03 ** 2)
    assert V["2024-01-04":].isna().all()
    # opłata ≤ narosłe odsetki: tuż po zakupie f = a/(100 + a), później 2 zł / wartość
    assert F["2020-01-02"] == pytest.approx((V["2020-01-02"] - 100) / V["2020-01-02"])
    assert F["2020-12-31"] == pytest.approx(2.0 / V["2020-12-31"])
    assert F["2022-01-01"] == 0.0                                     # wykup w terminie


def test_coupon_and_ots_chains(src):
    issues, rates = src
    V, F, _ = sb.chain(issues, rates, "CPN", "2020-01-01", "2020-12-31")
    assert V["2020-02-01"] == pytest.approx(100 * (1 + 0.06 / 12))
    assert V["2020-03-01"] == pytest.approx(100 * (1 + 0.06 / 12) * (1 + 0.12 / 12))
    # opłata względem odsetek bieżącego okresu (kupon): w połowie lutego a = 100·1%·14/29
    a = 100 * 0.12 / 12 * 14 / 29
    assert F["2020-02-15"] == pytest.approx(min(0.5, a) / (100 + a))
    V, F, _ = sb.chain(issues, rates, "OTX", "2020-01-01", "2020-12-31")
    assert V["2020-04-01"] == pytest.approx(100 * (1 + 0.02 * 0.25))
    a = V["2020-03-01"] - 100
    assert F["2020-03-01"] == pytest.approx(a / (100 + a))         # bez odsetek przy wcześniejszym wykupie


def test_log_returns_on_sessions(src):
    issues, rates = src
    V, _, _ = sb.chain(issues, rates, "CAP", "2020-01-01", "2023-12-31")
    sessions = pd.bdate_range("2020-01-02", "2023-12-29")
    r = sb.log_returns_on(V.to_frame("CAP.SP"), sessions)["CAP.SP"]
    # pierwsza sesja liczona od dnia zakupu, suma stóp = ln V(ostatnia sesja) / V(zakup)
    assert r.iloc[0] == pytest.approx(np.log(V["2020-01-02"] / 100.0))
    assert r.sum() == pytest.approx(np.log(V[sessions[-1]] / 100.0))
    assert (r > 0).sum() > 0 and r.min() >= 0


def test_config_lists_sf10():
    sf = config.load("subfunds")["subfunds"]
    assert deterministic_subfunds(sf) == ["SF10"]
    assert set(deterministic_tickers(sf)) == set(sb.tickers())


def bond_panel(n=400, seed=11, young=60):
    r = synthetic_panel(n, seed)
    rate = np.where(np.arange(n) < n // 2, 0.0002, 0.0003)
    r["B1.SP"] = rate
    r["B2.SP"] = 0.00015
    r.loc[r.index[: n - young], "B2.SP"] = np.nan                    # młoda rodzina (``young`` sesji)
    return r


def test_eligibility_deterministic():
    r = bond_panel()
    det = ["B1.SP", "B2.SP"]
    assert "B2.SP" not in eligible_assets(r, 200)
    assert "B2.SP" in eligible_assets(r, 200, det, 21)
    r.iloc[-1, r.columns.get_loc("B1.SP")] = np.nan                  # indeks wygasł w ostatniej sesji
    assert "B1.SP" not in eligible_assets(r, 200, det, 21)


def test_deterministic_margins_and_simulation():
    r = bond_panel().iloc[:, [0, 1, 2, 9]]
    fits, z = fit_margins(r, jobs=1, deterministic=["B1.SP"], n_last=21)
    f = fits["B1.SP"]
    assert f.deterministic and f.mu == pytest.approx(100 * 0.0003) and list(z.columns) == list(r.columns[:3])
    assert np.isnan(f.pit_forward(np.array([0.0003, 0.0003]))).all()
    U = np.random.default_rng(0).random((50, 21, 4))
    daily = simulate_daily([fits[c] for c in r.columns], U)
    assert np.allclose(daily[:, :, 3], 0.0003) and daily[:, :, :3].std() > 0
    # PIT poza próbą: aktywa deterministyczne pomijane
    U_oos = pit_panel(fits, r.iloc[-5:])
    assert list(U_oos.columns) == list(r.columns[:3])


def test_simulate_adaptive_with_deterministic_asset():
    r = bond_panel().iloc[:, [0, 1, 2, 9]]
    fits, z = fit_margins(r, jobs=1, deterministic=["B1.SP"])
    from pf.models.rvine import pseudo_observations

    dep = fit_model("gauss", pseudo_observations(z.dropna().to_numpy()), list(z.columns))
    margins = [fits[c] for c in r.columns]
    sf = {"SF1": {"tickers": list(r.columns[:3])}, "SF10": {"tickers": ["B1.SP"]}}
    scfg = {"batch": 200, "n_min": 400, "n_max": 600, "rel_se": 0.05}
    s = simulate_adaptive(margins, dep, 21, control_portfolios(list(r.columns), sf), scfg, 0.975,
                          np.random.default_rng(1), np.random.default_rng(2), skip=1)
    assert np.allclose(s.R[:, 3], np.expm1(21 * 0.0003))
    assert s.stats.set_index("portfolio").loc["SF10", "rel_se"] == pytest.approx(0.0, abs=1e-12)
    # wspólne liczby losowe: scenariusze aktywów stochastycznych jak bez obligacji
    s0 = simulate_adaptive(margins[:3], dep, 21, control_portfolios(list(r.columns[:3]), sf), scfg, 0.975,
                           np.random.default_rng(1), np.random.default_rng(2), skip=1)
    np.testing.assert_allclose(s.R[:len(s0.R), :3], s0.R[:len(s.R)])


def test_dependence_measures_constant_column():
    rng = np.random.default_rng(0)
    R = np.column_stack([rng.normal(size=500), rng.normal(size=500), np.full(500, 0.002)])
    L = lower_tail_matrix(R, 0.05)
    assert np.allclose(L[2], 0.0) and np.allclose(L[:, 2], 0.0)
    S = spearman_matrix(R)
    assert np.isfinite(S).all() and S[2, 2] == 1.0 and S[0, 2] == 0.0
    assert np.linalg.eigvalsh(S).min() > -1e-12


def test_realize_redemption_cost():
    idx = pd.bdate_range("2021-01-01", periods=60)
    returns = pd.DataFrame({"A": 0.001, "B.SP": 0.0002}, index=idx)
    dates = pd.DatetimeIndex([idx[0], idx[20], idx[40]])
    w = pd.DataFrame([
        {"date": dates[0], "sid": "s", "ticker": "A", "weight": 0.5},
        {"date": dates[0], "sid": "s", "ticker": "B.SP", "weight": 0.5},
        {"date": dates[1], "sid": "s", "ticker": "A", "weight": 0.8},
        {"date": dates[1], "sid": "s", "ticker": "B.SP", "weight": 0.2},
        {"date": dates[2], "sid": "s", "ticker": "A", "weight": 1.0}])
    lags = pd.Series({"s": 1})
    fee = pd.DataFrame({"B.SP": 0.01}, index=idx)
    _, p0 = realize(returns, w, lags, dates, 0.001)
    _, p1 = realize(returns, w, lags, dates, 0.001, sell_cost=fee)
    assert p1["redemption_cost"].iloc[0] == 0.0                       # zakup początkowy bez opłaty
    sold = p0["traded"].iloc[1] / 2                                   # sprzedaż B.SP = połowa obrotu
    assert p1["redemption_cost"].iloc[1] == pytest.approx(0.01 * sold, rel=1e-9)
    assert p1["cost"].iloc[1] == pytest.approx(p0["cost"].iloc[1] + 0.01 * sold)
    assert p1["r_net"].iloc[1] < p0["r_net"].iloc[1]


def test_backtest_smoke_with_sf10(tmp_path):
    cfg, bcfg, pcfg, sf = smoke_configs()
    bcfg = {**bcfg, "sensitivity": {**bcfg["sensitivity"], "enabled": False},
            "grid": {**bcfg["grid"], "models": ["gauss"]}}
    sf = {**sf, "SF10": {"name": "Obligacje", "deterministic": True, "tickers": ["B1.SP", "B2.SP"]}}
    r = bond_panel(young=250)
    from pipelines import backtest as bt

    dates = bt.all_dates(r, cfg, bcfg, sf)[:2]
    for i, t in enumerate(dates):
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        s = engine.run_date(t, nxt, r, cfg, bcfg, pcfg, sf, tmp_path, tmp_path / "cache", latest=nxt is None)
        assert s["n_deterministic"] == 2 and "SF10" in s["subfunds"]
    out = bt.consolidate(tmp_path, r, cfg, bcfg, pcfg, sf, rf=pd.Series(0.03, index=r.index))
    assert "SF10" in set(out["periods_sf"]["subfund"]) and (out["periods"]["redemption_cost"] == 0).all()
    dm = engine.fit_margins_step(dates[0], r, cfg, cache_dir=tmp_path / "cache", deterministic=["B1.SP", "B2.SP"])
    assert dm.cached["margins"] and "B1.SP" in dm.assets and "B1.SP" not in dm.stochastic
    W = pd.concat([pd.read_parquet(p) for p in (tmp_path / "dates").glob("*/weights.parquet")])
    w10 = W[(W["level"] == "W") & (W["subfund"] == "SF10") & (W["strategy"] == "ce") & (W["profile"] == "P1")]
    assert (w10["weight"] > 0).all()                                  # stopa pewna – udział w portfelu P1
    fc = pd.concat([pd.read_parquet(p) for p in (tmp_path / "dates").glob("*/forecasts.parquet")])
    assert np.isfinite(fc["es"]).all()
