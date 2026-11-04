"""Nakładka opcyjna – Etap 3: rejestr opcji (wycena, pasmo, wypłaty) i NAV sub-funduszu."""

import numpy as np
import pandas as pd
import pytest

from pf.options import bs, ledger, nav
from tests.options_synth import overlay_config

OCFG = overlay_config()
TICKERS = ["AAA.WA", "CCC"]
PERIOD = {"date": pd.Timestamp("2020-03-02"), "open": pd.Timestamp("2020-03-02"), "close": pd.Timestamp("2020-04-01"),
          "tau": 22 / 252, "n_sessions": 22, "days": 30}


def build(S_next=(90.0, 210.0), band=None, beta=None):
    cfg = dict(OCFG, band=band if band is not None else OCFG["band"])
    S_t = pd.Series([100.0, 200.0], index=TICKERS)
    sig = pd.DataFrame({"sigma_imp": [0.3, 0.2], "sigma_hat": [0.25, 0.18], "iv_source": ["proxy", "proxy"]},
                       index=TICKERS)
    r = ledger.period_rates(TICKERS, 0.06, 0.04, 30, PERIOD["tau"], OCFG["rates"])
    return ledger.build(PERIOD, TICKERS, {"AAA.WA": "SF1", "CCC": "SF2"}, S_t, pd.Series(S_next, index=TICKERS),
                        pd.Series([1.0, 4.0], index=TICKERS), pd.Series([1.0, 4.2], index=TICKERS), r, sig, cfg, beta)


def test_ledger_pricing_and_parity():
    L = build()
    assert len(L) == 2 * 2 * len(OCFG["moneyness"]) and list(L.columns) == ledger.LEDGER_COLUMNS
    np.testing.assert_allclose(L["K"], L["moneyness"] * L["S_t"])
    wide = L.pivot_table(index=["ticker", "moneyness"], columns="type", values="premium")
    meta = L.drop_duplicates(["ticker", "moneyness"]).set_index(["ticker", "moneyness"])
    np.testing.assert_allclose(wide["call"] - wide["put"],
                               meta["S_t"] - meta["K"] * np.exp(-meta["r"] * meta["tau"]), atol=1e-10)
    # stopa ciągła PLN: czynnik dyskontowy rynku pieniężnego
    r_pln = L.loc[L["ticker"] == "AAA.WA", "r"].iloc[0]
    assert np.exp(-r_pln * PERIOD["tau"]) == pytest.approx(1 / (1 + 0.06 * 30 / 365))
    assert (L.loc[L["type"] == "put", "delta"] < 0).all() and (L["gamma"] > 0).all() and (L["vega"] > 0).all()


def test_band_rejects_outside_strikes():
    L05 = build(band=0.05)
    put95 = L05[(L05["type"] == "put") & (L05["moneyness"] == 0.95)]
    assert (~put95["in_band"]).all()                    # ln(0,95·S/F) = −0,0513 − rτ < −0,05
    assert L05.loc[L05["moneyness"] != 0.95, "in_band"].all()
    L06 = build()
    assert L06["in_band"].all()
    assert (np.abs(L06["log_kf"]) <= 0.06).all()


@pytest.mark.parametrize("S1", [70.0, 95.0, 97.5, 100.0, 102.0, 110.0, 150.0])
def test_payoffs_on_grid(S1):
    L = build(S_next=(S1, 2 * S1))
    a = L[L["ticker"] == "AAA.WA"]
    exp = np.where(a["type"] == "call", np.maximum(S1 - a["K"], 0), np.maximum(a["K"] - S1, 0))
    np.testing.assert_allclose(a["payoff"], exp)
    c = L[L["ticker"] == "CCC"]
    np.testing.assert_allclose(c["pnl_pln"], c["payoff"] * 4.2 - c["premium"] * 4.0)
    # Π na jednostkę S_t: nogi z nav.overlay_payoff
    legs = [(0.975, 1.0, False), (1.05, 1.0, True)]
    g = S1 / 100.0
    assert float(nav.overlay_payoff(g, legs)) == pytest.approx(max(0.975 - g, 0) - max(g - 1.05, 0))


def test_reprice_identity_and_skew():
    L = build()
    R = ledger.reprice(L)
    np.testing.assert_allclose(R["premium"], L["premium"], rtol=1e-12)
    B = build(beta=-0.5)
    Rb = ledger.reprice(L, beta=-0.5)
    np.testing.assert_allclose(Rb["premium"], B["premium"], rtol=1e-12)
    puts = (L["type"] == "put") & (L["moneyness"] < 1)
    assert (B.loc[puts, "premium"] > L.loc[puts, "premium"]).all()   # β < 0: droższe puty OTM


# --------------------------------------------------------------------------- NAV

W = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
G = pd.Series([1.05, 0.92, 1.01], index=W.index)
f = pd.Series([1.0, 1.02, 0.98], index=W.index)
g = G / f
POS = pd.DataFrame({"ticker": ["A", "B", "B", "C"], "type": ["put", "put", "call", "call"],
                    "moneyness": [0.975, 1.0, 1.05, 1.025], "h": [1.0, 0.5, 1.0, 0.7],
                    "premium_rel": [0.02, 0.03, 0.01, 0.015], "delta": [-0.35, -0.5, 0.3, 0.4]})


def test_nav_zero_overlay_equals_base():
    for pos in (None, nav.empty_positions(), POS.assign(h=0.0)):
        out = nav.period_return(W, G, g, f, pos, 0.03)
        assert out["r_ov"] == pytest.approx(out["r_base"], abs=1e-15)
        assert out["r_base"] == pytest.approx(float(W @ G) - 1.0)


def test_nav_matches_unit_formula_in_pln():
    """V_{j+1} = Σ n_i [S_{t+1} + h^P (K^P − S_{t+1})⁺ − h^C (S_{t+1} − K^C)⁺] (PLN) z jednostkami n_i."""
    kappa, V = 0.03, 1000.0
    S0 = pd.Series([50.0, 120.0, 30.0], index=W.index)
    fx0 = pd.Series([1.0, 4.0, 4.0], index=W.index)
    S1, fx1 = S0 * g, fx0 * f
    D = 1 + np.array([0.02 * 1.03, 0.5 * 0.03 * 1.03 - 0.01 * 0.97, -0.7 * 0.015 * 0.97])
    Vp = V / float(W @ D)
    n = W * Vp / (S0 * fx0)
    V1 = 0.0
    for t in W.index:
        p = POS[POS["ticker"] == t]
        val = S1[t]
        for r in p.itertuples():
            K = r.moneyness * S0[t]
            val += r.h * (max(S1[t] - K, 0) * -1 if r.type == "call" else max(K - S1[t], 0))
        V1 += n[t] * val * fx1[t]
    out = nav.period_return(W, G, g, f, POS, kappa)
    assert out["r_ov"] == pytest.approx(V1 / V - 1, rel=1e-12)
    assert out["denominator"] == pytest.approx(float(W @ D))


def test_self_financing():
    est = nav.establishment(1.0, W, POS, 0.03)
    assert est["total"] == pytest.approx(1.0, abs=1e-14)
    assert est["stock"] == pytest.approx(est["V_prime"])
    assert nav.establishment(1.0, W, None, 0.03)["V_prime"] == 1.0


def test_scenario_returns_match_period_return():
    rng = np.random.default_rng(0)
    gS = np.exp(rng.normal(0, 0.06, size=(50, 3)))
    fS = np.exp(rng.normal(0, 0.02, size=(50, 3)))
    sc = nav.scenario_returns(W.to_numpy(), gS * fS, gS, fS, list(W.index), POS, 0.03)
    for s in (0, 17, 49):
        one = nav.period_return(W, pd.Series(gS[s] * fS[s], index=W.index), pd.Series(gS[s], index=W.index),
                                pd.Series(fS[s], index=W.index), POS, 0.03)["r_ov"]
        assert sc[s] == pytest.approx(one, rel=1e-12)


def test_delta_exposure_and_kind():
    e = nav.delta_exposure(W, POS, 0.03)
    D = float(W @ (1 + np.array([0.02 * 1.03, 0.5 * 0.03 * 1.03 - 0.01 * 0.97, -0.7 * 0.015 * 0.97])))
    assert e["A"] == pytest.approx(0.5 / D * (1 - 0.35))
    assert e["B"] == pytest.approx(0.3 / D * (1 + 0.5 * -0.5 - 0.3))
    assert e["C"] == pytest.approx(0.2 / D * (1 - 0.7 * 0.4))
    assert nav.overlay_kind(POS) == "collar"
    assert nav.overlay_kind(POS[POS["type"] == "put"]) == "put"
    assert nav.overlay_kind(POS[POS["type"] == "call"]) == "covered_call"
    assert nav.overlay_kind(None) == "none"
    np.testing.assert_allclose(nav.nav_from_returns([0.1, -0.5]), [1.0, 1.1, 0.55])


def test_bs_put_call_parity_with_q():
    S, K, T, r, q, s = 100.0, np.array([95.0, 100.0, 105.0]), 21 / 252, 0.05, 0.02, 0.25
    c, p = bs.price(S, K, T, r, q, s, True), bs.price(S, K, T, r, q, s, False)
    np.testing.assert_allclose(c - p, S * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-12)


def test_period_arrays_equal_pandas_version():
    idx = POS["ticker"].map({t: i for i, t in enumerate(W.index)}).to_numpy()
    out = nav.period_arrays(W.to_numpy(), G.to_numpy(), g.to_numpy(), f.to_numpy(), idx,
                            (POS["type"] == "call").to_numpy(), POS["moneyness"].to_numpy(), POS["h"].to_numpy(),
                            POS["premium_rel"].to_numpy(), POS["delta"].to_numpy(), 0.03)
    ref = nav.period_return(W, G, g, f, POS, 0.03)
    for k, v in ref.items():
        assert out[k] == pytest.approx(v, rel=1e-12, abs=1e-15)
    np.testing.assert_allclose(out["e"], nav.delta_exposure(W, POS, 0.03).to_numpy(), rtol=1e-12)
    assert out["kind"] == nav.overlay_kind(POS)
