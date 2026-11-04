"""Walidacja silnika wyceny nakładki: replikacja indeksów Cboe BXM i PPUT na S&P 500.

Konstrukcja indeksów odpowiada założeniom modułu: miesięczne opcje europejskie bliskie ATM trzymane do
wygaśnięcia. BXM – S&P 500 (total return) ze sprzedanym callem ATM (strike najbliższy powyżej indeksu),
PPUT – S&P 500 (total return) z kupionym putem 95% (strike najbliższy poniżej 0,95·S). Rolowanie w trzeci
piątek miesiąca (pierwsza sesja ≥ trzeciego piątku, ``pf.options.pput.roll_dates``); na każdym rolowaniu cały
majątek dzielony jest na n jednostek (indeks ± opcja), opcja jest rozliczana w kolejnym rolowaniu:

    BXM:  V₁/V₀ = (S₀·TR₁/TR₀ − (S₁ − K)⁺) / (S₀ − C₀),     PPUT: V₁/V₀ = (S₀·TR₁/TR₀ + (K − S₁)⁺) / (S₀ + P₀).

Wycena silnikiem nakładki: σ = atm_adj · σ_VIX(τ) (interpolacja VIX9D–VIX, ``pf.options.implied_vol``)
+ β ln(K/F), τ = n_sesji/252, r – ciągła stopa z bonów 13-tygodniowych (``pf.data.usd_rate.continuous_rate``),
q – stopa dywidendy z różnicy stóp indeksu total return i cenowego (opcje na indeks cenowy).
Bez kosztów transakcyjnych (indeksy Cboe ich nie zawierają) i bez przeszacowania w trakcie miesiąca:
tracking error liczony jest na stopach między rolowaniami, TE = σ(r_rep − r_off)·√12.
Różnice metodologiczne: Cboe rozlicza opcje kursem SOQ (otwarcie) i kupuje/sprzedaje po VWAP, a nie po zamknięciu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pf.data.usd_rate import continuous_rate
from pf.options import bs
from pf.options.implied_vol import index_vol, strike_vol
from pf.options.pput import roll_dates, trailing_div_yield


def strike(S: float, moneyness: float, step: float, rule: str) -> float:
    x = moneyness * S / step
    return float(step * (np.ceil(x - 1e-12) if rule == "ceil" else np.floor(x + 1e-12)))


def replicate_monthly(spx: pd.Series, spx_tr: pd.Series, vol: pd.DataFrame, rate_y: pd.Series, spec: dict,
                      icfg: dict, trading_days: int = 252, beta: float | None = None) -> pd.DataFrame:
    """Okresy rolowania (d0, d1) z wyceną opcji i stopą repliki."""
    df = pd.concat({"S": spx, "TR": spx_tr}, axis=1).dropna()
    q = trailing_div_yield(df["S"], df["TR"]).reindex(df.index).ffill().fillna(0.0)
    y = rate_y.reindex(df.index.union(rate_y.index)).ffill().reindex(df.index)
    lv = vol.reindex(vol.index.union(df.index)).ffill().reindex(df.index)
    rolls = roll_dates(df.index)
    beta = float(icfg.get("beta", 0.0) if beta is None else beta)
    call = spec["type"] == "call"
    rows = []
    for d0, d1 in zip(rolls[:-1], rolls[1:]):
        n = int(((df.index > d0) & (df.index <= d1)).sum())
        tau, days = n / trading_days, (d1 - d0).days
        S0, S1, TR0, TR1 = df.at[d0, "S"], df.at[d1, "S"], df.at[d0, "TR"], df.at[d1, "TR"]
        levels = {c: float(lv.at[d0, c]) for c in icfg["vix_tenors_days"] if c in lv.columns}
        sig_atm = index_vol("VIX", levels, days, icfg["vix_tenors_days"], icfg.get("atm_adj", 1.0))
        r = float(continuous_rate(y.at[d0] if np.isfinite(y.at[d0]) else 0.0, days, tau, 360))
        K = strike(S0, float(spec["moneyness"]), float(spec["strike_step"]), spec["strike_rule"])
        F = float(bs.forward(S0, tau, r, q.at[d0]))
        sigma = float(strike_vol(sig_atm, K, F, beta, icfg["sigma_min"]))
        prem = float(bs.price(S0, K, tau, r, q.at[d0], sigma, call))
        pay = float(bs.payoff(S1, K, call))
        if spec.get("side", "long") == "short":
            ret = (S0 * TR1 / TR0 - pay) / (S0 - prem) - 1.0
        else:
            ret = (S0 * TR1 / TR0 + pay) / (S0 + prem) - 1.0
        rows.append({"d0": d0, "d1": d1, "n_sessions": n, "S0": S0, "S1": S1, "K": K, "sigma": sigma, "r": r,
                     "q": float(q.at[d0]), "premium": prem, "premium_rel": prem / S0, "payoff": pay, "r_replica": ret})
    return pd.DataFrame(rows)


def tracking(rep: pd.DataFrame, official: pd.Series, windows: dict[str, list[str]]) -> pd.DataFrame:
    """TE i różnice stóp repliki względem indeksu (cała próba, lata kalendarzowe, okna stresowe)."""
    off = official.dropna()
    lv = off.reindex(off.index.union(rep["d1"])).ffill()
    r = rep.assign(r_official=lv.reindex(rep["d1"]).to_numpy() / lv.reindex(rep["d0"]).to_numpy() - 1.0)
    r = r[(r["d0"] >= off.index.min()) & r["r_official"].notna()]
    rows = []

    def add(label: str, g: pd.DataFrame) -> None:
        if len(g) < 2:
            return
        diff = g["r_replica"] - g["r_official"]
        rows.append({"period": label, "n_months": len(g), "te_ann_pct": 100 * diff.std(ddof=1) * np.sqrt(12),
                     "mean_diff_ann_pct": 100 * diff.mean() * 12,
                     "ret_replica_ann_pct": 100 * (np.prod(1 + g["r_replica"]) ** (12 / len(g)) - 1),
                     "ret_official_ann_pct": 100 * (np.prod(1 + g["r_official"]) ** (12 / len(g)) - 1),
                     "corr": float(g["r_replica"].corr(g["r_official"]))})

    add("all", r)
    for yr, g in r.groupby(r["d0"].dt.year):
        add(str(yr), g)
    for name, (a, b) in windows.items():
        add(f"stress_{name}", r[(r["d0"] >= pd.Timestamp(a)) & (r["d0"] <= pd.Timestamp(b))])
    return pd.DataFrame(rows)
