"""Rejestr hipotetycznych opcji europejskich bliskich ATM: wycena w t_j i rozliczenie w t_{j+1}.

Dla każdej daty t_j, sub-funduszu, tickera, typu (put/call) i moneyness m z siatki K/S_t:

    K = m·S_t,  F = S_t e^{(r − q)τ},  q = 0 (ceny total return),
    σ = σ_imp(K) = max(σ_imp + β ln(K/F), σ_min),
    premia (BSM), Δ, Γ, ν (``pf.options.bs``) w walucie notowania,
    wypłata w t_{j+1}: max(K − S_{t+1}, 0) (put), max(S_{t+1} − K, 0) (call),
    P&L długiej pozycji w PLN (bez kosztu κ): payoff·fx_{t+1} − premia·fx_t.

Stopa r – ciągła na okres opcji (``pf.data.usd_rate.continuous_rate``): PLN z WIBOR 1M (ACT/365),
USD z bonów 13-tygodniowych (ACT/360). Premia opcji na ticker w USD jest przeliczana po kursie z t_j,
wypłata po kursie z t_{j+1}.

Pasmo: |ln(K/F)| ≤ ``band`` (``in_band``); wiersze spoza pasma pozostają w rejestrze, ale nie są
wybierane przez polityki. Rejestr liczony jest dla całej siatki niezależnie od polityki.
Opcje są hipotetyczne: dla większości tickerów (spółki GPW, ETF-y) nie istnieją notowane opcje miesięczne.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pf.data.usd_rate import continuous_rate
from pf.options import bs
from pf.options.implied_vol import strike_vol
from pf.options.market import currency

LEDGER_COLUMNS = ["date", "t", "t_next", "subfund", "ticker", "currency", "type", "moneyness", "K", "S_t", "F",
                  "tau", "n_sessions", "days", "r", "q", "sigma_imp", "sigma_atm", "sigma_hat", "iv_source",
                  "log_kf", "in_band", "premium", "premium_rel", "delta", "gamma", "vega", "S_next", "payoff",
                  "payoff_rel", "fx_t", "fx_next", "pnl_pln", "pnl_rel_pln"]


def period_rates(tickers, rate_pln: float, rate_usd: float, days: float, tau: float, rcfg: dict) -> pd.Series:
    """Stopa ciągła r_i na okres opcji według waluty tickera."""
    out = {}
    for t in tickers:
        if currency(t) == "PLN":
            out[t] = float(continuous_rate(rate_pln, days, tau, rcfg.get("pln_basis", 365)))
        else:
            out[t] = float(continuous_rate(rate_usd, days, tau, rcfg.get("usd_basis", 360)))
    return pd.Series(out, dtype=float)


def build(period: dict, tickers: list[str], subfund_of: dict[str, str], S_t: pd.Series, S_next: pd.Series,
          fx_t: pd.Series, fx_next: pd.Series, r: pd.Series, sig: pd.DataFrame, ocfg: dict,
          beta: float | None = None) -> pd.DataFrame:
    """Wiersze rejestru dla jednej daty (``period`` – wiersz ``pf.options.calendar.overlay_periods``)."""
    grid = np.asarray(ocfg["moneyness"], dtype=float)
    icfg = ocfg["implied_vol"]
    beta = float(icfg.get("beta", 0.0) if beta is None else beta)
    tau = float(period["tau"])
    n_t, n_m = len(tickers), len(grid)
    tk = np.repeat(np.array(tickers, dtype=object), 2 * n_m)
    is_call = np.tile(np.repeat([False, True], n_m), n_t)
    m = np.tile(np.tile(grid, 2), n_t)
    S0 = np.repeat(S_t.reindex(tickers).to_numpy(float), 2 * n_m)
    S1 = np.repeat(S_next.reindex(tickers).to_numpy(float), 2 * n_m)
    rr = np.repeat(r.reindex(tickers).to_numpy(float), 2 * n_m)
    sig_atm = np.repeat(sig["sigma_imp"].reindex(tickers).to_numpy(float), 2 * n_m)
    q = np.zeros_like(S0)
    K = m * S0
    F = bs.forward(S0, tau, rr, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_kf = np.log(K / F)
        sigma = strike_vol(sig_atm, K, F, beta, icfg["sigma_min"])
        prem = bs.price(S0, K, tau, rr, q, sigma, is_call)
        g = bs.greeks(S0, K, tau, rr, q, sigma, is_call)
        pay = np.where(np.isfinite(S1), bs.payoff(S1, K, is_call), np.nan)
    fx0 = np.repeat(fx_t.reindex(tickers).to_numpy(float), 2 * n_m)
    fx1 = np.repeat(fx_next.reindex(tickers).to_numpy(float), 2 * n_m)
    pnl = pay * fx1 - prem * fx0
    out = pd.DataFrame({
        "date": period["date"], "t": period["open"], "t_next": period["close"],
        "subfund": [subfund_of[t] for t in tk], "ticker": tk, "currency": [currency(t) for t in tk],
        "type": np.where(is_call, "call", "put"), "moneyness": m, "K": K, "S_t": S0, "F": F, "tau": tau,
        "n_sessions": int(period["n_sessions"]), "days": int(period["days"]), "r": rr, "q": q,
        "sigma_imp": sigma, "sigma_atm": sig_atm,
        "sigma_hat": np.repeat(sig["sigma_hat"].reindex(tickers).to_numpy(float), 2 * n_m),
        "iv_source": np.repeat(sig["iv_source"].reindex(tickers).to_numpy(object), 2 * n_m),
        "log_kf": log_kf, "in_band": np.abs(log_kf) <= float(ocfg["band"]) + 1e-12,
        "premium": prem, "premium_rel": prem / S0, "delta": g["delta"], "gamma": g["gamma"], "vega": g["vega"],
        "S_next": S1, "payoff": pay, "payoff_rel": pay / S0, "fx_t": fx0, "fx_next": fx1,
        "pnl_pln": pnl, "pnl_rel_pln": pnl / (S0 * fx0)})
    out["in_band"] &= np.isfinite(out["premium"].to_numpy())
    return out[LEDGER_COLUMNS]


def reprice(ledger: pd.DataFrame, sigma_atm: np.ndarray | None = None, beta: float = 0.0,
            sigma_min: float = 0.01) -> pd.DataFrame:
    """Ponowna wycena wierszy rejestru przy zmienionym σ_ATM lub β (analiza wrażliwości)."""
    out = ledger.copy()
    sig = out["sigma_atm"].to_numpy(float) if sigma_atm is None else np.asarray(sigma_atm, dtype=float)
    S, K, tau, r, q = (out[c].to_numpy(float) for c in ("S_t", "K", "tau", "r", "q"))
    is_call = (out["type"] == "call").to_numpy()
    sigma = strike_vol(sig, K, out["F"].to_numpy(float), beta, sigma_min)
    prem = bs.price(S, K, tau, r, q, sigma, is_call)
    g = bs.greeks(S, K, tau, r, q, sigma, is_call)
    out["sigma_atm"], out["sigma_imp"] = sig, sigma
    out["premium"], out["premium_rel"] = prem, prem / S
    out["delta"], out["gamma"], out["vega"] = g["delta"], g["gamma"], g["vega"]
    out["pnl_pln"] = out["payoff"] * out["fx_next"] - prem * out["fx_t"]
    out["pnl_rel_pln"] = out["pnl_pln"] / (S * out["fx_t"])
    out["in_band"] = out["in_band"] & np.isfinite(prem)
    return out
