"""Syntetyczne dane rynkowe dla testów nakładki opcyjnej (bez sieci i bez plików z repozytorium)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from pf import config
from pf.options.market import MarketData

PLN_TICKERS = ["AAA.WA", "BBB.WA"]
USD_TICKERS = ["CCC", "DDD", "GLD", "EEE"]
SUBFUNDS = {
    "SF1": {"name": "akcje PL", "tickers": PLN_TICKERS},
    "SF2": {"name": "akcje US", "tickers": ["CCC", "DDD"]},
    "SF8": {"name": "złoto", "tickers": ["GLD", "EEE"]},
    "SF10": {"name": "obligacje", "deterministic": True, "tickers": ["OTS.SP"]},
}


def synthetic_market(n: int = 760, seed: int = 7, start: str = "2010-01-04") -> MarketData:
    """Dzienne log-stopy z zależnością (czynnik wspólny) i zmiennością warunkową; kurs, VIX, GVZ, S&P 500."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    tickers = PLN_TICKERS + USD_TICKERS
    common = rng.standard_t(5, size=n) * 0.007
    vol = np.exp(0.3 * np.sin(np.arange(n) / 60.0))
    loc = {}
    for j, t in enumerate(tickers):
        eps = rng.standard_t(6, size=n) * 0.011
        loc[t] = 0.0002 + vol * (0.8 * common + eps) * (0.6 + 0.15 * j)
    r_local = pd.DataFrame(loc, index=idx)
    # święta GPW: brak notowań tickerów PLN w części dni
    holidays = idx[rng.random(n) < 0.02]
    r_local.loc[holidays, PLN_TICKERS] = np.nan
    r_local.iloc[:40, r_local.columns.get_loc("EEE")] = np.nan     # późniejszy debiut
    fx_r = rng.normal(0, 0.006, size=n) - 0.4 * common
    fx = pd.Series(3.5 * np.exp(np.cumsum(fx_r)), index=idx, name="USDPLN")
    fx_lr = np.log(fx).diff().fillna(np.log(fx.iloc[0] / 3.5))
    r_pln = r_local.copy()
    for t in USD_TICKERS:
        r_pln[t] = r_local[t] + fx_lr.where(r_local[t].notna())
    # SF10 (obligacje): stała stopa dzienna w PLN, w obu wariantach
    r_local["OTS.SP"] = 0.0002
    r_pln["OTS.SP"] = 0.0002
    spx_r = 0.0003 + vol * (common * 1.1 + rng.normal(0, 0.004, size=n))
    spx = pd.Series(1000 * np.exp(np.cumsum(spx_r)), index=idx)
    vix = 100 * np.sqrt(252) * pd.Series(np.abs(spx_r), index=idx).rolling(21, min_periods=1).std().fillna(0.01) * 1.25
    vix = vix.clip(lower=9.0)
    vol_idx = pd.DataFrame({"VIX": vix, "VIX9D": vix * 0.95, "GVZ": 14 + 3 * np.sin(np.arange(n) / 40.0)},
                           index=idx)
    rate_pln = pd.Series(0.05, index=idx)
    rate_usd = pd.Series(0.03, index=idx)
    return MarketData(r_local, r_pln, fx, vol_idx, rate_pln, rate_usd, spx)


def overlay_config(**over) -> dict:
    """Konfiguracja nakładki z config/options.yaml z parametrami skróconymi do testów."""
    ocfg = copy.deepcopy(config.load("options")["monthly_overlay"])
    ocfg["lp"]["n_scenarios"] = 1500
    ocfg["portfolio"]["n_scenarios"] = 1500
    ocfg["signals"]["min_history_months"] = 3
    ocfg["annual_mc"].update(months=3, n_scenarios=300, sessions_per_month=10)
    for k, v in over.items():
        ocfg[k] = v
    return ocfg


def models_config() -> dict:
    cfg = copy.deepcopy(config.load("models"))
    cfg["rebalance"].update(window_days=300, min_obs=250, min_obs_deterministic=21)
    cfg["dependence"]["student_t"]["df_grid"] = [4, 8, 15, 30]
    cfg["dependence"]["student_t"]["refine"] = False
    return cfg
