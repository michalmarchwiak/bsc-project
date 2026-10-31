"""Miary efektywności na miesięcznych stopach netto w PLN (r_t) i stopie wolnej od ryzyka okresu (r_f,t).

    zwrot roczny       (Π(1 + r_t))^{12/T} − 1,       zmienność roczna  s(r)·√12;
    CE zrealizowany    1 + r̃_t = (1 + r_t)/(1 + r_f,t),  CE_γ = ((1/T) Σ (1 + r̃_t)^{1−γ})^{1/(1−γ)} − 1,
                       annualizacja (1 + CE)^{12} − 1 (γ = 1: średnia geometryczna);
    Sharpe             SR = mean(r − r_f)/s(r − r_f)·√12, SE Lo (2002, iid): √12·√((1 + SR_m²/2)/T);
    Sortino            mean(r − r_f)/√(mean(min(r − r_f, 0)²))·√12;
    max DD             min_t (W_t / max_{s≤t} W_s − 1), W – majątek z r_t;   Calmar = zwrot roczny / |max DD|;
    Omega(0)           Σ max(r_t, 0) / Σ max(−r_t, 0);
    STARR              12·mean(r − r_f) / (ES_α miesięczne ex post, w skali rocznej ×12 → stosunek bez skali);
    ES historyczny     ES_α z próby strat −r_t (Acerbi–Tasche).
"""

from __future__ import annotations

import numpy as np

from pf.opt.cvar import empirical_es


def annual_return(r) -> float:
    r = np.asarray(r, dtype=float)
    return float(np.prod(1.0 + r) ** (12.0 / len(r)) - 1.0)


def annual_vol(r) -> float:
    return float(np.std(np.asarray(r, dtype=float), ddof=1) * np.sqrt(12.0))


def realized_ce(r, rf, gamma: float) -> float:
    """CE zrealizowany na stopach nadwyżkowych (miesięczny)."""
    g = (1.0 + np.asarray(r, dtype=float)) / (1.0 + np.asarray(rf, dtype=float))
    if abs(gamma - 1.0) < 1e-12:
        return float(np.exp(np.mean(np.log(g))) - 1.0)
    a = 1.0 - gamma
    return float(np.mean(g ** a) ** (1.0 / a) - 1.0)


def annualize_ce(ce_m: float) -> float:
    return float((1.0 + ce_m) ** 12 - 1.0)


def sharpe(r, rf) -> tuple[float, float]:
    """(SR roczny, SE Lo 2002 przy iid)."""
    x = np.asarray(r, dtype=float) - np.asarray(rf, dtype=float)
    sd = np.std(x, ddof=1)
    sr_m = x.mean() / sd if sd > 0 else np.nan
    se_m = np.sqrt((1.0 + 0.5 * sr_m ** 2) / len(x))
    return float(sr_m * np.sqrt(12.0)), float(se_m * np.sqrt(12.0))


def sortino(r, rf) -> float:
    x = np.asarray(r, dtype=float) - np.asarray(rf, dtype=float)
    dd = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
    return float(x.mean() / dd * np.sqrt(12.0)) if dd > 0 else np.nan


def max_drawdown(r) -> float:
    w = np.cumprod(1.0 + np.asarray(r, dtype=float))
    return float(np.min(w / np.maximum.accumulate(np.concatenate([[1.0], w]))[1:] - 1.0))


def drawdown_series(r) -> np.ndarray:
    w = np.cumprod(1.0 + np.asarray(r, dtype=float))
    return w / np.maximum.accumulate(np.concatenate([[1.0], w]))[1:] - 1.0


def calmar(r) -> float:
    mdd = max_drawdown(r)
    return float(annual_return(r) / abs(mdd)) if mdd < 0 else np.nan


def omega(r, threshold: float = 0.0) -> float:
    x = np.asarray(r, dtype=float) - threshold
    down = np.sum(np.maximum(-x, 0.0))
    return float(np.sum(np.maximum(x, 0.0)) / down) if down > 0 else np.inf


def historical_es(r, alpha: float = 0.975) -> float:
    return empirical_es(-np.asarray(r, dtype=float), alpha)


def starr(r, rf, es_monthly: float) -> float:
    """E[r − r_f] / ES_α (oba miesięczne)."""
    x = np.asarray(r, dtype=float) - np.asarray(rf, dtype=float)
    return float(x.mean() / es_monthly) if es_monthly and np.isfinite(es_monthly) else np.nan


def summary(r, rf, gamma: float | None = None, alpha: float = 0.975, es_expost: float | None = None) -> dict:
    r, rf = np.asarray(r, dtype=float), np.asarray(rf, dtype=float)
    sr, se = sharpe(r, rf)
    es_h = historical_es(r, alpha)
    out = {"T": len(r), "ann_return": annual_return(r), "ann_vol": annual_vol(r), "sharpe": sr, "sharpe_se": se,
           "sortino": sortino(r, rf), "max_dd": max_drawdown(r), "calmar": calmar(r), "omega": omega(r),
           "es_hist": es_h, "starr_hist": starr(r, rf, es_h),
           "ann_excess": annual_return((1 + r) / (1 + rf) - 1)}
    if es_expost is not None:
        out["starr"] = starr(r, rf, es_expost)
    if gamma is not None:
        ce = realized_ce(r, rf, gamma)
        out.update({"ce_m": ce, "ce_ann": annualize_ce(ce)})
    return out
