"""Zmienność implikowana ze środka kwotowań bid/ask oraz filtry jakości kwotowań.

Solver: zwektoryzowana metoda Newtona z zabezpieczeniem bisekcją. Dla każdej serii
utrzymywany jest przedział [σ_lo, σ_hi] zawierający pierwiastek f(σ) = BS(σ) − cena
(BS jest ściśle rosnąca w σ, ν > 0); krok Newtona σ − f/ν jest przyjmowany tylko wtedy,
gdy pozostaje wewnątrz przedziału, w przeciwnym razie wykonywany jest krok bisekcji.
Gwarantuje to zbieżność także dla opcji głęboko OTM/ITM o znikomej vedze.

Ceny spoza przedziału braku arbitrażu (dolne ograniczenie: wartość wewnętrzna forwardowa,
górne: S e^{−qT} dla call, K e^{−rT} dla put) dają NaN.

Filtry kwotowań (``config/options.yaml: iv_filters``): bid = 0, względny spread
(ask − bid)/mid powyżej progu, zerowe open interest, przestarzała data ostatniej transakcji,
czas do wygaśnięcia spoza zadanego zakresu. Kolumna ``impliedVolatility`` z Yahoo jest ignorowana.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pf.options import bs

SIGMA_LO, SIGMA_HI = 1e-4, 5.0


def no_arbitrage_bounds(S, K, T, r, q, is_call) -> tuple[np.ndarray, np.ndarray]:
    S, K, T = (np.asarray(x, dtype=float) for x in (S, K, T))
    disc_s, disc_k = S * np.exp(-np.asarray(q) * T), K * np.exp(-np.asarray(r) * T)
    lower = np.where(is_call, np.maximum(disc_s - disc_k, 0.0), np.maximum(disc_k - disc_s, 0.0))
    upper = np.where(is_call, disc_s, disc_k)
    return lower, upper


def implied_vol(price, S, K, T, r, q, is_call, tol: float = 1e-12, max_iter: int = 100) -> np.ndarray:
    """Zmienność implikowana σ taka, że BS(σ) = ``price`` (względna dokładność ceny ``tol``)."""
    price, S, K, T, r, q, is_call = np.broadcast_arrays(*(np.asarray(x) for x in
                                                         (price, S, K, T, r, q, is_call)))
    price, S, K, T = (x.astype(float) for x in (price, S, K, T))
    lower, upper = no_arbitrage_bounds(S, K, T, r, q, is_call)
    valid = (T > 0) & np.isfinite(price) & (price > lower) & (price < upper)

    lo = np.full(price.shape, SIGMA_LO)
    hi = np.full(price.shape, SIGMA_HI)
    valid &= bs.price(S, K, T, r, q, hi, is_call) >= price
    # Przybliżenie startowe Brennera–Subrahmanyama: σ ≈ √(2π/T) · cena / S.
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.clip(np.sqrt(2 * np.pi / T) * price / S, 0.05, 1.0)
    sigma = np.where(valid, sigma, np.nan)
    active = valid.copy()
    for _ in range(max_iter):
        if not active.any():
            break
        f = bs.price(S, K, T, r, q, sigma, is_call) - price
        done = np.abs(f) <= tol * price
        active &= ~done
        hi = np.where(active & (f > 0), sigma, hi)
        lo = np.where(active & (f <= 0), sigma, lo)
        v = bs.vega(S, K, T, r, q, sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = sigma - f / v
        ok = np.isfinite(newton) & (newton > lo) & (newton < hi)
        step = np.where(ok, newton, 0.5 * (lo + hi))
        sigma = np.where(active, step, sigma)
        active &= (hi - lo) > 1e-14
    return np.where(valid, sigma, np.nan)


def quote_filters(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Zwraca powód odrzucenia każdej serii (pusty napis = seria przyjęta).

    Wymagane kolumny: quote_date, expiration, bid, ask; opcjonalne: openInterest, lastTradeDate.
    """
    reason = pd.Series("", index=df.index, dtype=object)

    def flag(mask, why: str) -> None:
        mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
        reason.loc[mask & (reason == "")] = why

    mid = 0.5 * (df["bid"] + df["ask"])
    days = (df["expiration"] - df["quote_date"]).dt.days
    flag(df["bid"].isna() | df["ask"].isna(), "missing_quote")
    flag(df["bid"] <= cfg["min_bid"], "zero_bid")
    flag(df["ask"] < df["bid"], "crossed")
    flag((df["ask"] - df["bid"]) / mid > cfg["max_rel_spread"], "wide_spread")
    if cfg.get("require_open_interest", True) and "openInterest" in df.columns:
        flag(df["openInterest"].fillna(0) <= 0, "zero_oi")
    if "lastTradeDate" in df.columns:
        ltd = pd.to_datetime(df["lastTradeDate"], utc=True).dt.tz_localize(None).dt.normalize()
        known = ltd.notna()
        stale = np.zeros(len(df), dtype=bool)
        stale[known.to_numpy()] = np.busday_count(
            ltd[known].values.astype("datetime64[D]"),
            df.loc[known, "quote_date"].values.astype("datetime64[D]")) > cfg["max_stale_days"]
        flag(stale | ~known.to_numpy(), "stale_trade")
    flag(days < cfg["min_days_to_expiry"], "short_expiry")
    flag(days > cfg["max_days_to_expiry"], "long_expiry")
    return reason


def implied_vols_from_quotes(df: pd.DataFrame, cfg: dict, r, q) -> pd.DataFrame:
    """Dodaje kolumny mid, T (lata, ACT/365), iv i reject do tabeli kwotowań.

    Wymagane kolumny: quote_date, expiration, strike, call_put ('C'/'P'), bid, ask,
    underlying_price. ``r`` i ``q`` – skalary lub tablice zgodne z wierszami ``df``.
    """
    out = df.copy()
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["T"] = (out["expiration"] - out["quote_date"]).dt.days / 365.0
    out["reject"] = quote_filters(out, cfg)
    iv = implied_vol(out["mid"].to_numpy(), out["underlying_price"].to_numpy(), out["strike"].to_numpy(),
                     out["T"].to_numpy(), r, q, (out["call_put"] == "C").to_numpy())
    out["iv"] = np.where(out["reject"] == "", iv, np.nan)
    out.loc[(out["reject"] == "") & out["iv"].isna(), "reject"] = "no_iv"
    return out
