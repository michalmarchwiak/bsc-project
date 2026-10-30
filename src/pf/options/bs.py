"""Model Blacka–Scholesa–Mertona dla opcji europejskich z ciągłą stopą dywidendy.

Oznaczenia: S – cena instrumentu bazowego, K – cena wykonania, T – czas do wygaśnięcia [lata],
r – stopa wolna od ryzyka, q – stopa dywidendy (kapitalizacja ciągła), σ – zmienność.

    d₁ = [ln(S/K) + (r − q + σ²/2) T] / (σ√T),   d₂ = d₁ − σ√T,
    C  = S e^{−qT} Φ(d₁) − K e^{−rT} Φ(d₂),
    P  = K e^{−rT} Φ(−d₂) − S e^{−qT} Φ(−d₁).

Greki (θ na rok kalendarzowy, ν na jednostkę σ, tj. na 100 p.p.):
    Δ_C = e^{−qT} Φ(d₁),  Δ_P = −e^{−qT} Φ(−d₁),
    Γ   = e^{−qT} φ(d₁) / (S σ √T),
    ν   = S e^{−qT} φ(d₁) √T,
    Θ_C = −S e^{−qT} φ(d₁) σ / (2√T) − r K e^{−rT} Φ(d₂) + q S e^{−qT} Φ(d₁),
    Θ_P = −S e^{−qT} φ(d₁) σ / (2√T) + r K e^{−rT} Φ(−d₂) − q S e^{−qT} Φ(−d₁).

Wszystkie funkcje przyjmują tablice numpy (broadcasting), co pozwala wyceniać jednocześnie
wszystkie scenariusze Monte Carlo. ``is_call`` – bool lub tablica bool.
Dla T = 0 cena równa się wartości wewnętrznej (wypłacie w wygaśnięciu).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

ArrayLike = float | np.ndarray

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _pdf(x: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def d1_d2(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike,
          sigma: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (np.asarray(r) - np.asarray(q) + 0.5 * sigma ** 2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def payoff(S: ArrayLike, K: ArrayLike, is_call: ArrayLike) -> np.ndarray:
    S, K = np.asarray(S, dtype=float), np.asarray(K, dtype=float)
    return np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))


def price(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike,
          is_call: ArrayLike) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    df_r, df_q = np.exp(-np.asarray(r) * T), np.exp(-np.asarray(q) * T)
    S, K = np.asarray(S, dtype=float), np.asarray(K, dtype=float)
    call = S * df_q * ndtr(d1) - K * df_r * ndtr(d2)
    put = K * df_r * ndtr(-d2) - S * df_q * ndtr(-d1)
    out = np.where(is_call, call, put)
    return np.where(T > 0, out, payoff(S, K, is_call))


def delta(S, K, T, r, q, sigma, is_call) -> np.ndarray:
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    df_q = np.exp(-np.asarray(q) * np.asarray(T))
    return np.where(is_call, df_q * ndtr(d1), -df_q * ndtr(-d1))


def gamma(S, K, T, r, q, sigma) -> np.ndarray:
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    T = np.asarray(T, dtype=float)
    return np.exp(-np.asarray(q) * T) * _pdf(d1) / (np.asarray(S) * np.asarray(sigma) * np.sqrt(T))


def vega(S, K, T, r, q, sigma) -> np.ndarray:
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    T = np.asarray(T, dtype=float)
    return np.asarray(S) * np.exp(-np.asarray(q) * T) * _pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, q, sigma, is_call) -> np.ndarray:
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    S, K, T = (np.asarray(x, dtype=float) for x in (S, K, T))
    r, q, sigma = np.asarray(r), np.asarray(q), np.asarray(sigma)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    common = -S * df_q * _pdf(d1) * sigma / (2.0 * np.sqrt(T))
    call = common - r * K * df_r * ndtr(d2) + q * S * df_q * ndtr(d1)
    put = common + r * K * df_r * ndtr(-d2) - q * S * df_q * ndtr(-d1)
    return np.where(is_call, call, put)


def greeks(S, K, T, r, q, sigma, is_call) -> dict[str, np.ndarray]:
    return {
        "delta": delta(S, K, T, r, q, sigma, is_call),
        "gamma": gamma(S, K, T, r, q, sigma),
        "vega": vega(S, K, T, r, q, sigma),
        "theta": theta(S, K, T, r, q, sigma, is_call),
    }


def forward(S: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike) -> np.ndarray:
    return np.asarray(S, dtype=float) * np.exp((np.asarray(r) - np.asarray(q)) * np.asarray(T))
