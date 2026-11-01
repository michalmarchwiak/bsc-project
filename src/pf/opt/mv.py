"""Benchmark średnia–wariancja na danych historycznych okna (``mv_hist``) i shrinkage Ledoita–Wolfa.

Shrinkage Ledoita–Wolfa (2004) do skalowanej macierzy jednostkowej, dla scentrowanych obserwacji
x_t ∈ ℝⁿ (t = 1, …, T), S = (1/T) Σ x_t x_tᵀ, ⟨A, B⟩ = tr(ABᵀ)/n:
    m = ⟨S, I⟩ = tr(S)/n,   d² = ‖S − mI‖²,   b̄² = (1/T²) Σ_t ‖x_t x_tᵀ − S‖²,   b² = min(b̄², d²),
    Σ̂ = (b²/d²)·m·I + (1 − b²/d²)·S   (intensywność shrinkage δ = b²/d² ∈ [0, 1]).

Portfel (poziom 1 w SF k i poziom 2 między SF, profil p):
    max_w  μᵀw − (γ_p/2) wᵀΣw,   Σw = 1,  0 ≤ w ≤ u,
μ = 21·średnia dziennych stóp prostych z okna, Σ = 21·Σ̂_LW (dni z kompletem notowań aktywów SF).
Poziom 2: dzienne stopy SF r_k = Σᵢ w_{k,i} r_i (brak sesji → 0), ograniczenia jak w strategiach
(W_max, look-through c_LT z luzowaniem proporcjonalnym).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from pf.opt.profiles import Profile, TwoLevelResult, feasible_start, relaxed_upper_bounds


def ledoit_wolf(X: np.ndarray) -> tuple[np.ndarray, float]:
    """(Σ̂, δ) – estymator Ledoita–Wolfa (2004) z celem m·I."""
    X = np.asarray(X, dtype=float)
    T, n = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    m = np.trace(S) / n
    d2 = np.sum((S - m * np.eye(n)) ** 2) / n
    # (1/T²) Σ_t ‖x_t x_tᵀ − S‖²/n = (1/T²) Σ_t [ (x_tᵀx_t)² − 2 x_tᵀ S x_t + ‖S‖² ] / n
    sq = np.sum(Xc * Xc, axis=1)
    b2_bar = (np.sum(sq ** 2) - 2.0 * np.einsum("ti,ij,tj->", Xc, S, Xc) + T * np.sum(S * S)) / (n * T * T)
    b2 = min(b2_bar, d2)
    delta = 0.0 if d2 <= 0 else b2 / d2
    return delta * m * np.eye(n) + (1.0 - delta) * S, float(delta)


def solve_mv(mu: np.ndarray, Sigma: np.ndarray, gamma: float, ub, opt: dict | None = None) -> np.ndarray:
    opt = opt or {}
    n = len(mu)
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (n,)).copy()
    if n == 1:
        return np.ones(1)

    def f(w):
        Sw = Sigma @ w
        return -(mu @ w - 0.5 * gamma * w @ Sw), -(mu - gamma * Sw)

    w0 = feasible_start(ub)
    res = optimize.minimize(f, w0, jac=True, method="SLSQP", bounds=[(0.0, float(u)) for u in ub],
                            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                          "jac": lambda w: np.ones_like(w)}],
                            options={"maxiter": opt.get("maxiter", 500), "ftol": opt.get("ftol", 1e-12)})
    w = np.clip(res.x, 0.0, ub)
    w = w / w.sum()
    return w if f(w)[0] <= f(w0)[0] + 1e-15 else w0


def mv_two_level(window: pd.DataFrame, sfs_tickers: dict[str, list[str]], profile: Profile, horizon: int = 21,
                 opt: dict | None = None) -> TwoLevelResult:
    """Benchmark ``mv_hist`` na dziennych log-stopach okna (dane ≤ t−1)."""
    simple = np.expm1(window)
    w_sf, shrink = {}, {}
    daily_sf = {}
    for k, tickers in sfs_tickers.items():
        X = simple[tickers].dropna(how="any")
        n = len(tickers)
        if n == 1:
            w = np.ones(1)
            shrink[k] = np.nan
        else:
            Sig, delta = ledoit_wolf(X.to_numpy())
            shrink[k] = delta
            w = solve_mv(horizon * X.mean().to_numpy(), horizon * Sig, profile.gamma,
                         max(profile.c_sf, 1.0 / n), opt)
        w_sf[k] = pd.Series(w, index=tickers)
        daily_sf[k] = simple[tickers].fillna(0.0).to_numpy() @ w
    names = list(sfs_tickers)
    D = pd.DataFrame(daily_sf, index=simple.index)[names]
    D = D[(simple[[t for ts in sfs_tickers.values() for t in ts]].notna().any(axis=1)).to_numpy()]
    m = np.array([w_sf[k].max() for k in names])
    ub, flags = relaxed_upper_bounds(m, profile.w_max, profile.c_lt)
    Sig, delta = ledoit_wolf(D.to_numpy())
    W = solve_mv(horizon * D.mean().to_numpy(), horizon * Sig, profile.gamma, ub, opt)
    return TwoLevelResult("mv_hist", profile.id, pd.Series(W, index=names), w_sf, flags,
                          {"shrinkage_level1": shrink, "shrinkage_level2": delta})
