"""Testy wsteczne prognoz VaR i ES (miesięczne straty, poziom α = 0,975).

Przekroczenia I_t = 1{L_t > VaR_t}, p = 1 − α, T obserwacji, x = Σ I_t.

Kupiec (POF): LR_uc = −2 ln[(1−p)^{T−x} p^x] + 2 ln[(1−x/T)^{T−x} (x/T)^x] ~ χ²₁;
    dokładna p-wartość: dwustronny test dwumianowy Bin(T, p).
Christoffersen: n_ij – liczba przejść i → j, π_i = n_i1/(n_i0 + n_i1), π = (n01 + n11)/(T − 1),
    LR_ind = −2 ln[(1−π)^{n00+n10} π^{n01+n11}] + 2 ln[(1−π₀)^{n00} π₀^{n01} (1−π₁)^{n10} π₁^{n11}] ~ χ²₁,
    LR_cc = LR_uc + LR_ind ~ χ²₂ (pokrycie warunkowe).
Acerbi–Szekely (2014): Z₂ = 1 − (1/(T(1−α))) Σ_t L_t I_t / ES_t; pod H0 E[Z₂] = 0, odrzucenie przy Z₂
    istotnie < 0. Rozkład pod H0 symulowany z zapisanych ogonów prognoz: w każdym t z prawd. 1 − α
    losowana jest strata z ogona (najgorsze ⌈(1−α)N⌉ scenariuszy), w przeciwnym razie brak przekroczenia.
Du–Escanciano (2017): H_t = (u_t − α)·1{u_t ≥ α}/(1 − α), u_t – PIT zrealizowanej straty;
    test bezwarunkowy U = √T (H̄ − (1−α)/2)/√((1−α)(1/3 − (1−α)/4)) ~ N(0, 1),
    warunkowy C_m = T Σ_{j=1}^m ρ̂_j² ~ χ²_m (Box–Pierce na autokorelacjach H_t − (1−α)/2).

Przy próbie miesięcznej oczekiwana liczba przekroczeń T(1 − α) jest mała (np. 1,7 dla T = 69), więc moc
wszystkich testów jest niska – brak odrzucenia nie potwierdza poprawności modelu.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec(hits: np.ndarray, p: float) -> dict:
    h = np.asarray(hits, dtype=bool)
    T, x = len(h), int(h.sum())
    pi = x / T if T else np.nan
    lr = -2.0 * (_xlogy(T - x, 1 - p) + _xlogy(x, p)) + 2.0 * (_xlogy(T - x, 1 - pi) + _xlogy(x, pi)) if T else np.nan
    return {"T": T, "exceedances": x, "expected": T * p, "lr_uc": float(lr),
            "p_uc": float(stats.chi2.sf(lr, 1)) if T else np.nan,
            "p_binom": float(stats.binomtest(x, T, p).pvalue) if T else np.nan}


def christoffersen(hits: np.ndarray, p: float) -> dict:
    h = np.asarray(hits, dtype=int)
    uc = kupiec(h, p)
    if len(h) < 2:
        return {**uc, "lr_ind": np.nan, "p_ind": np.nan, "lr_cc": np.nan, "p_cc": np.nan}
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    l0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    l1 = _xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1)
    lr_ind = max(-2.0 * (l0 - l1), 0.0)
    lr_cc = uc["lr_uc"] + lr_ind
    return {**uc, "n00": n00, "n01": n01, "n10": n10, "n11": n11, "lr_ind": lr_ind,
            "p_ind": float(stats.chi2.sf(lr_ind, 1)), "lr_cc": lr_cc, "p_cc": float(stats.chi2.sf(lr_cc, 2))}


def z2_stat(loss: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> float:
    loss, var, es = (np.asarray(x, dtype=float) for x in (loss, var, es))
    T = len(loss)
    return float(1.0 - np.sum(loss * (loss > var) / es) / (T * (1.0 - alpha)))


def z2_test(loss: np.ndarray, var: np.ndarray, es: np.ndarray, tails: list[np.ndarray], alpha: float,
            n_sim: int = 20000, rng: np.random.Generator | None = None, level: float = 0.05) -> dict:
    """Z₂ i jej rozkład pod H0 symulowany z ogonów prognoz; p = P(Z₂* ≤ Z₂)."""
    rng = rng or np.random.default_rng(0)
    z = z2_stat(loss, var, es, alpha)
    T = len(loss)
    es = np.asarray(es, dtype=float)
    acc = np.zeros(n_sim)
    for t in range(T):
        tail = np.asarray(tails[t], dtype=float)
        hit = rng.random(n_sim) < (1.0 - alpha)
        draw = tail[rng.integers(0, len(tail), n_sim)]
        acc += np.where(hit, draw / es[t], 0.0)
    zs = 1.0 - acc / (T * (1.0 - alpha))
    return {"z2": z, "p_z2": float((np.sum(zs <= z) + 1) / (n_sim + 1)), "z2_crit": float(np.quantile(zs, level)),
            "z2_sim_mean": float(zs.mean())}


def du_escanciano(u: np.ndarray, alpha: float, lags=(1, 5)) -> dict:
    u = np.asarray(u, dtype=float)
    u = u[np.isfinite(u)]
    T = len(u)
    H = (u - alpha) * (u >= alpha) / (1.0 - alpha)
    mean0 = (1.0 - alpha) / 2.0
    var0 = (1.0 - alpha) * (1.0 / 3.0 - (1.0 - alpha) / 4.0)
    U = np.sqrt(T) * (H.mean() - mean0) / np.sqrt(var0) if T else np.nan
    out = {"de_uc": float(U), "p_de_uc": float(2.0 * stats.norm.sf(abs(U))) if T else np.nan, "H_mean": float(H.mean())}
    d = H - mean0
    g0 = np.mean(d * d)
    for m in lags:
        if T <= m or g0 <= 0:
            out[f"de_cc{m}"], out[f"p_de_cc{m}"] = np.nan, np.nan
            continue
        rho = np.array([np.sum(d[j:] * d[:-j]) / (T - j) / g0 for j in range(1, m + 1)])
        C = T * np.sum(rho ** 2)
        out[f"de_cc{m}"], out[f"p_de_cc{m}"] = float(C), float(stats.chi2.sf(C, m))
    return out


def backtest_series(loss, var, es, u, tails, alpha: float, n_sim: int, lags=(1, 5), seed: int = 0,
                    level: float = 0.05) -> dict:
    """Wszystkie testy dla jednego szeregu (pomija obserwacje bez zrealizowanej straty)."""
    loss, var, es, u = (np.asarray(x, dtype=float) for x in (loss, var, es, u))
    ok = np.isfinite(loss)
    hits = loss[ok] > var[ok]
    res = christoffersen(hits, 1.0 - alpha)
    res.update(z2_test(loss[ok], var[ok], es[ok], [t for t, o in zip(tails, ok) if o], alpha, n_sim,
                       np.random.default_rng(seed), level))
    res.update(du_escanciano(u[ok], alpha, lags))
    return res
