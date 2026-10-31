"""Miara zależności portfela D(w) = wᵀ M w na scenariuszach miesięcznych stóp zwrotu.

M liczona jest z pseudo-obserwacji scenariuszy V_si = rank(R_si) / (S + 1):

    lambda_L:  Λ_ij(q) = (1/(S q)) Σ_s 1{V_si ≤ q} · 1{V_sj ≤ q}  – macierz zależności w dolnym ogonie;
               Λ = IᵀI / (S q), I_si = 1{V_si ≤ q} – macierz Grama, więc dodatnio półokreślona;
               dla niezależnych składowych Λ_ij ≈ q (i ≠ j), dla komonotonicznych Λ_ij = 1;
    rho_S:     ρ_S = corr(V) – macierz korelacji rang Spearmana (dodatnio półokreślona).

D(w) jest więc wypukła, a −κ·σ^EW·D(w) wklęsła; ∇D(w) = 2 M w.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def pseudo_obs(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    return rankdata(R, axis=0) / (R.shape[0] + 1.0)


def lower_tail_matrix(R: np.ndarray, q: float = 0.05) -> np.ndarray:
    V = pseudo_obs(R)
    I = (V <= q).astype(float)
    return I.T @ I / (len(V) * q)


def spearman_matrix(R: np.ndarray) -> np.ndarray:
    V = pseudo_obs(R)
    if V.shape[1] == 1:
        return np.ones((1, 1))
    return np.corrcoef(V, rowvar=False)


def dependence_matrix(R: np.ndarray, measure: str = "lambda_L", q: float = 0.05) -> np.ndarray:
    if measure == "lambda_L":
        return lower_tail_matrix(R, q)
    if measure == "rho_S":
        return spearman_matrix(R)
    raise ValueError(f"Nieznana miara zależności: {measure}")


def penalty_and_grad(M: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    Mw = M @ w
    return float(w @ Mw), 2.0 * Mw
