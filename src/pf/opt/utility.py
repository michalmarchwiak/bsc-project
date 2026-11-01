"""Ekwiwalent pewności CRRA na scenariuszach miesięcznych stóp prostych.

Dla scenariuszy R_s (s = 1, …, S) i wag w, majątek końcowy g_s = 1 + R_s w:

    CE_γ(w) = ( (1/S) Σ_s g_s^{1−γ} )^{1/(1−γ)} − 1,     γ ≠ 1,
    CE_1(w) = exp( (1/S) Σ_s ln g_s ) − 1.

CE_γ jest średnią potęgową rzędu 1 − γ < 1 (dla γ > 0) funkcji liniowej w, więc jest wklęsła w w;
z nierówności średnich CE_γ(w) ≤ E[R w] (równość dla stałej stopy). Gradient (M = (1/S) Σ g_s^{1−γ}):

    ∇CE_γ(w) = M^{γ/(1−γ)} · (1/S) Σ_s g_s^{−γ} R_s,      ∇CE_1(w) = e^{(1/S)Σ ln g_s} · (1/S) Σ_s R_s / g_s.

Obliczenia w logarytmach (log-sum-exp) – stabilne dla dużego γ. Zabezpieczenie g_s ≥ ``min_wealth``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def ce_and_grad(R: np.ndarray, w: np.ndarray, gamma: float, min_wealth: float = 1e-8) -> tuple[float, np.ndarray]:
    """(CE_γ(w), ∇CE_γ(w)) dla scenariuszy R (S × n)."""
    R = np.asarray(R, dtype=float)
    g = np.maximum(1.0 + R @ w, min_wealth)
    lg = np.log(g)
    S = len(g)
    if abs(gamma - 1.0) < 1e-12:
        m = lg.mean()
        val = np.exp(m)
        return float(val - 1.0), val * (R / g[:, None]).mean(axis=0)
    a = 1.0 - gamma
    log_M = logsumexp(a * lg) - np.log(S)                       # ln M
    ce = np.exp(log_M / a) - 1.0
    # M^{γ/(1−γ)} · (1/S) Σ g^{−γ} R_s = Σ_s exp(−γ ln g_s + (γ/(1−γ)) ln M − ln S) R_s
    log_c = -gamma * lg + (gamma / a) * log_M - np.log(S)
    grad = np.exp(log_c) @ R
    return float(ce), grad


def certainty_equivalent(R: np.ndarray, w: np.ndarray | None, gamma: float, min_wealth: float = 1e-8) -> float:
    """CE_γ(w); ``w = None`` – R traktowane jako wektor stóp portfela."""
    R = np.asarray(R, dtype=float)
    if w is None:
        return ce_and_grad(R[:, None], np.ones(1), gamma, min_wealth)[0]
    return ce_and_grad(R, w, gamma, min_wealth)[0]


def crra_utility(g: np.ndarray, gamma: float) -> np.ndarray:
    """u_γ(g) = g^{1−γ}/(1−γ) (γ ≠ 1), ln g (γ = 1)."""
    g = np.asarray(g, dtype=float)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(g)
    return g ** (1.0 - gamma) / (1.0 - gamma)
