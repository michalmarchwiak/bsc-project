"""Optymalizacja portfela sub-funduszy na scenariuszach: max E[R] przy ES_α ≤ L (Rockafellar–Uryasev).

Dla scenariuszy R_s (s = 1, …, S) i strat L_s(W) = −R_sᵀW:

    max  (1/S) Σ_s R_sᵀ W
    p.w. ζ + 1/((1 − α) S) Σ_s u_s ≤ L,
         u_s ≥ −R_sᵀ W − ζ,  u_s ≥ 0,
         Σ_k W_k = 1,  0 ≤ W_k ≤ w_max.

W optimum ζ jest VaR_α, a ζ + (1/((1−α)S)) Σ u_s – ES_α portfela. Gdy limit L jest
nieosiągalny, zwracany jest portfel o minimalnym ES (z flagą ``feasible = False``).
Problem liniowy rozwiązywany jest solverem HiGHS (``scipy.optimize.linprog``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, sparse


@dataclass
class Solution:
    weights: np.ndarray
    expected_return: float
    es: float
    var: float
    feasible: bool
    status: str


def empirical_es(losses: np.ndarray, alpha: float) -> float:
    """ES_α z próby (definicja Acerbiego–Taschego, z ułamkową wagą kwantyla)."""
    x = np.sort(np.asarray(losses))[::-1]
    S = len(x)
    k = (1.0 - alpha) * S
    m = int(np.floor(k))
    tail = x[:m].sum() + (k - m) * (x[m] if m < S else 0.0)
    return float(tail / k)


def _solve(R: np.ndarray, alpha: float, es_limit: float | None, w_max: float, minimize_es: bool):
    S, K = R.shape
    n = K + 1 + S                       # [W, ζ, u]
    c = np.zeros(n)
    if minimize_es:
        c[K] = 1.0
        c[K + 1:] = 1.0 / ((1.0 - alpha) * S)
    else:
        c[:K] = -R.mean(axis=0)
    # −R_s W − ζ − u_s ≤ 0
    A_scen = sparse.hstack([sparse.csr_matrix(-R), sparse.csr_matrix(-np.ones((S, 1))), -sparse.eye(S)])
    rows, b = [A_scen], [np.zeros(S)]
    if es_limit is not None:
        es_row = np.concatenate([np.zeros(K), [1.0], np.full(S, 1.0 / ((1.0 - alpha) * S))])
        rows.append(sparse.csr_matrix(es_row))
        b.append([es_limit])
    A_ub = sparse.vstack(rows).tocsr()
    A_eq = sparse.csr_matrix(np.concatenate([np.ones(K), np.zeros(1 + S)]))
    bounds = [(0.0, w_max)] * K + [(None, None)] + [(0.0, None)] * S
    return optimize.linprog(c, A_ub=A_ub, b_ub=np.concatenate(b), A_eq=A_eq, b_eq=[1.0],
                            bounds=bounds, method="highs")


def max_return_es(R: np.ndarray, alpha: float = 0.975, es_limit: float = 0.08,
                  w_max: float = 1.0) -> Solution:
    """Portfel maksymalnego oczekiwanego zwrotu przy ES_α(−R W) ≤ ``es_limit``."""
    K = R.shape[1]
    res = _solve(R, alpha, es_limit, w_max, minimize_es=False)
    feasible = res.status == 0
    if not feasible:
        res = _solve(R, alpha, None, w_max, minimize_es=True)
        if res.status != 0:
            raise RuntimeError(f"LP nierozwiązywalny: {res.message}")
    w = np.clip(res.x[:K], 0.0, None)
    w /= w.sum()
    losses = -R @ w
    return Solution(weights=w, expected_return=float(R.mean(axis=0) @ w), es=empirical_es(losses, alpha),
                    var=float(res.x[K]), feasible=feasible, status=res.message)
