"""Profile inwestora i dwupoziomowa optymalizacja składu funduszu na scenariuszach (Etap 3).

Poziom 1 (skład sub-funduszu k, profil p, scenariusze R^{(k)} ∈ ℝ^{S×n_k}):

    max_w  CE_γp(w) − κ_p · σ_k^EW · D_k(w),   Σ wᵢ = 1,  0 ≤ wᵢ ≤ max(c_SF,p, 1/n_k),

σ_k^EW – odchylenie standardowe scenariuszowej stopy SF przy równych wagach (nadaje karze jednostkę
stopy zwrotu), D_k(w) = wᵀ M_k w (``pf.opt.dependence_penalty``).

Poziom 2 (wagi W między SF, scenariusze R^SF_{s,k} = R^{(k)}_s w_{k,p}):

    max_W  CE_γp(W) − κ_p · σ^EW_SF · D^SF(W),   Σ W_k = 1,  0 ≤ W_k ≤ W_max,p,  W_k · maxᵢ w_{k,p,i} ≤ c_LT,p.

Ograniczenie look-through jest ograniczeniem górnym W_k ≤ u_k = min(W_max, c_LT / m_k), m_k = maxᵢ w_{k,i}.
Gdy Σ_k u_k < 1 (problem niedopuszczalny), c_LT jest luzowany proporcjonalnie: c_LT → s·c_LT
z minimalnym s ≥ 1 takim, że Σ_k min(W_max, s c_LT / m_k) ≥ 1 (flaga ``lt_relaxed``); gdy K·W_max < 1,
także W_max → 1/K (flaga ``wmax_relaxed``).

Strategie (oba poziomy): ``ce_dep`` (CE − kara za zależność), ``ce`` (κ = 0), ``maxret_es`` (max E[R]
przy ES_α ≤ ES_α portfela równych wag, LP Rockafellara–Uryaseva), ``min_es`` (min ES_α, LP), limity z profilu.
Benchmarki: ``ew`` (1/N w SF i między SF, bez limitów look-through), ``mv_hist`` (``pf.opt.mv``).

Problem CE jest wklęsły (CE_γ wklęsła, D wypukła), rozwiązywany SLSQP z gradientem analitycznym
startującym z wag równych (lub dopuszczalnego punktu u/Σu, gdy wagi równe naruszają ograniczenia).
Dekompozycja dwupoziomowa optymalizuje CE pojedynczego SF z γ inwestora – to heurystyka,
a nie optimum łączne względem wag look-through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from pf.opt.cvar import _solve, empirical_es, max_return_es
from pf.opt.dependence_penalty import dependence_matrix, penalty_and_grad
from pf.opt.utility import ce_and_grad

log = logging.getLogger(__name__)

MODEL_STRATEGIES = ("ce_dep", "ce", "maxret_es", "min_es")


@dataclass(frozen=True)
class Profile:
    id: str
    name: str
    gamma: float
    kappa: float
    c_sf: float
    c_lt: float
    w_max: float


def load_profiles(cfg: dict) -> dict[str, Profile]:
    return {pid: Profile(pid, p["name"], float(p["gamma"]), float(p["kappa"]), float(p["c_sf"]),
                         float(p["c_lt"]), float(p["w_max"])) for pid, p in cfg["profiles"].items()}


@dataclass
class TwoLevelResult:
    strategy: str
    profile: str
    W: pd.Series                                  # wagi sub-funduszy
    w: dict[str, pd.Series]                       # skład każdego SF (tickery)
    flags: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @property
    def x(self) -> pd.Series:
        """Ekspozycja look-through x = Σ_k W_k w_k (tickery)."""
        parts = [self.W[k] * w for k, w in self.w.items() if k in self.W.index]
        return pd.concat(parts).groupby(level=0).sum()


# --------------------------------------------------------------------------- pomocnicze

def feasible_start(ub: np.ndarray) -> np.ndarray:
    """Wagi równe, jeśli dopuszczalne; w przeciwnym razie u/Σu (Σu ≥ 1)."""
    n = len(ub)
    w = np.full(n, 1.0 / n)
    if np.all(w <= ub + 1e-15):
        return w
    return ub / ub.sum()


def solve_ce(R: np.ndarray, gamma: float, kappa: float, ub, M: np.ndarray | None = None,
             sigma: float | None = None, opt: dict | None = None) -> tuple[np.ndarray, dict]:
    """max CE_γ(w) − κ·σ·wᵀMw przy Σw = 1, 0 ≤ w ≤ ub (SLSQP, gradient analityczny)."""
    opt = opt or {}
    R = np.asarray(R, dtype=float)
    n = R.shape[1]
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (n,)).copy()
    if n == 1:
        return np.ones(1), {"success": True, "nit": 0, "objective": ce_and_grad(R, np.ones(1), gamma)[0]}
    min_wealth = opt.get("min_wealth", 1e-8)
    k = float(kappa) * (float(sigma) if sigma is not None else 1.0)
    use_pen = k > 0 and M is not None

    def f(w):
        ce, g = ce_and_grad(R, w, gamma, min_wealth)
        if use_pen:
            d, gd = penalty_and_grad(M, w)
            return -(ce - k * d), -(g - k * gd)
        return -ce, -g

    w0 = feasible_start(ub)
    res = optimize.minimize(f, w0, jac=True, method="SLSQP", bounds=[(0.0, float(u)) for u in ub],
                            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                          "jac": lambda w: np.ones_like(w)}],
                            options={"maxiter": opt.get("maxiter", 500), "ftol": opt.get("ftol", 1e-12)})
    w = np.clip(res.x, 0.0, ub)
    w = w / w.sum()
    # rozwiązanie nie gorsze od punktu startowego (zabezpieczenie przy braku zbieżności)
    if f(w)[0] > f(w0)[0] + 1e-12:
        w, res_success = w0, False
    else:
        res_success = bool(res.success)
    return w, {"success": res_success, "nit": int(res.nit), "objective": float(-f(w)[0]), "message": res.message}


def relaxed_upper_bounds(m: np.ndarray, w_max: float, c_lt: float) -> tuple[np.ndarray, dict]:
    """u_k = min(W_max, s·c_LT/m_k) z minimalnym s ≥ 1 zapewniającym Σu ≥ 1."""
    K = len(m)
    flags = {"lt_relaxed": False, "lt_factor": 1.0, "wmax_relaxed": False}
    if K * w_max < 1.0 - 1e-12:
        w_max = 1.0 / K
        flags["wmax_relaxed"] = True
    m = np.maximum(np.asarray(m, dtype=float), 1e-12)

    def total(s):
        return np.minimum(w_max, s * c_lt / m).sum()

    s = 1.0
    if total(1.0) < 1.0 - 1e-12:
        lo, hi = 1.0, 2.0
        while total(hi) < 1.0:
            hi *= 2.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if total(mid) < 1.0 else (lo, mid)
        s = hi
        flags.update(lt_relaxed=True, lt_factor=float(s))
    ub = np.minimum(w_max, s * c_lt / m)
    return ub, flags


def _lp(R: np.ndarray, alpha: float, ub: np.ndarray, strategy: str) -> tuple[np.ndarray, dict]:
    n = R.shape[1]
    if n == 1:
        return np.ones(1), {"feasible": True}
    if strategy == "min_es":
        res = _solve(R, alpha, None, ub, minimize_es=True)
        if res.status != 0:
            raise RuntimeError(f"LP min ES nierozwiązywalny: {res.message}")
        w = np.clip(res.x[:n], 0.0, ub)
        return w / w.sum(), {"feasible": True}
    w_ew = np.full(n, 1.0 / n)
    es_ew = empirical_es(-R @ w_ew, alpha)
    sol = max_return_es(R, alpha, es_ew + 1e-10 * max(1.0, abs(es_ew)), ub)
    w = np.clip(sol.weights, 0.0, ub)
    return w / w.sum(), {"feasible": bool(sol.feasible), "es_limit": es_ew}


# --------------------------------------------------------------------------- dwupoziomowa optymalizacja

@dataclass
class SubfundData:
    """Scenariusze aktywów jednego SF i wielkości niezależne od profilu (M_k, σ_k^EW)."""
    name: str
    tickers: list[str]
    R: np.ndarray
    sigma_ew: float
    M: dict = field(default_factory=dict)       # (measure, q) → M_k

    def matrix(self, measure: str, q: float) -> np.ndarray:
        key = (measure, round(q, 10))
        if key not in self.M:
            self.M[key] = dependence_matrix(self.R, measure, q)
        return self.M[key]


def subfund_data(R: np.ndarray, assets: list[str], subfunds: dict) -> list[SubfundData]:
    pos = {a: i for i, a in enumerate(assets)}
    out = []
    for sf, spec in subfunds.items():
        members = [a for a in spec["tickers"] if a in pos]
        if not members:
            continue
        Rk = R[:, [pos[a] for a in members]]
        out.append(SubfundData(sf, members, Rk, float((Rk.mean(axis=1)).std(ddof=1))))
    return out


def optimize_two_level(sfs: list[SubfundData], profile: Profile, strategy: str, alpha: float, opt: dict,
                       measure: str = "lambda_L", q: float = 0.05, kappa: float | None = None,
                       level1_cache: dict | None = None) -> TwoLevelResult:
    """Skład SF (poziom 1) i wagi SF (poziom 2) dla jednej strategii i profilu."""
    kap = profile.kappa if kappa is None else float(kappa)
    if strategy == "ce":
        kap = 0.0
    w_sf, info = {}, {"level1": {}}
    for sf in sfs:
        n = len(sf.tickers)
        cap = max(profile.c_sf, 1.0 / n)
        if strategy in ("ce_dep", "ce"):
            key = (sf.name, strategy, round(cap, 12), profile.gamma, kap, measure if kap > 0 else None,
                   round(q, 10) if kap > 0 else None)
        else:
            key = (sf.name, strategy, round(cap, 12))
        if level1_cache is not None and key in level1_cache:
            w, inf = level1_cache[key]
        elif strategy in ("ce_dep", "ce"):
            M = sf.matrix(measure, q) if kap > 0 else None
            w, inf = solve_ce(sf.R, profile.gamma, kap, cap, M, sf.sigma_ew, opt)
        elif strategy in ("maxret_es", "min_es"):
            w, inf = _lp(sf.R, alpha, np.full(n, cap), strategy)
        else:
            raise ValueError(f"Nieznana strategia: {strategy}")
        if level1_cache is not None:
            level1_cache[key] = (w, inf)
        w_sf[sf.name] = pd.Series(w, index=sf.tickers)
        info["level1"][sf.name] = inf
    names = [sf.name for sf in sfs]
    R_sf = np.column_stack([sf.R @ w_sf[sf.name].to_numpy() for sf in sfs])
    m = np.array([w_sf[k].max() for k in names])
    ub, flags = relaxed_upper_bounds(m, profile.w_max, profile.c_lt)
    if flags["lt_relaxed"] or flags["wmax_relaxed"]:
        log.warning("%s/%s: luzowanie limitów poziomu 2 (c_LT ×%.3f, W_max %s)", profile.id, strategy,
                    flags["lt_factor"], "→ 1/K" if flags["wmax_relaxed"] else "bez zmian")
    if strategy in ("ce_dep", "ce"):
        M = dependence_matrix(R_sf, measure, q) if kap > 0 else None
        sigma = float(R_sf.mean(axis=1).std(ddof=1))
        W, inf2 = solve_ce(R_sf, profile.gamma, kap, ub, M, sigma, opt)
    else:
        W, inf2 = _lp(R_sf, alpha, ub, strategy)
    info["level2"] = inf2
    flags["infeasible_es"] = strategy == "maxret_es" and not (
        inf2.get("feasible", True) and all(v.get("feasible", True) for v in info["level1"].values()))
    return TwoLevelResult(strategy, profile.id, pd.Series(W, index=names), w_sf, flags, info)


def equal_weight(sfs_tickers: dict[str, list[str]]) -> TwoLevelResult:
    """Benchmark 1/N: równe wagi w SF i między SF (bez limitów look-through)."""
    K = len(sfs_tickers)
    w = {k: pd.Series(1.0 / len(t), index=t) for k, t in sfs_tickers.items()}
    return TwoLevelResult("ew", "ALL", pd.Series(1.0 / K, index=list(sfs_tickers)), w)
