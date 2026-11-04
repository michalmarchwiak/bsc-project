"""Integracja nakładki z optymalizacją składu funduszu: warianty SFk (bez nakładki) i SFk_OV (z polityką).

Poziom 1 (skład SF) bez zmian (``pf.opt.profiles.optimize_two_level``): wagi w_k tickerów SF k dla strategii
i profilu. Poziom 2 rozszerzony o kolumny SFk_OV dla profili ≥ ``ov_min_profile``: scenariuszowa stopa SFk_OV
to dokładna stopa NAV z nakładką (``pf.options.nav.scenario_returns``) przy wagach w_k i pozycjach wybranej
polityki w t_j (h_i z decyzji SF przy wagach równych – heurystyka: decyzja nakładki nie jest optymalizowana
łącznie z w_k). Ograniczenia poziomu 2:

    W_k + W_k^OV ≤ u_k = min(W_max, c_LT / maxᵢ w_{k,i})   (wspólny limit SF i look-through, ``relaxed_upper_bounds``),
    Σ W = 1,  W ≥ 0.

Strategie: ``ce_dep`` / ``ce`` (CE_γ − κ·σ^EW·D(W) na 2K kolumnach) oraz ``maxret_es`` / ``min_es`` (LP R–U);
w ``maxret_es`` limit ES pozostaje ES portfela równych wag bazowych SF – ten sam budżet ES dla wariantu z i bez
nakładki. Porównanie przy tym samym budżecie ES ex ante (``es_matched``): portfel o wyższym ÊS jest mieszany
z lokatą po stopie wolnej od ryzyka r_f okresu, a = (ES_cel + r_f)/(ES + r_f), bo ES(aR + (1 − a) r_f) = a·ES(R) − (1 − a)·r_f.

Wariant SOCP z limitem σ nie występuje w repozytorium; wiążący jest ES ze scenariuszy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pf.opt.cvar import _solve, empirical_es, max_return_es
from pf.opt.dependence_penalty import dependence_matrix
from pf.opt.profiles import (Profile, SubfundData, TwoLevelResult, feasible_start, optimize_two_level,
                             relaxed_upper_bounds, solve_ce)
from pf.opt.utility import certainty_equivalent
from pf.options import nav

OV_SUFFIX = "_OV"


def profile_rank(pid: str) -> int:
    digits = "".join(ch for ch in str(pid) if ch.isdigit())
    return int(digits) if digits else 0


def ov_allowed(profile_id: str, ocfg: dict) -> bool:
    return profile_rank(profile_id) >= profile_rank(ocfg["portfolio"].get("ov_min_profile", "P3"))


def ov_column(sf: SubfundData, w: pd.Series, gross: dict, positions: pd.DataFrame, kappa: float) -> np.ndarray:
    """Scenariuszowa stopa SFk_OV przy wagach w_k (``gross``: ticker → (g, f) – brutto-stopy waluty i kursu)."""
    tks = list(w.index)
    g = np.column_stack([gross[t][0] for t in tks])
    f = np.column_stack([gross[t][1] for t in tks])
    return nav.scenario_returns(w.to_numpy(float), g * f, g, f, tks, positions, kappa)


def level2(R: np.ndarray, names: list[str], ub: np.ndarray, groups: list[tuple[int, int, float]], profile: Profile,
           strategy: str, alpha: float, opt: dict, measure: str, q: float, es_limit: float | None,
           kappa: float | None = None) -> tuple[np.ndarray, dict]:
    """Poziom 2 z ograniczeniami grup W_i + W_j ≤ u (``groups``: (i, j, u))."""
    K = len(names)
    A = np.zeros((len(groups), K))
    b = np.zeros(len(groups))
    for r, (i, j, u) in enumerate(groups):
        A[r, [i, j]] = 1.0
        b[r] = u
    w0 = np.zeros(K)
    base = [k for k, nm in enumerate(names) if not nm.endswith(OV_SUFFIX)]
    w0[base] = feasible_start(ub[base])
    if strategy in ("ce_dep", "ce"):
        kap = 0.0 if strategy == "ce" else (profile.kappa if kappa is None else float(kappa))
        M = dependence_matrix(R, measure, q) if kap > 0 else None
        sigma = float(R[:, base].mean(axis=1).std(ddof=1))
        return solve_ce(R, profile.gamma, kap, ub, M, sigma, opt, A, b, w0)
    if strategy == "maxret_es":
        sol = max_return_es(R, alpha, es_limit + 1e-10 * max(1.0, abs(es_limit)), ub, A, b)
        w = np.clip(sol.weights, 0.0, ub)
        return w / w.sum(), {"feasible": bool(sol.feasible), "es_limit": es_limit}
    if strategy == "min_es":
        res = _solve(R, alpha, None, ub, True, A, b)
        if res.status != 0:
            raise RuntimeError(f"LP min ES nierozwiązywalny: {res.message}")
        w = np.clip(res.x[:K], 0.0, ub)
        return w / w.sum(), {"feasible": True}
    raise ValueError(f"Nieznana strategia: {strategy}")


def optimize_with_overlay(sfs: list[SubfundData], gross: dict, positions: dict[str, pd.DataFrame],
                          profile: Profile, strategy: str, alpha: float, opt: dict, kappa_opt: float,
                          measure: str = "lambda_L", q: float = 0.05, level1_cache: dict | None = None,
                          overlay_sfs: set[str] | None = None) -> dict:
    """Wynik bazowy (K kolumn) i z nakładką (K + |OV| kolumn) dla strategii i profilu.

    ``positions``: SF → pozycje polityki w t_j; ``kappa_opt`` – koszt κ premii; zwraca słownik
    {variant: (TwoLevelResult, R_kolumn, nazwy kolumn)}."""
    base = optimize_two_level(sfs, profile, strategy, alpha, opt, measure, q, None, level1_cache)
    names = [sf.name for sf in sfs]
    R_base = np.column_stack([sf.R @ base.w[sf.name].to_numpy() for sf in sfs])
    out = {"base": (base, R_base, names)}
    ov = [sf for sf in sfs if (overlay_sfs is None or sf.name in overlay_sfs) and sf.name in positions]
    if not ov:
        return out
    m = np.array([base.w[k].max() for k in names])
    ub_base, flags = relaxed_upper_bounds(m, profile.w_max, profile.c_lt)
    cols = [R_base] + [ov_column(sf, base.w[sf.name], gross, positions[sf.name], kappa_opt)[:, None] for sf in ov]
    R = np.hstack(cols)
    names_ov = names + [sf.name + OV_SUFFIX for sf in ov]
    ub = np.concatenate([ub_base, [ub_base[names.index(sf.name)] for sf in ov]])
    groups = [(names.index(sf.name), len(names) + j, float(ub_base[names.index(sf.name)])) for j, sf in enumerate(ov)]
    es_limit = None
    if strategy == "maxret_es":
        es_limit = empirical_es(-R_base @ np.full(len(names), 1.0 / len(names)), alpha)
    W, inf2 = level2(R, names_ov, ub, groups, profile, strategy, alpha, opt, measure, q, es_limit)
    res = TwoLevelResult(strategy, profile.id, pd.Series(W, index=names_ov), dict(base.w), dict(flags),
                         {"level1": base.info.get("level1", {}), "level2": inf2})
    out["ov"] = (res, R, names_ov)
    return out


def ex_ante(R: np.ndarray, W: np.ndarray, alpha: float, gamma: float) -> dict:
    r = R @ W
    return {"exp_return": float(r.mean()), "es": float(empirical_es(-r, alpha)),
            "ce": float(certainty_equivalent(r, None, gamma))}


def es_scale(es: float, es_target: float, rf: float) -> float:
    """a ∈ (0, 1]: ES(aR + (1 − a) r_f) = ES_cel (gdy ES > ES_cel), inaczej 1."""
    if not (np.isfinite(es) and np.isfinite(es_target)) or es <= es_target or es + rf <= 0:
        return 1.0
    return float(np.clip((es_target + rf) / (es + rf), 0.0, 1.0))
