"""Ranking strategii i modeli zależności na miesięcznych stopach netto w PLN (Etap 6).

Miary (``pf.backtest.metrics``) liczone osobno dla próby kalibracyjnej i hold-out; miarą główną jest
CE zrealizowany przy γ profilu. Wnioskowanie:

* Model Confidence Set (Hansen, Lunde, Nason 2011; ``arch.bootstrap.MCS``, bootstrap stacjonarny,
  α_MCS) per profil na stracie ℓ_t = −u_γ(1 + r̃_t), u_γ(g) = g^{1−γ}/(1−γ) (γ = 1: ln g),
  1 + r̃_t = (1 + r_t)/(1 + r_f,t).
* Test różnicy wskaźników Sharpe'a względem ``ew`` (Ledoit, Wolf 2008): Δ = SR_i − SR_n na stopach
  nadwyżkowych, v = (μ_i, μ_n, γ_i, γ_n), γ = E[x²], Δ = f(v), SE(Δ̂) = √(∇fᵀ Ψ̂ ∇f / T); Ψ̂ – estymator
  blokowy b·(1/l) Σ_j ζ̄_j ζ̄_jᵀ (ζ̄_j – średnia odchyleń y_t w j-tym bloku długości b);
  p-wartość ze studentyzowanego bootstrapu blokowego cyklicznego:
  p = (#{|Δ* − Δ̂|/SE* ≥ |Δ̂|/SE} + 1)/(B + 1).
* Ranga złożona: średnia ważona rang miar (1 – najlepsza) z wagami ``ranking.composite_weights``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pf.backtest import metrics

HIGHER_BETTER = {"ce_ann": True, "sharpe": True, "sortino": True, "calmar": True, "max_dd": True, "omega": True,
                 "starr": True, "ann_return": True, "ann_vol": False, "turnover": False, "cost_ann": False,
                 "es_expost": False}


def crra_loss(r, rf, gamma: float) -> np.ndarray:
    g = (1.0 + np.asarray(r, dtype=float)) / (1.0 + np.asarray(rf, dtype=float))
    if abs(gamma - 1.0) < 1e-12:
        return -np.log(g)
    return -(g ** (1.0 - gamma)) / (1.0 - gamma)


def mcs(losses: pd.DataFrame, size: float, reps: int, seed: int = 0, block: int | None = None) -> pd.DataFrame:
    from arch.bootstrap import MCS

    m = MCS(losses, size=size, reps=reps, block_size=block, method="R", bootstrap="stationary", seed=seed)
    m.compute()
    pv = m.pvalues.iloc[:, 0]
    return pd.DataFrame({"mcs_pvalue": pv, "in_mcs": pv.index.isin(m.included)}, index=pv.index)


# --------------------------------------------------------------------------- test Ledoita–Wolfa

def _delta_and_grad(v):
    mi, mn, gi, gn = v
    si, sn = gi - mi * mi, gn - mn * mn
    d = mi / np.sqrt(si) - mn / np.sqrt(sn)
    grad = np.array([gi / si ** 1.5, -gn / sn ** 1.5, -0.5 * mi / si ** 1.5, 0.5 * mn / sn ** 1.5])
    return d, grad


def _block_psi(y: np.ndarray, b: int) -> np.ndarray:
    T = len(y)
    l = T // b
    yc = y[: l * b] - y[: l * b].mean(axis=0)
    zeta = yc.reshape(l, b, -1).sum(axis=1) / np.sqrt(b)
    return zeta.T @ zeta / l


def sharpe_diff_test(xi: np.ndarray, xn: np.ndarray, block: int = 3, reps: int = 5000, seed: int = 0) -> dict:
    """Ledoit–Wolf (2008): H0 SR_i = SR_n na stopach nadwyżkowych (miesięczne), bootstrap blokowy cykliczny."""
    xi, xn = np.asarray(xi, dtype=float), np.asarray(xn, dtype=float)
    T = len(xi)
    Y = np.column_stack([xi, xn, xi * xi, xn * xn])
    v = Y.mean(axis=0)
    d, g = _delta_and_grad(v)
    se = np.sqrt(g @ _block_psi(Y, block) @ g / T)
    rng = np.random.default_rng(seed)
    l = int(np.ceil(T / block))
    stat0 = abs(d) / se
    count = 0
    for _ in range(reps):
        starts = rng.integers(0, T, l)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % T
        Yb = Y[idx[:T]]
        db, gb = _delta_and_grad(Yb.mean(axis=0))
        seb = np.sqrt(gb @ _block_psi(Yb, block) @ gb / T)
        count += abs(db - d) / seb >= stat0
    return {"sr_diff_m": float(d), "sr_diff_ann": float(d * np.sqrt(12)), "sr_diff_se_m": float(se),
            "p_sharpe_vs_ew": float((count + 1) / (reps + 1))}


# --------------------------------------------------------------------------- tabele

def series_metrics(periods: pd.DataFrame, gammas: dict[str, float], alpha: float,
                   es_expost: pd.Series | None = None) -> pd.DataFrame:
    """Miary na miesięcznych stopach netto dla każdego (sid, próba); γ z profilu (ew: wszystkie profile)."""
    rows = []
    for (sid, smp), g in periods.groupby(["sid", "sample"]):
        g = g.sort_values("date")
        prof = g["profile"].iloc[0]
        r, rf = g["r_net"].to_numpy(), g["rf"].to_numpy()
        ese = es_expost.get((sid, smp), np.nan) if es_expost is not None else None
        base = metrics.summary(r, rf, None, alpha, ese)
        base.update({"sid": sid, "sample": smp, "model": g["model"].iloc[0], "strategy": g["strategy"].iloc[0],
                     "profile": prof, "variant": g["variant"].iloc[0], "turnover": g["turnover"].iloc[1:].mean()
                     if len(g) > 1 else np.nan, "cost_ann": 12 * g["cost"].mean(), "es_expost": ese})
        for p, gam in gammas.items():
            if prof in (p, "ALL"):
                ce = metrics.realized_ce(r, rf, gam)
                rows.append({**base, "eval_profile": p, "gamma": gam, "ce_m": ce, "ce_ann": metrics.annualize_ce(ce)})
    return pd.DataFrame(rows)


def add_ranks(table: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = table.copy()
    for col, hb in HIGHER_BETTER.items():
        if col in out.columns and out[col].notna().any():
            out[f"rank_{col}"] = out[col].rank(ascending=not hb, method="min")
    wsum, acc = 0.0, 0.0
    for col, w in weights.items():
        rc = f"rank_{col}"
        if rc in out.columns and w > 0:
            acc = acc + w * out[rc].fillna(out[rc].max())
            wsum += w
    out["rank_composite"] = acc / wsum if wsum else np.nan
    out["rank_composite_order"] = out["rank_composite"].rank(method="min")
    return out


def monotone_kappa(ce: pd.DataFrame, order: list[str]) -> dict[str, float]:
    """Wybór κ per profil maksymalizujący Σ CE przy κ_P1 ≥ κ_P2 ≥ … (programowanie dynamiczne)."""
    kappas = sorted(ce.columns, reverse=True)
    best = {k: (ce.loc[order[0], k], [k]) for k in kappas}
    for p in order[1:]:
        nxt = {}
        for k in kappas:
            cands = [(v + ce.loc[p, k], path + [k]) for kk, (v, path) in best.items() if kk >= k]
            nxt[k] = max(cands, key=lambda c: c[0])
        best = nxt
    val, path = max(best.values(), key=lambda c: c[0])
    return dict(zip(order, path))
