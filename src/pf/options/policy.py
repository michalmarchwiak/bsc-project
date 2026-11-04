"""Polityki decyzyjne nakładki opcyjnej: P0 (brak), P1 (statyczna), P2 (regułowa), P3 (optymalizacyjna, LP).

Pozycje polityki w dacie t_j to wiersze (ticker, type, moneyness, h, premium_rel, delta) wybrane z rejestru
(``pf.options.ledger``; tylko wiersze ``in_band``, puty K/S ≤ 1, calle K/S ≥ 1). Wszystkie sygnały pochodzą
z danych ≤ t_j (modele z okna ≤ t_j − 1, ceny i indeksy z zamknięcia t_j).

P1 – stały put K = 0,975·S z h^P = h̄ (wariant P1_cc: stały covered call K = 1,025·S, h^C = h̄).

P2 – reguły (progi ``rules`` ustalone ex ante):
    Q_t = percentyl m_t (cena ubezpieczenia), T_{i,t} = S_{i,t}/SMA_10M − 1 (trend), V_{i,t} = percentyl σ̂_{i,t};
    percentyle w oknie rozszerzającym się dat ≤ t_j, wymagane ≥ ``min_history_months`` wcześniejszych obserwacji
    (wcześniej sygnał nieokreślony, polityka = brak):
      put (h^P = 1):        Q ≤ q_cheap (K/S = put_cheap) lub (T < 0 i Q ≤ q_expensive) (K/S = put_trend);
      collar:               T < 0 i Q > q_expensive – put collar_put + sprzedany call collar_call;
      covered call (h^C=1): Q ≥ q_expensive, T > 0 i V ≤ v_max (K/S = covered_call);
      w pozostałych przypadkach brak nakładki.

P3 – w każdym t_j osobno dla każdego SF (wagi wewnętrzne w_i), scenariusze S z modelu GARCH-kopuła (miara P,
dryf konserwatywny, ``pf.options.scenarios.drift_adjust``), zmienne h_j ≥ 0 dla nóg j (put/call, strike z siatki):

    R⁽ˢ⁾ = Σ_i w_i [G_i⁽ˢ⁾ − 1] + Σ_{j∈P} h_j w_i [f_i⁽ˢ⁾ (m_j − g_i⁽ˢ⁾)⁺ − p_j (1 + κ)]
                               − Σ_{j∈C} h_j w_i [f_i⁽ˢ⁾ (g_i⁽ˢ⁾ − m_j)⁺ − c_j (1 − κ)],

    max_h  E[R] − λ·CVaR_α(R)   p.w.  Σ_{j∈P(i)} h_j ≤ 1,  Σ_{j∈C(i)} h_j ≤ 1,  0 ≤ h_j ≤ 1.

R⁽ˢ⁾ jest liniowe w h – przybliżenie pierwszego rzędu względem mianownika finansowania Σ w_i D_i
(``pf.options.nav``): dokładna stopa to (Σ w_i[G_i + f_iΠ_i]) / (Σ w_i D_i) − 1, a pominięty składnik jest rzędu
premia × stopa (≈ 10⁻⁴ miesięcznie). CVaR liniaryzacją Rockafellara–Uryaseva (ζ, u_s ≥ −R⁽ˢ⁾ − ζ, u_s ≥ 0),
solver HiGHS. Pasmo bezczynności: nakładka jest ustanawiana tylko, gdy poprawa funkcji celu względem h = 0
przekracza ε. Interpretacja: tylko h^P > 0 – ochrona, tylko h^C > 0 – covered call, oba – collar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, sparse

from pf.opt.cvar import empirical_es

LEG_COLUMNS = ["ticker", "type", "moneyness", "h", "premium_rel", "delta"]


def candidate_legs(ledger_t: pd.DataFrame, ocfg: dict) -> pd.DataFrame:
    """Nogi dopuszczalne w nakładce: in_band, puty K/S ≤ put_max, calle K/S ≥ call_min, premia skończona."""
    L = ledger_t[ledger_t["in_band"] & np.isfinite(ledger_t["premium_rel"])]
    puts = (L["type"] == "put") & (L["moneyness"] <= ocfg["put_max_moneyness"] + 1e-12)
    calls = (L["type"] == "call") & (L["moneyness"] >= ocfg["call_min_moneyness"] - 1e-12)
    return L[puts | calls]


def _legs_index(legs: pd.DataFrame) -> dict:
    """(ticker, type, round(K/S, 6)) → (premia względna, Δ, K/S z rejestru)."""
    return {(t, typ, round(float(m), 6)): (float(p), float(dl), float(m)) for t, typ, m, p, dl in
            zip(legs["ticker"], legs["type"], legs["moneyness"], legs["premium_rel"], legs["delta"])}


def _leg(index: dict, ticker: str, typ: str, m: float, h: float) -> dict | None:
    hit = index.get((ticker, typ, round(float(m), 6)))
    if hit is None or h <= 0:
        return None
    return {"ticker": ticker, "type": typ, "moneyness": hit[2], "h": float(h), "premium_rel": hit[0], "delta": hit[1]}


def _frame(rows: list) -> pd.DataFrame:
    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows, columns=LEG_COLUMNS) if rows else pd.DataFrame(columns=LEG_COLUMNS)


# --------------------------------------------------------------------------- P1

def static_positions(legs: pd.DataFrame, tickers: list[str], spec: dict) -> pd.DataFrame:
    legs = _legs_index(legs)
    rows = []
    for t in tickers:
        if "put" in spec:
            rows.append(_leg(legs, t, "put", spec["put"], spec.get("h_put", 1.0)))
        if "call" in spec:
            rows.append(_leg(legs, t, "call", spec["call"], spec.get("h_call", 1.0)))
    return _frame(rows)


# --------------------------------------------------------------------------- P2

def expanding_percentile(values: pd.Series, min_history: int) -> pd.Series:
    """u_t = #{s ≤ t: x_s ≤ x_t} / #{s ≤ t}; NaN, gdy wcześniejszych obserwacji < ``min_history``."""
    x = values.to_numpy(float)
    out = np.full(len(x), np.nan)
    for k in range(len(x)):
        hist = x[:k + 1]
        hist = hist[np.isfinite(hist)]
        if np.isfinite(x[k]) and len(hist) - 1 >= min_history:
            out[k] = np.mean(hist <= x[k])
    return pd.Series(out, index=values.index)


def rule_signals(state: pd.DataFrame, tick: pd.DataFrame, prices: pd.DataFrame, min_history: int,
                 sma_months: int) -> pd.DataFrame:
    """Sygnały P2 (date, ticker, Q, T, V).

    ``state``: date → m (mnożnik m_t); ``tick``: date, ticker, sigma_hat; ``prices``: dates × tickers (S_t
    w datach t_j). Kolejność dat rosnąca; sygnał w t używa wyłącznie dat ≤ t."""
    st = state.sort_values("date").set_index("date")
    Q = expanding_percentile(st["m"], min_history)
    sh = tick.pivot_table(index="date", columns="ticker", values="sigma_hat").sort_index()
    V = sh.apply(lambda c: expanding_percentile(c, min_history))
    P = prices.sort_index()
    sma = P.rolling(sma_months, min_periods=sma_months).mean()
    T = P / sma - 1.0
    out = tick[["date", "ticker"]].copy()
    out["Q"] = out["date"].map(Q)
    out["V"] = [V.at[d, t] if t in V.columns and d in V.index else np.nan for d, t in zip(out["date"], out["ticker"])]
    out["T"] = [T.at[d, t] if t in T.columns and d in T.index else np.nan for d, t in zip(out["date"], out["ticker"])]
    return out


def rule_kind(Q: float, T: float, V: float, rcfg: dict) -> str:
    """Typ nakładki P2 dla tickera: 'put' / 'collar' / 'covered_call' / 'none' (NaN → warunek fałszywy)."""
    qc, qe = rcfg["q_cheap"], rcfg["q_expensive"]
    if Q <= qc or (T < 0 and Q <= qe):
        return "put_cheap" if Q <= qc else "put"
    if T < 0 and Q > qe:
        return "collar"
    if Q >= qe and T > 0 and V <= rcfg["v_max"]:
        return "covered_call"
    return "none"


def rule_positions(legs: pd.DataFrame, signals_t: pd.DataFrame, rcfg: dict) -> pd.DataFrame:
    """Pozycje P2 w dacie t (``signals_t``: ticker, Q, T, V)."""
    legs = _legs_index(legs)
    rows = []
    for s in signals_t.itertuples(index=False):
        kind = rule_kind(s.Q, s.T, s.V, rcfg)
        if kind == "put_cheap":
            rows.append(_leg(legs, s.ticker, "put", rcfg["put_cheap"], 1.0))
        elif kind == "put":
            rows.append(_leg(legs, s.ticker, "put", rcfg["put_trend"], 1.0))
        elif kind == "collar":
            rows += [_leg(legs, s.ticker, "put", rcfg["collar_put"], 1.0),
                     _leg(legs, s.ticker, "call", rcfg["collar_call"], 1.0)]
        elif kind == "covered_call":
            rows.append(_leg(legs, s.ticker, "call", rcfg["covered_call"], 1.0))
    return _frame(rows)


# --------------------------------------------------------------------------- P3

def lp_matrix(w: np.ndarray, G: np.ndarray, g: np.ndarray, f: np.ndarray, tickers: list[str], legs: pd.DataFrame,
              kappa: float) -> tuple[np.ndarray, np.ndarray]:
    """R = R0 + A h: R0_s = Σ w_i(G_is − 1), kolumna nogi j – wkład jednostki h_j do stopy SF w scenariuszach."""
    pos = {t: i for i, t in enumerate(tickers)}
    R0 = (G - 1.0) @ w
    A = np.zeros((len(G), len(legs)))
    for j, leg in enumerate(legs.itertuples(index=False)):
        i = pos[leg.ticker]
        if leg.type == "put":
            A[:, j] = w[i] * (f[:, i] * np.maximum(leg.moneyness - g[:, i], 0.0) - leg.premium_rel * (1 + kappa))
        else:
            A[:, j] = -w[i] * (f[:, i] * np.maximum(g[:, i] - leg.moneyness, 0.0) - leg.premium_rel * (1 - kappa))
    return R0, A


def objective(R: np.ndarray, lam: float, alpha: float) -> float:
    """E[R] − λ·ES_α(−R) (ES Acerbiego–Taschego; równe CVaR z linearyzacji R–U)."""
    return float(np.mean(R) - lam * empirical_es(-R, alpha))


def solve_lp(R0: np.ndarray, A: np.ndarray, groups: list[np.ndarray], lam: float, alpha: float) -> tuple[np.ndarray, dict]:
    """max mean(R0 + A h) − λ CVaR_α  p.w. Σ_{j∈grupa} h_j ≤ 1, 0 ≤ h ≤ 1 (zmienne [h, ζ, u])."""
    S, J = A.shape
    if J == 0:
        return np.zeros(0), {"status": "no_legs", "objective": objective(R0, lam, alpha)}
    c = np.concatenate([-A.mean(axis=0), [lam], np.full(S, lam / ((1.0 - alpha) * S))])
    # −A_s h − ζ − u_s ≤ R0_s
    A_scen = sparse.hstack([sparse.csr_matrix(-A), sparse.csr_matrix(-np.ones((S, 1))), -sparse.eye(S)])
    rows, b = [A_scen], [R0]
    if groups:
        G = np.zeros((len(groups), J + 1 + S))
        for k, idx in enumerate(groups):
            G[k, idx] = 1.0
        rows.append(sparse.csr_matrix(G))
        b.append(np.ones(len(groups)))
    res = optimize.linprog(c, A_ub=sparse.vstack(rows).tocsr(), b_ub=np.concatenate(b),
                           bounds=[(0.0, 1.0)] * J + [(None, None)] + [(0.0, None)] * S, method="highs")
    if res.status != 0:
        return np.zeros(J), {"status": f"failed: {res.message}", "objective": objective(R0, lam, alpha)}
    h = np.clip(res.x[:J], 0.0, 1.0)
    h[h < 1e-7] = 0.0
    return h, {"status": "ok", "objective": float(-res.fun + R0.mean()), "zeta": float(res.x[J])}


def lp_positions(w: np.ndarray, G: np.ndarray, g: np.ndarray, f: np.ndarray, tickers: list[str], legs: pd.DataFrame,
                 kappa: float, lam: float, alpha: float, epsilon: float) -> tuple[pd.DataFrame, dict]:
    """Decyzja P3 dla jednego SF: pozycje (h > 0) i informacje (funkcja celu z i bez nakładki, poprawa)."""
    legs = legs[legs["ticker"].isin(tickers)].reset_index(drop=True)
    R0, A = lp_matrix(w, G, g, f, tickers, legs, kappa)
    groups = [np.flatnonzero(((legs["ticker"] == t) & (legs["type"] == typ)).to_numpy())
              for t in tickers for typ in ("put", "call")]
    groups = [gr for gr in groups if len(gr) > 1]
    h, info = solve_lp(R0, A, groups, lam, alpha)
    obj0 = objective(R0, lam, alpha)
    obj = objective(R0 + A @ h, lam, alpha) if len(h) else obj0
    gain = obj - obj0
    active = bool(len(h) and gain > epsilon and h.sum() > 0)
    info.update({"obj0": obj0, "obj": obj, "gain": gain, "active": active, "n_legs": int(len(legs)),
                 "lambda": float(lam)})
    if not active:
        return _frame([]), info
    sel = legs.assign(h=h)[h > 0]
    return sel[LEG_COLUMNS].reset_index(drop=True), info
