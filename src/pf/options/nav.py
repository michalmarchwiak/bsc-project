"""NAV sub-funduszu z nakładką opcyjną (opcje trzymane do wygaśnięcia, finansowanie premii z kapitału SF).

Oznaczenia dla tickera i w okresie (t_j, t_{j+1}]: w_i – wagi wewnętrzne SF (Σ w_i = 1), g_i = S_{i,t+1}/S_{i,t}
(brutto-stopa w walucie notowania), f_i – zmiana kursu USD/PLN (PLN: 1), G_i = g_i f_i (brutto-stopa w PLN).
Pozycje opcyjne tickera: puty (m^P_k, h^P_{i,k}, p_{i,k} = P/S) i calle (m^C_k, h^C_{i,k}, c_{i,k} = C/S),
Σ_k h^P_{i,k} ≤ 1, Σ_k h^C_{i,k} ≤ 1 (covered only: sprzedaż calli do wysokości posiadanego instrumentu).

Finansowanie premii z kapitału SF (V_j – NAV w t_j, κ – koszt transakcyjny jako ułamek premii):

    D_i = 1 + Σ_k h^P_{i,k} p_{i,k} (1 + κ) − Σ_k h^C_{i,k} c_{i,k} (1 − κ),     V′_j = V_j / Σ_i w_i D_i,
    n_i = w_i V′_j / S_{i,t}   (liczba jednostek instrumentu bazowego),
    V_{j+1} = Σ_i n_i [S_{i,t+1} + Σ_k h^P_{i,k} max(K^P − S_{i,t+1}, 0) − Σ_k h^C_{i,k} max(S_{i,t+1} − K^C, 0)]  (PLN),

stąd stopa okresu R = Σ_i w_i [G_i + f_i Π_i] / Σ_i w_i D_i − 1, Π_i = Σ h^P (m^P − g_i)⁺ − Σ h^C (g_i − m^C)⁺.
Przy h ≡ 0: R = Σ_i w_i G_i − 1 (NAV bazowego SF). Wartość w t_j po ustanowieniu nakładki:
akcje V′Σ w_i + opcje V′Σ w_i (Σ h^P p − Σ h^C c) + koszty V′Σ w_i κ Σ h (p + c) = V_j (samofinansowanie).

Ekspozycja skorygowana o deltę (look-through): e_i = w′_i (1 + Σ_k h^P Δ^P − Σ_k h^C Δ^C), w′_i = w_i / Σ_j w_j D_j.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POSITION_COLUMNS = ["ticker", "type", "moneyness", "h", "premium_rel"]


def empty_positions() -> pd.DataFrame:
    return pd.DataFrame(columns=POSITION_COLUMNS + ["delta"])


def _per_ticker(positions: pd.DataFrame, tickers, kappa: float) -> tuple[np.ndarray, list]:
    """D_i oraz listy (m, h, is_call) dla tickerów w kolejności ``tickers``."""
    pos = {t: i for i, t in enumerate(tickers)}
    D = np.ones(len(tickers))
    legs: list[list[tuple[float, float, bool]]] = [[] for _ in tickers]
    if positions is None or positions.empty:
        return D, legs
    for row in positions.itertuples(index=False):
        if row.ticker not in pos or not (row.h > 0):
            continue
        i = pos[row.ticker]
        call = row.type == "call"
        D[i] += -row.h * row.premium_rel * (1 - kappa) if call else row.h * row.premium_rel * (1 + kappa)
        legs[i].append((float(row.moneyness), float(row.h), call))
    return D, legs


def overlay_payoff(g, legs) -> np.ndarray:
    """Π = Σ h^P (m − g)⁺ − Σ h^C (g − m)⁺ (na jednostkę S_t; ``g`` – skalar lub tablica scenariuszy)."""
    g = np.asarray(g, dtype=float)
    out = np.zeros_like(g)
    for m, h, call in legs:
        out = out - h * np.maximum(g - m, 0.0) if call else out + h * np.maximum(m - g, 0.0)
    return out


def period_return(w: pd.Series, G: pd.Series, g: pd.Series, f: pd.Series, positions: pd.DataFrame | None,
                  kappa: float) -> dict:
    """Stopa SF z nakładką w okresie oraz składniki kosztów (ułamki V_j)."""
    tickers = list(w.index)
    wv = w.to_numpy(float)
    Gv, gv, fv = (x.reindex(tickers).to_numpy(float) for x in (G, g, f))
    D, legs = _per_ticker(positions, tickers, kappa)
    Pi = np.array([float(overlay_payoff(gv[i], legs[i])) for i in range(len(tickers))])
    denom = float(wv @ D)
    r_ov = float(wv @ (Gv + fv * Pi) / denom - 1.0)
    paid = recv = kcost = 0.0
    if positions is not None and not positions.empty:
        p = positions[positions["ticker"].isin(tickers) & (positions["h"] > 0)]
        wp = p["ticker"].map(w).to_numpy(float) * p["h"].to_numpy(float) * p["premium_rel"].to_numpy(float)
        call = (p["type"] == "call").to_numpy()
        paid = float(wp[~call].sum() * (1 + kappa) / denom)
        recv = float(wp[call].sum() * (1 - kappa) / denom)
        kcost = float(kappa * wp.sum() / denom)
    return {"r_ov": r_ov, "r_base": float(wv @ Gv - 1.0), "denominator": denom, "premium_paid": paid,
            "premium_received": recv, "kappa_cost": kcost, "payoff": float(wv @ (fv * Pi) / denom)}


def scenario_returns(w: np.ndarray, G: np.ndarray, g: np.ndarray, f: np.ndarray, tickers: list[str],
                     positions: pd.DataFrame | None, kappa: float) -> np.ndarray:
    """Stopy SF z nakładką w scenariuszach (dokładny wzór z mianownikiem finansowania); G, g, f – (S × n)."""
    D, legs = _per_ticker(positions, tickers, kappa)
    num = G.copy()
    for i, lg in enumerate(legs):
        if lg:
            num[:, i] = G[:, i] + f[:, i] * overlay_payoff(g[:, i], lg)
    return num @ w / float(w @ D) - 1.0


def establishment(V: float, w: pd.Series, positions: pd.DataFrame | None, kappa: float) -> dict:
    """Rozbicie V_j po ustanowieniu nakładki: akcje, opcje (długie − krótkie, wg premii), koszty κ."""
    tickers = list(w.index)
    D, _ = _per_ticker(positions, tickers, kappa)
    Vp = V / float(w.to_numpy(float) @ D)
    stock = Vp * float(w.sum())
    opt = cost = 0.0
    if positions is not None and not positions.empty:
        p = positions[positions["ticker"].isin(tickers) & (positions["h"] > 0)]
        wp = p["ticker"].map(w).to_numpy(float) * p["h"].to_numpy(float) * p["premium_rel"].to_numpy(float)
        sign = np.where(p["type"] == "call", -1.0, 1.0)
        opt = Vp * float((sign * wp).sum())
        cost = Vp * kappa * float(wp.sum())
    return {"V_prime": Vp, "stock": stock, "options": opt, "costs": cost, "total": stock + opt + cost}


def delta_exposure(w: pd.Series, positions: pd.DataFrame | None, kappa: float) -> pd.Series:
    """e_i = w′_i (1 + Σ h^P Δ^P − Σ h^C Δ^C), w′_i = w_i / Σ w D (Δ z kolumny ``delta`` pozycji)."""
    tickers = list(w.index)
    D, _ = _per_ticker(positions, tickers, kappa)
    wp = w / float(w.to_numpy(float) @ D)
    adj = pd.Series(1.0, index=tickers)
    if positions is not None and not positions.empty:
        p = positions[positions["ticker"].isin(tickers) & (positions["h"] > 0)]
        sign = np.where(p["type"] == "call", -1.0, 1.0)
        contrib = pd.Series(sign * p["h"].to_numpy(float) * p["delta"].to_numpy(float), index=p["ticker"])
        adj = adj.add(contrib.groupby(level=0).sum(), fill_value=0.0)
    return wp * adj.reindex(tickers)


def overlay_kind(positions: pd.DataFrame | None) -> str:
    """'none' / 'put' (ochrona) / 'covered_call' / 'collar' według obecnych pozycji h > 0."""
    if positions is None or positions.empty:
        return "none"
    p = positions[positions["h"] > 1e-9]
    has_put, has_call = (p["type"] == "put").any(), (p["type"] == "call").any()
    return "collar" if has_put and has_call else "put" if has_put else "covered_call" if has_call else "none"


def nav_from_returns(r) -> np.ndarray:
    """V_j = Π_{i<j}(1 + R_i), V_0 = 1 (długość len(r) + 1)."""
    return np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(r, dtype=float))])


def period_arrays(w: np.ndarray, G: np.ndarray, g: np.ndarray, f: np.ndarray, idx: np.ndarray, is_call: np.ndarray,
                  m: np.ndarray, h: np.ndarray, prem: np.ndarray, delta: np.ndarray, kappa: float) -> dict:
    """Wersja tablicowa ``period_return`` i ``delta_exposure`` (nogi: indeks tickera, typ, K/S, h, premia, Δ)."""
    n = len(w)
    keep = h > 0
    idx, is_call, m, h, prem, delta = (x[keep] for x in (idx, is_call, m, h, prem, delta))
    sign = np.where(is_call, -1.0, 1.0)
    D = 1.0 + np.bincount(idx, weights=np.where(is_call, -h * prem * (1 - kappa), h * prem * (1 + kappa)),
                          minlength=n)
    gi = g[idx]
    Pi = np.bincount(idx, weights=np.where(is_call, -h * np.maximum(gi - m, 0.0), h * np.maximum(m - gi, 0.0)),
                     minlength=n)
    denom = float(w @ D)
    wp = w[idx] * h * prem
    return {"r_ov": float(w @ (G + f * Pi) / denom - 1.0), "r_base": float(w @ G - 1.0), "denominator": denom,
            "premium_paid": float(wp[~is_call].sum() * (1 + kappa) / denom),
            "premium_received": float(wp[is_call].sum() * (1 - kappa) / denom),
            "kappa_cost": float(kappa * wp.sum() / denom), "payoff": float(w @ (f * Pi) / denom),
            "e": w / denom * (1.0 + np.bincount(idx, weights=sign * h * delta, minlength=n)),
            "kind": ("collar" if (~is_call).any() and is_call.any() else "put" if (~is_call).any()
                     else "covered_call" if is_call.any() else "none")}
