"""Zrealizowane stopy zwrotu szeregów portfeli: kupno i trzymanie z dryfem wag, koszty transakcyjne, PIT.

Szereg s ma w dacie rebalansowania t wagi look-through x_t (tickery) i opóźnienie L. W okresie realizacji
(sesje h = 1, …, T_t, ``pf.backtest.data.holding_sessions``) wartość portfela

    V_h = Σᵢ x_{t,i} Π_{j ≤ h} (1 + r_{i,j}),     r^gross_h = V_h / V_{h−1} − 1,    r^gross_t = V_{T_t} − 1,

a wagi po dryfie x̃_{t+1,i} = x_{t,i} Π_j (1 + r_{i,j}) / V_{T_t}. Koszt w dniu rebalansowania
k_t = c·Σᵢ |x_{t,i} − x̃_{t,i}| (zakup początkowy: k = c), stopa netto okresu (1 − k_t)(1 + r^gross_t) − 1,
a dzienna – koszt w pierwszej sesji okresu. Brak notowania aktywa w sesji → r = 0.
Sprzedaż obligacji oszczędnościowych (SF10) przed wykupem obciąża dodatkowo opłata za wcześniejszy wykup
f_i(t) (ułamek wartości, ``pf.data.savings_bonds``): k_t = c·Σᵢ |x_{t,i} − x̃_{t,i}| + Σᵢ f_i(t)·(x̃_{t,i} − x_{t,i})⁺,
f_i(t) z zamknięcia sesji transakcji.
Stopa wolna od ryzyka okresu: r_f(t0, t1) = WIBOR1M(t0)·(t1 − t0)/365, t0, t1 – zamknięcia otwarcia i
zamknięcia pozycji.

PIT zrealizowanej straty L_t = −r^gross_t względem prognozy ex ante F̂_t (rozkład scenariuszowy):
u_t = F̂_t(L_t) = 1 − #{scenariusze ze stratą > L_t}/N – dokładnie, gdy L_t ≥ najmniejszej straty ogona
(najgorsze ⌈(1−α)N⌉ strat), w przeciwnym razie z interpolacji siatki kwantyli.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pf.backtest.data import holding_bounds, holding_sessions
from pf.data.riskfree import period_rate

KEY_COLS = ["model", "strategy", "profile", "variant"]


def collect(run_dir: Path, name: str) -> pd.DataFrame:
    files = sorted((Path(run_dir) / "dates").glob(f"*/{name}.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_tails(run_dir: Path) -> pd.DataFrame:
    files = sorted((Path(run_dir) / "tails").glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def add_sid(df: pd.DataFrame, extra: str | None = None) -> pd.DataFrame:
    sid = df["model"] + "|" + df["strategy"] + "|" + df["profile"] + "|" + df["variant"]
    if extra is not None:
        sid = sid + "|" + df[extra]
    return df.assign(sid=sid)


def sell_cost_at(sell_cost: pd.DataFrame | None, t: pd.Timestamp) -> pd.Series:
    """f_i(t): ostatni wiersz tabeli opłat za wcześniejszy wykup z datą ≤ t (brak – pusta seria)."""
    if sell_cost is None or sell_cost.empty:
        return pd.Series(dtype=float)
    part = sell_cost.loc[sell_cost.index <= pd.Timestamp(t)]
    return part.iloc[-1].fillna(0.0) if len(part) else pd.Series(dtype=float)


def realize(returns: pd.DataFrame, weights: pd.DataFrame, lags: pd.Series, dates: pd.DatetimeIndex,
            cost: float, rf: pd.Series | None = None, day_count: int = 365,
            sell_cost: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dzienne i okresowe stopy szeregów (``weights``: date, sid, ticker, weight; ``lags``: sid → L).

    ``sell_cost`` – opłaty za wcześniejszy wykup f_i(t) (sesje × tickery, ułamek wartości sprzedanej pozycji).
    Zwraca (daily: session, date, sid, r_gross, r_net; periods: date, end, sid, lag, r_gross, r_net, cost,
    redemption_cost, traded, turnover, rf, n_sessions). Ostatnia data (bez t_next) nie jest realizowana.
    """
    dates = pd.DatetimeIndex(sorted(dates))
    simple = np.expm1(returns)
    by_date = {t: g for t, g in weights.groupby("date")}
    prev_drift: dict[str, pd.Series] = {}
    daily_parts, period_rows = [], []
    for i, t in enumerate(dates[:-1]):
        t_next = dates[i + 1]
        g = by_date.get(t)
        if g is None:
            continue
        new_drift = {}
        g = g.assign(lag=g["sid"].map(lags))
        for L, gl in g.groupby("lag"):
            L = int(L)
            sessions = holding_sessions(returns.index, t, t_next, L)
            if len(sessions) == 0:
                continue
            X = gl.pivot_table(index="sid", columns="ticker", values="weight", aggfunc="sum", fill_value=0.0)
            Rd = simple.loc[sessions, X.columns].fillna(0.0).to_numpy()
            G = np.cumprod(1.0 + Rd, axis=0)                       # (T × n)
            V = X.to_numpy() @ G.T                                  # (S × T)
            V0 = X.to_numpy().sum(axis=1, keepdims=True)
            prev_v = np.concatenate([V0, V[:, :-1]], axis=1)
            r_daily = V / prev_v - 1.0
            gross = V[:, -1] / V0[:, 0] - 1.0
            drift = X.to_numpy() * G[-1] / V[:, -1:]
            t0, t1 = holding_bounds(returns.index, t, t_next, L)
            rf_p = period_rate(rf, t0, t1, day_count) if rf is not None else 0.0
            fee = sell_cost_at(sell_cost, t0)
            costs = np.zeros(len(X))
            for j, sid in enumerate(X.index):
                x = X.iloc[j]
                x = x[x > 0]
                prev = prev_drift.get(sid)
                redemption = 0.0
                if prev is None:
                    traded = float(x.sum())
                else:
                    idx = x.index.union(prev.index)
                    diff = x.reindex(idx, fill_value=0.0) - prev.reindex(idx, fill_value=0.0)
                    traded = float(diff.abs().sum())
                    if len(fee):
                        redemption = float(((-diff).clip(lower=0.0) * fee.reindex(idx).fillna(0.0)).sum())
                k = cost * traded + redemption
                costs[j] = k
                new_drift[sid] = pd.Series(drift[j], index=X.columns)[lambda s: s > 0]
                net = (1.0 - k) * (1.0 + gross[j]) - 1.0
                period_rows.append({"date": t, "end": t_next, "sid": sid, "lag": L, "r_gross": gross[j],
                                    "r_net": net, "cost": k, "redemption_cost": redemption, "traded": traded,
                                    "turnover": 0.5 * traded,
                                    "rf": rf_p, "n_sessions": len(sessions)})
            rn = r_daily.copy()
            rn[:, 0] = (1.0 - costs) * (1.0 + rn[:, 0]) - 1.0
            daily_parts.append(pd.DataFrame({
                "session": np.tile(sessions.to_numpy(), len(X)), "date": t,
                "sid": np.repeat(X.index.to_numpy(), len(sessions)),
                "r_gross": r_daily.ravel().astype(np.float32), "r_net": rn.ravel().astype(np.float32)}))
        prev_drift = new_drift
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    return daily, pd.DataFrame(period_rows)


def pit(loss: float, n: int, tail: np.ndarray, quantiles: np.ndarray, grid: np.ndarray) -> float:
    """u = F̂(L): dokładnie z ogona, gdy L ≥ min(ogon); inaczej interpolacja siatki kwantyli."""
    if not np.isfinite(loss):
        return np.nan
    tail = np.asarray(tail, dtype=float)
    if len(tail) and loss >= tail.min():
        return 1.0 - np.sum(tail > loss) / n
    q = np.asarray(quantiles, dtype=float)
    if loss <= q[0]:
        return float(0.5 * grid[0])      # poniżej najniższego kwantyla siatki
    return float(np.interp(loss, q, grid))


def evaluate_forecasts(fc: pd.DataFrame, periods: pd.DataFrame, tails: pd.DataFrame, quantiles: pd.DataFrame,
                       grid: np.ndarray) -> pd.DataFrame:
    """Prognozy z zrealizowaną stratą L_t = −r^gross_t, przekroczeniem VaR i PIT."""
    fc = add_sid(fc)
    out = fc.merge(periods[["date", "sid", "end", "r_gross", "r_net", "rf"]], on=["date", "sid"], how="left")
    out["loss"] = -out["r_gross"]
    out["hit"] = (out["loss"] > out["var"]).where(out["loss"].notna())
    t = add_sid(tails)[["date", "sid", "tail"]]
    q = add_sid(quantiles)[["date", "sid", "quantiles"]]
    out = out.merge(t, on=["date", "sid"], how="left").merge(q, on=["date", "sid"], how="left")
    out["pit"] = [pit(L, n, tl, qq, grid) if np.isfinite(L) else np.nan
                  for L, n, tl, qq in zip(out["loss"], out["n"], out["tail"], out["quantiles"])]
    return out


def realize_overlay(returns: pd.DataFrame, returns_local: pd.DataFrame, weights: pd.DataFrame, legs: pd.DataFrame,
                    lags: pd.Series, dates: pd.DatetimeIndex, cost: float, rf: pd.Series | None = None,
                    day_count: int = 365, sell_cost: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zrealizowane stopy szeregów z kolumnami SFk_OV (nakładka opcyjna, ``pf.options.portfolio``).

    W dacie t: x_i – ekspozycja akcyjna (Σ_k W_k w_ki + Σ_k W_k^OV w_ki / D_k), B₀ = Σ_k W_k^OV (1 − 1/D_k) – koszt
    nakładki (premie netto i κ), Σ x + B₀ = 1. Opcje nie są przeszacowywane w trakcie miesiąca (wartość księgowa B₀),
    a w ostatniej sesji okresu zastępuje ją wypłata Σ_j a_j f_i [(m_j − g_i)⁺ lub −(g_i − m_j)⁺], a_j = W_k^OV w_ki h_j / D_k,
    g_i – brutto-stopa w walucie notowania (``returns_local``), f_i = G_i / g_i – zmiana kursu USD/PLN tickera:

        V_h = Σ_i x_i Π_{s ≤ h}(1 + r_is) + B₀  (h < T),    V_T = Σ_i x_i G_i + Σ_j a_j f_i Π_j.

    Koszt obrotu c·Σ_i |x_{t,i} − x̃_{t,i}| na ekspozycji akcyjnej (x̃ – akcje po dryfie poprzedniego okresu; wypłaty opcji
    są gotówką reinwestowaną przy rebalansowaniu), opłata za wcześniejszy wykup obligacji SF10 jak w ``realize``."""
    dates = pd.DatetimeIndex(sorted(dates))
    simple = np.expm1(returns)
    by_date = {t: g for t, g in weights.groupby("date")}
    legs_by = {k: g for k, g in legs.groupby(["date", "sid"])} if len(legs) else {}
    prev_drift: dict[str, pd.Series] = {}
    daily_parts, period_rows = [], []
    for i, t in enumerate(dates[:-1]):
        t_next = dates[i + 1]
        g = by_date.get(t)
        if g is None:
            continue
        new_drift = {}
        for sid, gs in g.groupby("sid"):
            L = int(lags[sid])
            sessions = holding_sessions(returns.index, t, t_next, L)
            if len(sessions) == 0:
                continue
            x = gs.groupby("ticker")["weight"].sum()
            x = x[x > 0]
            Rd = simple.loc[sessions, x.index].fillna(0.0).to_numpy()
            Gc = np.cumprod(1.0 + Rd, axis=0)                         # (T × n)
            lg = legs_by.get((t, sid))
            B0, payoff = 0.0, 0.0
            if lg is not None and len(lg):
                per_sf = lg.drop_duplicates("subfund")
                B0 = float((per_sf["W_ov"] * (1.0 - 1.0 / per_sf["D"])).sum())
                tk = list(lg["ticker"].unique())
                G_T = np.exp(returns.loc[sessions, tk].fillna(0.0).sum())
                g_T = np.exp(returns_local.reindex(index=sessions, columns=tk).fillna(0.0).sum())
                f_T = G_T / g_T
                a = lg["W_ov"] * lg["w"] * lg["h"] / lg["D"]
                gi, fi, m = lg["ticker"].map(g_T).to_numpy(), lg["ticker"].map(f_T).to_numpy(), lg["moneyness"].to_numpy()
                call = (lg["type"] == "call").to_numpy()
                pay = np.where(call, -np.maximum(gi - m, 0.0), np.maximum(m - gi, 0.0))
                payoff = float((a.to_numpy() * fi * pay).sum())
            V = Gc @ x.to_numpy() + B0
            V[-1] = V[-1] - B0 + payoff
            V0 = float(x.sum()) + B0
            gross = V[-1] / V0 - 1.0
            prev_v = np.concatenate([[V0], V[:-1]])
            r_daily = V / prev_v - 1.0
            t0, t1 = holding_bounds(returns.index, t, t_next, L)
            rf_p = period_rate(rf, t0, t1, day_count) if rf is not None else 0.0
            fee = sell_cost_at(sell_cost, t0)
            prev = prev_drift.get(sid)
            redemption = 0.0
            if prev is None:
                traded = float(x.sum())
            else:
                idx = x.index.union(prev.index)
                diff = x.reindex(idx, fill_value=0.0) - prev.reindex(idx, fill_value=0.0)
                traded = float(diff.abs().sum())
                if len(fee):
                    redemption = float(((-diff).clip(lower=0.0) * fee.reindex(idx).fillna(0.0)).sum())
            k = cost * traded + redemption
            new_drift[sid] = pd.Series(x.to_numpy() * Gc[-1] / V[-1], index=x.index)[lambda s: s > 0]
            net = (1.0 - k) * (1.0 + gross) - 1.0
            period_rows.append({"date": t, "end": t_next, "sid": sid, "lag": L, "r_gross": gross, "r_net": net,
                                "cost": k, "redemption_cost": redemption, "traded": traded, "turnover": 0.5 * traded,
                                "rf": rf_p, "n_sessions": len(sessions), "overlay_cost": B0, "overlay_payoff": payoff})
            rn = r_daily.copy()
            rn[0] = (1.0 - k) * (1.0 + rn[0]) - 1.0
            daily_parts.append(pd.DataFrame({"session": sessions.to_numpy(), "date": t, "sid": sid,
                                             "r_gross": r_daily.astype(np.float32), "r_net": rn.astype(np.float32)}))
        prev_drift = new_drift
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    return daily, pd.DataFrame(period_rows)
