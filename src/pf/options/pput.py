"""Replikacja indeksu Cboe PPUT wyceną syntetyczną (BS + powierzchnia) i pomiar tracking error.

Metodologia PPUT (uproszczona): portfel S&P 500 (dywidendy reinwestowane) z długą pozycją
w miesięcznym pucie europejskim o cenie wykonania ≈ 95% indeksu; rolowanie w trzeci piątek
miesiąca (lub najbliższą następną sesję). W dniu rolowania put wygasający rozliczany jest
po wartości wewnętrznej, a cały majątek V dzielony na n jednostek (indeks + put):
n = V / (S + P₀). Między rolowaniami:

    V_t = n · S_{t₀} · TR_t / TR_{t₀} + n · P(S_t, K, T_t, r_t, q_t, σ_surf,t),

gdzie TR – indeks total return, P – cena BS z σ z powierzchni (``VolSurface``).
Różnice względem oficjalnego szeregu: rozliczenie SOQ (kurs otwarcia) i zakup puta
po VWAP 11:30–12:00, a nie po zamknięciu; dyskretna siatka strike'ów SPX (5 pkt).

Tracking error = odchylenie standardowe różnic dziennych logarytmicznych stóp zwrotu × √252.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pf.options.surface import SkewParams, VolSurface

STRESS_WINDOWS = {"2008": ("2008-09-01", "2009-03-31"), "2020": ("2020-02-15", "2020-04-30"),
                  "2022": ("2022-01-01", "2022-10-31")}


def third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)


def roll_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Pierwsza sesja ≥ trzeciego piątku każdego miesiąca w zakresie ``index``."""
    months = pd.period_range(index.min(), index.max(), freq="M")
    out = []
    for m in months:
        pos = index.searchsorted(third_friday(m.year, m.month))
        if pos < len(index):
            out.append(index[pos])
    return pd.DatetimeIndex(sorted(set(out)))


def replicate(spx: pd.Series, spx_tr: pd.Series, vol_indices: pd.DataFrame, params: SkewParams,
              rate: pd.Series, div_yield: pd.Series, moneyness: float = 0.95,
              strike_step: float = 5.0, tenors_days: dict | None = None) -> pd.Series:
    """Dzienny poziom zreplikowanego PPUT (start = 100)."""
    df = pd.concat({"S": spx, "TR": spx_tr, "r": rate, "q": div_yield}, axis=1).join(vol_indices)
    df = df.dropna(subset=["S", "TR", "VIX"])
    df[["r", "q"]] = df[["r", "q"]].ffill().fillna(0.0)
    rolls = roll_dates(df.index)
    df = df.loc[rolls[0]:]
    tenor_cols = [c for c in (tenors_days or {}) if c in df.columns] or ["VIX"]
    kw = {"tenors_days": tenors_days} if tenors_days else {}

    values = np.empty(len(df))
    n = K = expiry = S0 = TR0 = None
    V = 100.0
    roll_set = set(rolls)
    for i, (d, row) in enumerate(df.iterrows()):
        if n is not None:
            T = max((expiry - d).days, 0) / 365.0
            levels = {c: row[c] for c in tenor_cols if np.isfinite(row[c])}
            if "SKEW" in df.columns:
                levels["SKEW"] = row["SKEW"]
            surf = VolSurface(levels, params, **kw)
            put = float(surf.price(row["S"], K, T, row["r"], row["q"], False)) if T > 0 \
                else max(K - row["S"], 0.0)
            V = n * S0 * row["TR"] / TR0 + n * put
        if d in roll_set:
            nxt = d + pd.offsets.MonthBegin(1)
            expiry = third_friday(nxt.year, nxt.month)
            K = strike_step * np.floor(moneyness * row["S"] / strike_step)
            T = (expiry - d).days / 365.0
            levels = {c: row[c] for c in tenor_cols if np.isfinite(row[c])}
            if "SKEW" in df.columns:
                levels["SKEW"] = row["SKEW"]
            P0 = float(VolSurface(levels, params, **kw).price(row["S"], K, T, row["r"], row["q"], False))
            n, S0, TR0 = V / (row["S"] + P0), row["S"], row["TR"]
        values[i] = V
    return pd.Series(values, index=df.index, name="PPUT_replica")


def trailing_div_yield(spx: pd.Series, spx_tr: pd.Series, window: int = 252) -> pd.Series:
    """q_t z różnicy rocznych log-zwrotów indeksu total return i cenowego."""
    both = pd.concat([spx, spx_tr], axis=1).dropna()
    q = (np.log(both.iloc[:, 1]).diff(window) - np.log(both.iloc[:, 0]).diff(window)) * 252 / window
    return q.clip(lower=0.0)


def tracking_report(replica: pd.Series, official: pd.Series) -> pd.DataFrame:
    """TE roczny (kalendarzowo) i w oknach stresowych; różnica stóp zwrotu i korelacja."""
    both = pd.concat({"rep": replica, "off": official}, axis=1).dropna()
    rets = np.log(both).diff().dropna()
    diff = rets["rep"] - rets["off"]
    rows = []

    def row(label: str, mask) -> None:
        d, r = diff[mask], rets[mask]
        if len(d) < 5:
            return
        rows.append({"period": label, "n_days": len(d), "te_ann_pct": 100 * d.std() * np.sqrt(252),
                     "ret_replica_pct": 100 * (np.exp(r["rep"].sum()) - 1),
                     "ret_official_pct": 100 * (np.exp(r["off"].sum()) - 1),
                     "corr": float(r["rep"].corr(r["off"]))})

    row("all", slice(None))
    for y in sorted(set(diff.index.year)):
        row(str(y), diff.index.year == y)
    for name, (a, b) in STRESS_WINDOWS.items():
        row(f"stress_{name}", (diff.index >= a) & (diff.index <= b))
    return pd.DataFrame(rows)

