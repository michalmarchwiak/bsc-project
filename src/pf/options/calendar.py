"""Kalendarz nakładki opcyjnej: okresy opcji między kolejnymi datami rebalansowania.

Daty t_j są datami rebalansowania backtestu (pierwsze sesje miesięcy, ``pf.backtest.data``), a kalendarz
sesji to suma sesji GPW i NYSE (indeks tabeli stóp zwrotu). Przy opóźnieniu wykonania L (``execution_lag``)
pozycja jest otwierana na zamknięciu sesji i(t_j) + L − 1 i zamykana na zamknięciu sesji i(t_{j+1}) + L − 1
(``pf.backtest.data.holding_bounds``). Opcja europejska kupiona/sprzedana przy otwarciu pozycji wygasa
dokładnie przy jej zamknięciu (rozliczenie pieniężne po cenie zamknięcia), a czas do wygaśnięcia to

    τ_j = n_j / 252,   n_j = liczba sesji okresu realizacji (t_open, t_close].

Okno estymacji modeli kończy się w sesji poprzedzającej t_j (dane ≤ t_j − 1, jak w backteście); prognoza GARCH
na okres opcji obejmuje więc kroki h = L + 1, …, L + n_j od końca okna.
Ostatnia data (bez t_{j+1}) ma n_j = ``horizon_days`` i nieznane t_close (wycena bez rozliczenia).
"""

from __future__ import annotations

import pandas as pd

from pf.backtest.data import holding_bounds, holding_sessions


def overlay_periods(index: pd.DatetimeIndex, dates: pd.DatetimeIndex, lag: int, horizon_days: int = 21,
                    trading_days: int = 252) -> pd.DataFrame:
    """Tabela okresów: date, t_next, open, close, n_sessions, tau, days, window_end, skip (= L)."""
    index = pd.DatetimeIndex(index)
    dates = pd.DatetimeIndex(sorted(dates))
    rows = []
    for j, t in enumerate(dates):
        t_next = dates[j + 1] if j + 1 < len(dates) else None
        t_open, t_close = holding_bounds(index, t, t_next, lag)
        pos = index.searchsorted(t)
        window_end = index[pos - 1] if pos > 0 else pd.NaT
        if t_next is None:
            n = int(horizon_days)
            t_close = pd.NaT
            days = int(round(n * 365.0 / trading_days))
        else:
            n = len(holding_sessions(index, t, t_next, lag))
            days = int((t_close - t_open).days)
        rows.append({"date": t, "t_next": t_next if t_next is not None else pd.NaT, "open": t_open,
                     "close": t_close, "n_sessions": n, "tau": n / trading_days, "days": days,
                     "window_end": window_end, "skip": int(lag)})
    return pd.DataFrame(rows)


def sessions_of(index: pd.DatetimeIndex, period: pd.Series | dict, lag: int) -> pd.DatetimeIndex:
    """Sesje okresu realizacji (t_open, t_close] dla wiersza ``overlay_periods``."""
    t_next = period["t_next"]
    if pd.isna(t_next):
        return pd.DatetimeIndex([])
    return holding_sessions(pd.DatetimeIndex(index), period["date"], t_next, lag)
