"""Dane backtestu: daty rebalansowania, okna estymacji, okresy realizacji z opóźnieniem wykonania, run_id.

Daty rebalansowania t – pierwsze sesje miesięcy, przed którymi jest co najmniej ``min_obs`` sesji
i w których każdy sub-fundusz ma aktywo z ≥ ``min_obs`` obserwacjami w oknie (``covers_all_subfunds``).

Opóźnienie wykonania L (``execution_lag``):
    L = 1: decyzja w sesji t na danych ≤ t−1, transakcja na zamknięciu t, okres realizacji (t, t_next];
    L = 0: transakcja na zamknięciu t−1 (dane ≤ t−1), okres realizacji [t, t_next).
Ogólnie: okres realizacji obejmuje sesje o indeksach [i(t) + L, i(t_next) + L), gdzie i(·) – pozycja
sesji w kalendarzu danych; początek i koniec inwestycji to zamknięcia sesji i(t) + L − 1 i i(t_next) + L − 1.

run_id = SHA-256 kanonicznego JSON-a konfiguracji i skrótu pliku danych (pierwsze ``length`` znaków).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pf.artifacts import sha256


def load_returns(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def rebalance_dates(index: pd.DatetimeIndex, min_history: int) -> pd.DatetimeIndex:
    """Pierwsza sesja każdego miesiąca, przed którą jest co najmniej ``min_history`` sesji."""
    s = pd.Series(index, index=index)
    firsts = pd.DatetimeIndex(s.groupby(index.to_period("M")).min().to_numpy())
    pos = index.get_indexer(firsts)
    return firsts[pos >= min_history]


def estimation_window(returns: pd.DataFrame, t: pd.Timestamp, window_days: int) -> pd.DataFrame:
    """Ostatnie ``window_days`` sesji przed t (dane ≤ t−1)."""
    return returns.loc[returns.index < t].iloc[-window_days:]


def eligible_assets(window: pd.DataFrame, min_obs: int) -> list[str]:
    counts = window.notna().sum()
    return counts.index[counts >= min_obs].tolist()


def covers_all_subfunds(returns: pd.DataFrame, t: pd.Timestamp, rcfg: dict, subfunds: dict) -> bool:
    """Czy w oknie przed t każdy sub-fundusz ma co najmniej jedno aktywo z ``min_obs`` obserwacjami."""
    assets = set(eligible_assets(estimation_window(returns, t, rcfg["window_days"]), rcfg["min_obs"]))
    return all(assets & set(spec["tickers"]) for spec in subfunds.values())


def eligible_dates(returns: pd.DataFrame, rcfg: dict, subfunds: dict) -> pd.DatetimeIndex:
    """Daty rebalansowania spełniające ``min_obs`` i ``covers_all_subfunds``."""
    dates = rebalance_dates(returns.index, rcfg["min_obs"])
    return pd.DatetimeIndex([t for t in dates if covers_all_subfunds(returns, t, rcfg, subfunds)])


def in_range(dates: pd.DatetimeIndex, bounds) -> pd.DatetimeIndex:
    lo, hi = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
    return dates[(dates >= lo) & (dates <= hi)]


def sample_label(t: pd.Timestamp, bcfg: dict) -> str | None:
    """'calibration' / 'holdout' / None – próba, do której należy data rebalansowania t."""
    for name in ("calibration", "holdout"):
        lo, hi = (pd.Timestamp(x) for x in bcfg[name])
        if lo <= t <= hi:
            return name
    return None


def holding_sessions(index: pd.DatetimeIndex, t: pd.Timestamp, t_next: pd.Timestamp | None,
                     lag: int) -> pd.DatetimeIndex:
    """Sesje okresu realizacji: [i(t) + L, i(t_next) + L) w kalendarzu ``index`` (L = ``lag``)."""
    i0 = index.searchsorted(pd.Timestamp(t)) + lag
    i1 = len(index) if t_next is None else min(index.searchsorted(pd.Timestamp(t_next)) + lag, len(index))
    return index[i0:i1]


def holding_bounds(index: pd.DatetimeIndex, t: pd.Timestamp, t_next: pd.Timestamp | None,
                   lag: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Zamknięcia sesji otwarcia i zamknięcia pozycji: (i(t) + L − 1, i(t_next) + L − 1)."""
    i0 = index.searchsorted(pd.Timestamp(t)) + lag - 1
    i1 = len(index) if t_next is None else min(index.searchsorted(pd.Timestamp(t_next)) + lag, len(index))
    return index[max(i0, 0)], index[i1 - 1]


def realized_returns(returns: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp | None,
                     lag: int = 0) -> pd.Series:
    """Proste stopy zwrotu aktywów w okresie realizacji (kupno i trzymanie; brak notowań = 0).

    ``lag = 0`` – okres [t0, t1) (zachowanie pipeline'u monthly_rebalance), ``lag = 1`` – (t0, t1].
    """
    period = returns.loc[holding_sessions(returns.index, t0, t1, lag)]
    return np.expm1(period.fillna(0.0).sum())


def file_digest(path: Path) -> str:
    return sha256(Path(path))


def config_digest(obj) -> str:
    """SHA-256 kanonicznego JSON-a (posortowane klucze) dowolnej struktury konfiguracji."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_id(configs: dict, data_path: Path, length: int = 12) -> str:
    """Identyfikator przebiegu: SHA-256 konfiguracji i skrótu pliku danych (``length`` znaków)."""
    return config_digest({"configs": configs, "data_sha256": file_digest(data_path)})[:length]
