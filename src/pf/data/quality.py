"""Kontrola kompletności danych: raport pokrycia (``coverage.csv``) i tabela log-stóp zwrotu.

Wymagania dla próby [start, end] z tolerancją ``tol`` dni kalendarzowych:
    * każdy ticker ma źródło (brak wpisów MISSING) i co najmniej jedną obserwację;
    * ostatnia obserwacja ≥ end − tol (szereg nie urywa się przed końcem próby);
    * tabela stóp nie ma pustych kolumn, a jej indeks obejmuje [start + tol, end − tol].
Późniejszy początek notowań (ETF-y uruchomione po ``start``) jest raportowany, ale nie jest błędem.
"""

from __future__ import annotations

import pandas as pd


def coverage_report(coverage: pd.DataFrame, start: str, end: str, tol_days: int = 10) -> dict:
    cov = coverage.copy()
    first = pd.to_datetime(cov["first_date"], errors="coerce")
    last = pd.to_datetime(cov["last_date"], errors="coerce")
    missing = cov.loc[(cov["source"] == "MISSING") | (cov["n_obs"].fillna(0) <= 0), "ticker"].tolist()
    stale = cov.loc[last < pd.Timestamp(end) - pd.Timedelta(days=tol_days), "ticker"].tolist()
    late = cov.loc[first > pd.Timestamp(start) + pd.Timedelta(days=tol_days), ["ticker", "first_date"]]
    return {"ok": not missing and not stale, "missing": missing, "stale": sorted(set(stale) - set(missing)),
            "late_start": {r.ticker: str(r.first_date) for r in late.itertuples()},
            "sources": cov["source"].value_counts().to_dict()}


def returns_report(returns: pd.DataFrame, tickers: list[str], start: str, end: str, tol_days: int = 10) -> dict:
    empty = returns.columns[returns.notna().sum() == 0].tolist()
    absent = [t for t in tickers if t not in returns.columns]
    lo, hi = returns.index.min(), returns.index.max()
    covers = (lo <= pd.Timestamp(start) + pd.Timedelta(days=tol_days)
              and hi >= pd.Timestamp(end) - pd.Timedelta(days=tol_days))
    return {"ok": not empty and not absent and bool(covers), "empty_columns": empty, "absent": absent,
            "first_date": str(lo.date()), "last_date": str(hi.date()), "covers_range": bool(covers),
            "n_days": int(len(returns)), "n_columns": int(returns.shape[1])}
