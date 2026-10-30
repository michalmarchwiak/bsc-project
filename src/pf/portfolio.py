"""Sub-fundusze: skład (``config/subfunds.yaml``) i macierz look-through A.

A (K × N): A_ki – udział aktywa i w sub-funduszu k; wewnątrz sub-funduszu wagi równe
między aktywami dostępnymi w danym dniu. Ekspozycja portfela na aktywa: x = Aᵀ W.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pf import config


def load_subfunds() -> dict[str, dict]:
    return config.load("subfunds")["subfunds"]


def lookthrough_matrix(subfunds: dict[str, dict], available: list[str]) -> pd.DataFrame:
    avail = set(available)
    rows = {}
    for sf, spec in subfunds.items():
        members = [t for t in spec["tickers"] if t in avail]
        if members:
            rows[sf] = pd.Series(1.0 / len(members), index=members)
    A = pd.DataFrame(rows).T.reindex(columns=available).fillna(0.0)
    return A


def subfund_returns(asset_simple_returns: np.ndarray | pd.DataFrame, A: pd.DataFrame) -> np.ndarray:
    """Proste stopy zwrotu sub-funduszy R = R_aktywa · Aᵀ (scenariusze × K)."""
    R = asset_simple_returns.to_numpy() if isinstance(asset_simple_returns, pd.DataFrame) else asset_simple_returns
    return R @ A.to_numpy().T
