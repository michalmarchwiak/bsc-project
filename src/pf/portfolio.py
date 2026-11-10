"""Sub-fundusze: skład (``config/subfunds.yaml``) i macierz look-through A.

Plik konfiguracji zawiera kilka uniwersów (``universes``); aktywne wskazuje klucz ``universe`` albo zmienna
środowiskowa ``PF_UNIVERSE``. SF z kluczem ``select`` to pula spółek, z której skład w dacie t wybiera
``pf.universe`` (punktowo w czasie); ``load_subfunds`` zwraca pule.

A (K × N): A_ki – udział aktywa i w sub-funduszu k; wewnątrz sub-funduszu wagi równe
między aktywami dostępnymi w danym dniu. Ekspozycja portfela na aktywa: x = Aᵀ W.

Sub-fundusz z ``deterministic: true`` (SF10, obligacje oszczędnościowe Skarbu Państwa) składa się z aktywów
o stopie zwrotu znanej ex ante w horyzoncie rebalansowania (margines deterministyczny, poza kopułą).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pf import config


def active_universe() -> str:
    """Nazwa aktywnego uniwersum: ``PF_UNIVERSE`` albo klucz ``universe`` w ``config/subfunds.yaml``."""
    return os.environ.get("PF_UNIVERSE") or config.load("subfunds")["universe"]


def load_subfunds(universe: str | None = None) -> dict[str, dict]:
    """Sub-fundusze uniwersum ``universe`` (domyślnie aktywnego); dla SF z ``select`` – pełna pula tickerów."""
    cfg = config.load("subfunds")
    name = universe or active_universe()
    if name not in cfg["universes"]:
        raise KeyError(f"Nieznane uniwersum {name!r}; dostępne: {', '.join(cfg['universes'])}")
    return cfg["universes"][name]


def deterministic_subfunds(subfunds: dict[str, dict]) -> list[str]:
    return [sf for sf, spec in subfunds.items() if spec.get("deterministic", False)]


def deterministic_tickers(subfunds: dict[str, dict]) -> list[str]:
    """Tickery sub-funduszy z ``deterministic: true`` (stopa znana ex ante)."""
    return [t for sf in deterministic_subfunds(subfunds) for t in subfunds[sf]["tickers"]]


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
