"""Silnik backtestu – krok modelowy dla daty rebalansowania t.

Dla każdej daty t (dane ≤ t−1):
    1. okno ``window_days`` sesji, aktywa z ≥ ``min_obs`` obserwacjami;
    2. marginesy AR(1)–GJR-GARCH(1,1) skew-t (raz na datę, pamięć podręczna), panel reszt ẑ,
       pseudo-obserwacje U = rangi / (n + 1);
    3. modele zależności z rejestru (``pf.models.dependence``) na tych samych U (pamięć podręczna);
    4. scenariusze miesięczne każdego modelu z adaptacyjną liczbą N (``simulate_adaptive``),
       ze wspólnymi wejściowymi W ~ U(0, 1) dla wszystkich modeli w dacie t.

Pamięć podręczna (``artifacts/backtest/cache/``, poza repozytorium): klucz marginesów to skrót SHA-256
danych okna i konfiguracji marginesów, klucz modeli zależności – dodatkowo konfiguracji ``rvine`` i
``dependence``. Zmiana danych spoza okna (np. dopisanie nowych sesji) nie unieważnia wyników
historycznych dat, a każda zmiana danych w oknie – tak.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pf import ARTIFACTS_DIR
from pf.backtest.data import config_digest, eligible_assets, estimation_window
from pf.models.dependence import DependenceModel, fit_model, load_model, save_model
from pf.models.margins import MarginFit, fit_margins, residual_panel
from pf.models.rvine import pseudo_observations
from pf.sim.scenarios import ScenarioSet, control_portfolios, model_seed, simulate_adaptive

log = logging.getLogger(__name__)

CACHE_DIR = ARTIFACTS_DIR / "backtest" / "cache"


def window_digest(window: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(window, index=True).to_numpy()
    cols = "|".join(map(str, window.columns)).encode("utf-8")
    return hashlib.sha256(h.tobytes() + cols).hexdigest()


@dataclass
class DateModels:
    t: pd.Timestamp
    assets: list[str]
    margins: dict[str, MarginFit]
    panel: pd.DataFrame                    # reszty ẑ (sesje × aktywa)
    U: np.ndarray                          # pseudo-obserwacje panelu
    models: dict[str, DependenceModel] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    cached: dict[str, bool] = field(default_factory=dict)

    @property
    def window_end(self) -> pd.Timestamp:
        return self.panel.index.max()


def margins_key(window: pd.DataFrame, cfg: dict) -> str:
    return config_digest({"window": window_digest(window), "margins": cfg["margins"],
                          "min_obs": cfg["rebalance"]["min_obs"]})[:16]


def dependence_key(cfg: dict) -> str:
    return config_digest({"rvine": cfg["rvine"], "dependence": cfg["dependence"]})[:12]


def fit_margins_step(t: pd.Timestamp, returns: pd.DataFrame, cfg: dict, jobs: int | None = 1,
                     cache_dir: Path | None = CACHE_DIR) -> DateModels:
    """Marginesy i pseudo-obserwacje dla daty t (wczytywane z pamięci podręcznej, jeśli istnieją)."""
    rcfg = cfg["rebalance"]
    window = estimation_window(returns, t, rcfg["window_days"])
    assets = eligible_assets(window, rcfg["min_obs"])
    window = window[assets]
    tag = pd.Timestamp(t).strftime("%Y-%m-%d")
    folder = None if cache_dir is None else Path(cache_dir) / tag / margins_key(window, cfg)
    t0 = time.time()
    if folder is not None and (folder / "margins.parquet").exists() and (folder / "panel.parquet").exists():
        rows = pd.read_parquet(folder / "margins.parquet").to_dict(orient="records")
        margins = {r["name"]: MarginFit(**r) for r in rows}
        panel = pd.read_parquet(folder / "panel.parquet")
        cached = True
    else:
        margins, z = fit_margins(window, scale=cfg["margins"]["scale"], jobs=jobs)
        panel = residual_panel(z)
        cached = False
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([f.to_dict() for f in margins.values()]).to_parquet(folder / "margins.parquet")
            panel.to_parquet(folder / "panel.parquet")
    dm = DateModels(pd.Timestamp(t), assets, margins, panel, pseudo_observations(panel.to_numpy()))
    dm.timings["margins"] = time.time() - t0
    dm.cached["margins"] = cached
    dm.folder = folder  # type: ignore[attr-defined]
    return dm


def fit_models_step(dm: DateModels, cfg: dict, names: list[str]) -> DateModels:
    """Modele zależności ``names`` na pseudo-obserwacjach daty (pamięć podręczna obok marginesów)."""
    folder = getattr(dm, "folder", None)
    dkey = dependence_key(cfg)
    for name in names:
        path = None if folder is None else folder / f"{name}_{dkey}.json.gz"
        t0 = time.time()
        if path is not None and path.exists():
            dm.models[name] = load_model(path)
            dm.cached[name] = True
        else:
            dm.models[name] = fit_model(name, dm.U, dm.assets, cfg)
            dm.cached[name] = False
            if path is not None:
                save_model(dm.models[name], path)
        dm.timings[f"fit_{name}"] = time.time() - t0
    return dm


def scenarios_step(dm: DateModels, cfg: dict, bcfg: dict, subfunds: dict, names: list[str] | None = None,
                   latest: bool = False) -> dict[str, ScenarioSet]:
    """Scenariusze miesięczne każdego modelu (adaptacyjne N, wspólne W w dacie t)."""
    rcfg, scfg = cfg["rebalance"], bcfg["scenarios"]
    lag = int(bcfg.get("execution_lag", 0))
    portfolios = control_portfolios(dm.assets, subfunds)
    margins = [dm.margins[a] for a in dm.assets]
    out = {}
    for name in names or list(dm.models):
        w_rng = np.random.default_rng(model_seed(rcfg["seed"], dm.t))
        m_rng = np.random.default_rng(model_seed(rcfg["seed"], dm.t, name))
        sset = simulate_adaptive(margins, dm.models[name], rcfg["horizon_days"], portfolios, scfg,
                                 rcfg["es_alpha"], w_rng, m_rng, skip=lag,
                                 rel_se=scfg["rel_se_last"] if latest else scfg["rel_se"])
        dm.timings[f"sim_{name}"] = sset.seconds
        out[name] = sset
    return out


def scenario_stats_table(t: pd.Timestamp, scen: dict[str, ScenarioSet]) -> pd.DataFrame:
    """N, ÊS, SE per (data, model, portfel kontrolny)."""
    rows = []
    for name, s in scen.items():
        st = s.stats.assign(date=pd.Timestamp(t), model=name, n=s.n, n_batches=s.n_batches,
                            converged=s.converged, rel_se_target=s.rel_se_target, seconds=s.seconds)
        rows.append(st)
    cols = ["date", "model", "portfolio", "n", "n_batches", "converged", "rel_se_target", "var", "es", "se",
            "rel_se", "seconds"]
    return pd.concat(rows, ignore_index=True)[cols] if rows else pd.DataFrame(columns=cols)
