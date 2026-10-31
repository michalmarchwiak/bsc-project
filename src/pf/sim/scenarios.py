"""Estymacja modelu (marginesy + R-vine) w oknie danych i scenariusze stóp zwrotu w horyzoncie H.

Scenariusz s: kopuła R-vine generuje H dziennych wektorów U ∈ (0,1)^N (innowacje iid
w czasie), kwantyle skew-t dają innowacje z, a rekursja AR(1)–GJR-GARCH(1,1) – ścieżkę
log-stóp startującą ze stanu warunkowego na koniec okna. Wynik: proste stopy zwrotu
aktywów w horyzoncie R_i = exp(Σ_h r_ih) − 1 (PLN).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pf.backtest.data import eligible_assets  # noqa: F401 (ponowny eksport)
from pf.models.margins import MarginFit, fit_margins, residual_panel, simulate_paths
from pf.models.rvine import RVine, pseudo_observations

log = logging.getLogger(__name__)


@dataclass
class Model:
    assets: list[str]
    margins: dict[str, MarginFit]
    vine: RVine
    panel_obs: int
    end_date: pd.Timestamp


def estimate(window: pd.DataFrame, models_cfg: dict, min_obs: int, jobs: int | None = None) -> Model:
    """Marginesy i R-vine na log-stopach z ``window`` (daty × tickery)."""
    assets = eligible_assets(window, min_obs)
    excluded = sorted(set(window.columns) - set(assets))
    if excluded:
        log.info("Pominięte (mniej niż %d obserwacji): %s", min_obs, ", ".join(excluded))
    mcfg, vcfg = models_cfg["margins"], models_cfg["rvine"]
    fits, z = fit_margins(window[assets], scale=mcfg["scale"], jobs=jobs)
    panel = residual_panel(z)
    U = pseudo_observations(panel.to_numpy())
    vine = RVine.fit(U, assets, truncation=vcfg["truncation"], families=tuple(vcfg["families"]),
                     rotations=vcfg["rotations"], criterion=vcfg["criterion"],
                     indep_level=vcfg["indep_test_level"], t_df_bounds=tuple(vcfg["t_df_bounds"]))
    return Model(assets, fits, vine, len(panel), window.index.max())


def simulate(model: Model, n_scenarios: int, horizon: int, rng: np.random.Generator) -> pd.DataFrame:
    """Proste stopy zwrotu aktywów w horyzoncie (n_scenarios × N)."""
    d = len(model.assets)
    U = model.vine.simulate(n_scenarios * horizon, rng).reshape(n_scenarios, horizon, d)
    logret = simulate_paths([model.margins[a] for a in model.assets], U)
    return pd.DataFrame(np.expm1(logret), columns=model.assets)


def load_model(vine_path, margins_path, end_date: pd.Timestamp, panel_obs: int) -> Model:
    """Odtworzenie modelu zapisanego przez pipeline rebalansowania (R-vine + parametry GARCH)."""
    vine = RVine.load(vine_path)
    rows = pd.read_parquet(margins_path).to_dict(orient="records")
    margins = {r["name"]: MarginFit(**r) for r in rows}
    return Model(list(vine.names), margins, vine, int(panel_obs), pd.Timestamp(end_date))
