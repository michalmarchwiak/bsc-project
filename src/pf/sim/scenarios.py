"""Estymacja modelu (marginesy + R-vine) w oknie danych i scenariusze stóp zwrotu w horyzoncie H.

Scenariusz s: kopuła R-vine generuje H dziennych wektorów U ∈ (0,1)^N (innowacje iid
w czasie), kwantyle skew-t dają innowacje z, a rekursja AR(1)–GJR-GARCH(1,1) – ścieżkę
log-stóp startującą ze stanu warunkowego na koniec okna. Wynik: proste stopy zwrotu
aktywów w horyzoncie R_i = exp(Σ_h r_ih) − 1 (PLN).

Adaptacyjna liczba scenariuszy (``simulate_adaptive``): partie po B scenariuszy; po każdej partii
(od N ≥ n_min) dla portfeli kontrolnych k (równe wagi w każdym sub-funduszu oraz równe wagi
sub-funduszy) liczone są straty L = −Rᵀw, VaR̂_α, ÊS_α i błąd standardowy

    SE(ÊS_α) = √( [Var(L | L > VaR_α) + α (ES_α − VaR_α)²] / ((1 − α) N) ),

a symulacja kończy się, gdy max_k SE_k / |ÊS_k| ≤ ε albo N ≥ n_max.

Wspólne liczby losowe: wejściowe W ~ U(0, 1) kolejnych partii pochodzą ze strumienia zależnego
tylko od daty, więc modele przyjmujące W (vine, Gauss, niezależność) dzielą te same W, a pierwsze
N scenariuszy są identyczne niezależnie od liczby partii. Losowość specyficzna dla modelu
(χ² kopuły t, kopuła empiryczna) pochodzi ze strumienia zależnego od daty i identyfikatora modelu.
Dla opóźnienia wykonania L symulowanych jest H + L dni, a suma obejmuje dni L+1, …, L+H.
"""

from __future__ import annotations

import logging
import time
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pf.backtest.data import eligible_assets  # noqa: F401 (ponowny eksport)
from pf.models.margins import MarginFit, fit_margins, residual_panel, simulate_paths
from pf.models.rvine import RVine, pseudo_observations
from pf.opt.cvar import empirical_es

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
                     indep_level=vcfg["indep_test_level"], t_df_bounds=tuple(vcfg["t_df_bounds"]),
                     estimation=vcfg.get("estimation", "itau"))
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


# --------------------------------------------------------------------------- scenariusze adaptacyjne

def es_with_se(losses: np.ndarray, alpha: float) -> tuple[float, float, float]:
    """(VaR̂_α, ÊS_α, SE(ÊS_α)) z próby strat; SE ze wzoru asymptotycznego (ogon: straty > VaR̂)."""
    x = np.sort(np.asarray(losses, dtype=float))[::-1]
    n = len(x)
    k = (1.0 - alpha) * n
    m = max(int(np.floor(k)), 1)
    var = float(x[min(m, n - 1)]) if m < n else float(x[-1])
    es = empirical_es(x, alpha)
    tail = x[:m]
    v_tail = float(tail.var(ddof=1)) if m > 1 else 0.0
    se = float(np.sqrt((v_tail + alpha * (es - var) ** 2) / ((1.0 - alpha) * n)))
    return var, es, se


def model_seed(seed: int, t: pd.Timestamp, model: str | None = None) -> list[int]:
    """Ziarno RNG z daty (i identyfikatora modelu – stabilny CRC32 nazwy)."""
    base = [int(seed), int(pd.Timestamp(t).strftime("%Y%m%d"))]
    return base if model is None else base + [zlib.crc32(model.encode("utf-8"))]


def control_portfolios(assets: list[str], subfunds: dict[str, dict]) -> dict[str, np.ndarray]:
    """Portfele kontrolne: równe wagi w każdym sub-funduszu oraz równe wagi sub-funduszy (\"EW\")."""
    pos = {a: i for i, a in enumerate(assets)}
    out = {}
    for sf, spec in subfunds.items():
        members = [pos[a] for a in spec["tickers"] if a in pos]
        if members:
            w = np.zeros(len(assets))
            w[members] = 1.0 / len(members)
            out[sf] = w
    if out:
        out["EW"] = np.mean(list(out.values()), axis=0)
    return out


@dataclass
class ScenarioSet:
    R: np.ndarray                      # proste stopy zwrotu aktywów w horyzoncie (N × d)
    assets: list[str]
    stats: pd.DataFrame                # portfel, VaR, ES, SE, SE/|ES|
    n_batches: int
    converged: bool
    seconds: float
    rel_se_target: float
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.R)


def portfolio_stats(R: np.ndarray, portfolios: dict[str, np.ndarray], alpha: float) -> pd.DataFrame:
    rows = []
    for name, w in portfolios.items():
        var, es, se = es_with_se(-R @ w, alpha)
        rows.append({"portfolio": name, "var": var, "es": es, "se": se,
                     "rel_se": se / abs(es) if es != 0 else np.inf})
    return pd.DataFrame(rows)


def simulate_adaptive(margins: list[MarginFit], dep_model, horizon: int, portfolios: dict[str, np.ndarray],
                      scfg: dict, alpha: float, w_rng: np.random.Generator, model_rng: np.random.Generator,
                      skip: int = 0, rel_se: float | None = None) -> ScenarioSet:
    """Scenariusze w partiach do osiągnięcia max_k SE/|ES| ≤ ε (lub n_max); ``dep_model`` – DependenceModel."""
    t0 = time.time()
    eps = float(rel_se if rel_se is not None else scfg["rel_se"])
    B, n_min, n_max = int(scfg["batch"]), int(scfg["n_min"]), int(scfg["n_max"])
    d, H = len(margins), horizon + skip
    blocks, N, n_batches = [], 0, 0
    stats, converged = None, False
    while N < n_max:
        b = min(B, n_max - N)
        W = w_rng.random((b * H, d))
        U = dep_model.simulate(b * H, model_rng, W=W).reshape(b, H, d)
        blocks.append(np.expm1(simulate_paths(margins, U, skip)))
        N += b
        n_batches += 1
        if N >= n_min:
            stats = portfolio_stats(np.vstack(blocks), portfolios, alpha)
            if stats["rel_se"].max() <= eps:
                converged = True
                break
    R = np.vstack(blocks)
    if stats is None:
        stats = portfolio_stats(R, portfolios, alpha)
    return ScenarioSet(R, [m.name for m in margins], stats, n_batches, converged, time.time() - t0, eps)
