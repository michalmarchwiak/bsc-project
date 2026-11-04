"""Warunkowe scenariusze miesięczne dla nakładki opcyjnej (miara rzeczywista P).

Model w dacie t_j (dane ≤ t_j − 1, okno ``window_days`` sesji, jak w backteście):
    * marginesy AR(1)–GJR-GARCH(1,1) skew-t na log-stopach w walucie notowania (``log_returns_local``) oraz
      na log-stopach kursu USD/PLN (aktywo ``USDPLN``, kurs średni NBP); aktywa deterministyczne (SF10) – margines
      deterministyczny;
    * kopuła (domyślnie t, ``monthly_overlay.lp.dependence_model``) na panelu reszt aktywów stochastycznych
      z kursem (``pf.models.margins.residual_panel``; kurs NBP należy do kalendarza GPW);
    * margines S&P 500 (poza kopułą) – σ̂_SPX do mnożnika m_t.

Scenariusz s: ścieżka dzienna z kopuły i rekursji GARCH startującej ze stanu na koniec okna (``simulate_daily``),
suma log-stóp z sesji L+1, …, L+n_j (L – opóźnienie wykonania), czyli brutto-stopy g_i (waluta notowania)
i f (USD/PLN). Brutto-stopa w PLN: G_i = g_i·f (USD), G_i = g_i (PLN). Wypłata opcji na ticker w USD zależy
od g_i, a jej wartość w PLN – od f.

Dryf konserwatywny do decyzji P3 (``drift_adjust``): scenariusze każdego aktywa są mnożone przez stałą tak, by
E[g_i] = exp((r_i + SR·σ̂_i)τ) (stopa wolna od ryzyka + premia równowagowa SR·σ̂, SR – parametr) oraz
E[f] = exp((r_PLN − r_USD)τ) (parytet stóp procentowych). Mnożenie przez stałą zachowuje rangi, więc kopuła
i kształt rozkładów brzegowych się nie zmieniają; o decyzji decyduje premia za ryzyko zmienności
(σ_imp vs σ w mierze P) i ogon rozkładu, a nie prognoza kierunku.

Wspólne liczby losowe i odtwarzalność: ziarna z daty i znacznika (``pf.sim.scenarios.model_seed``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pf.backtest.data import config_digest, eligible_assets, estimation_window
from pf.backtest.engine import window_digest
from pf.models.dependence import DependenceModel, fit_model, load_model, save_model
from pf.models.margins import MarginFit, fit_margin, fit_margins, residual_panel, simulate_daily
from pf.models.rvine import pseudo_observations
from pf.options.market import MarketData, currency
from pf.sim.scenarios import full_uniforms, model_seed

FX = "USDPLN"
CACHE_VERSION = "overlay-v1"


def market_of(name: str) -> str:
    return "PL" if name.endswith(".WA") or name == FX else "US"


@dataclass
class OverlayModel:
    t: pd.Timestamp
    assets: list[str]                       # kolejność marginesów (aktywa + USDPLN)
    margins: dict[str, MarginFit]
    dep: DependenceModel                    # na aktywach stochastycznych (``dep.names``)
    spx: MarginFit | None
    window_end: pd.Timestamp
    timings: dict = field(default_factory=dict)
    cached: bool = False

    @property
    def stochastic(self) -> list[str]:
        return list(self.dep.names)


def model_window(market: MarketData, t: pd.Timestamp, window_days: int, tickers: list[str]) -> pd.DataFrame:
    """Okno log-stóp (waluta notowania) z kolumną USDPLN: ostatnie ``window_days`` sesji przed t."""
    data = market.r_local[[c for c in tickers if c in market.r_local.columns]].copy()
    data[FX] = market.fx_log_returns()
    return estimation_window(data, t, window_days)


def _cache_folder(cache_dir, t, window, cfg, ocfg, det) -> Path | None:
    if cache_dir is None:
        return None
    key = config_digest({"window": window_digest(window), "margins": cfg["margins"], "rebalance": cfg["rebalance"],
                         "dep": ocfg["lp"]["dependence_model"], "dependence": cfg["dependence"],
                         "deterministic": sorted(det), "version": CACHE_VERSION})[:16]
    return Path(cache_dir) / pd.Timestamp(t).strftime("%Y-%m-%d") / key


def fit_spx(market: MarketData, t: pd.Timestamp, window_days: int, min_obs: int, scale: float) -> MarginFit | None:
    if market.spx is None:
        return None
    r = np.log(market.spx.astype(float)).diff().dropna()
    r = r[r.index < pd.Timestamp(t)].iloc[-window_days:]
    if len(r) < min_obs:
        return None
    return fit_margin(r.rename("SPX"), scale)[0]


def fit_overlay_model(market: MarketData, t: pd.Timestamp, cfg: dict, ocfg: dict, tickers: list[str],
                      deterministic: list[str], cache_dir: Path | None = None, jobs: int | None = 1) -> OverlayModel:
    """Marginesy (waluta notowania + USDPLN), kopuła i margines S&P 500 w dacie t (pamięć podręczna)."""
    t = pd.Timestamp(t)
    rcfg, mcfg = cfg["rebalance"], cfg["margins"]
    window = model_window(market, t, rcfg["window_days"], tickers)
    assets = eligible_assets(window, rcfg["min_obs"], deterministic, rcfg.get("min_obs_deterministic"))
    det = [a for a in assets if a in set(deterministic)]
    window = window[assets]
    folder = _cache_folder(cache_dir, t, window, cfg, ocfg, det)
    t0 = time.time()
    name = ocfg["lp"]["dependence_model"]
    if folder is not None and (folder / "done.json").exists():
        rows = pd.read_parquet(folder / "margins.parquet").to_dict(orient="records")
        margins = {r["name"]: MarginFit(**r) for r in rows}
        dep = load_model(folder / f"{name}.json.gz")
        meta = json.loads((folder / "done.json").read_text())
        spx = MarginFit(**meta["spx"]) if meta.get("spx") else None
        model = OverlayModel(t, assets, margins, dep, spx, pd.Timestamp(meta["window_end"]), cached=True)
    else:
        margins, z = fit_margins(window, scale=mcfg["scale"], jobs=jobs, deterministic=det,
                                 n_last=mcfg.get("deterministic_sessions", 21))
        panel = residual_panel(z, market_of=market_of)
        dep = fit_model(name, pseudo_observations(panel.to_numpy()), list(panel.columns), cfg)
        spx = fit_spx(market, t, rcfg["window_days"], rcfg["min_obs"], mcfg["scale"])
        model = OverlayModel(t, assets, margins, dep, spx, window.index.max())
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([f.to_dict() for f in margins.values()]).to_parquet(folder / "margins.parquet")
            save_model(dep, folder / f"{name}.json.gz")
            (folder / "done.json").write_text(json.dumps({"window_end": str(model.window_end.date()),
                                                          "spx": spx.to_dict() if spx else None}, default=float))
    model.timings["fit"] = time.time() - t0
    return model


@dataclass
class Scenarios:
    assets: list[str]
    gross: np.ndarray                        # brutto-stopy w horyzoncie (S × d), waluta notowania (USDPLN: kurs)
    seconds: float = 0.0

    def col(self, name: str) -> np.ndarray:
        return self.gross[:, self.assets.index(name)]

    def panels(self, tickers: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(g, f, G) – (S × n) dla ``tickers``: waluta notowania, kurs USD/PLN tickera, PLN."""
        pos = {a: i for i, a in enumerate(self.assets)}
        g = self.gross[:, [pos[t] for t in tickers]]
        fx = self.gross[:, pos[FX]] if FX in pos else np.ones(len(self.gross))
        f = np.column_stack([fx if currency(t) == "USD" else np.ones(len(self.gross)) for t in tickers]) \
            if tickers else np.ones((len(self.gross), 0))
        return g, f, g * f


def simulate(model: OverlayModel, n_scenarios: int, horizon: int, skip: int, seed: int, tag: str,
             eps_cap: float | None = 2.0) -> Scenarios:
    """Brutto-stopy w horyzoncie ``horizon`` sesji po ``skip`` sesjach (wspólne W dla daty i znacznika)."""
    t0 = time.time()
    margins = [model.margins[a] for a in model.assets]
    H = int(horizon) + int(skip)
    d_s = model.dep.d
    w_rng = np.random.default_rng(model_seed(seed, model.t, f"{tag}|W"))
    m_rng = np.random.default_rng(model_seed(seed, model.t, f"{tag}|{model.dep.name}"))
    W = w_rng.random((n_scenarios * H, d_s))
    U = model.dep.simulate(n_scenarios * H, m_rng, W=W).reshape(n_scenarios, H, d_s)
    U = full_uniforms(U, margins, model.dep.names)
    daily = simulate_daily(margins, U, eps_cap)
    gross = np.exp(daily[:, skip:H, :].sum(axis=1))
    return Scenarios(list(model.assets), gross, time.time() - t0)


def drift_adjust(g: np.ndarray, f: np.ndarray, tickers: list[str], r: pd.Series, sigma_hat: pd.Series, tau: float,
                 sharpe: float, r_pln: float, r_usd: float) -> tuple[np.ndarray, np.ndarray]:
    """Skalowanie scenariuszy do E[g_i] = exp((r_i + SR·σ̂_i)τ) i E[f] = exp((r_PLN − r_USD)τ)."""
    target = np.exp((r.reindex(tickers).to_numpy(float) + sharpe * np.nan_to_num(
        sigma_hat.reindex(tickers).to_numpy(float))) * tau)
    g_adj = g * (target / g.mean(axis=0))
    f_adj = f.copy()
    usd = np.array([currency(t) == "USD" for t in tickers], dtype=bool)
    if usd.any():
        fx = f[:, usd][:, 0]
        f_adj[:, usd] = (fx * np.exp((r_pln - r_usd) * tau) / fx.mean())[:, None]
    return g_adj, f_adj
