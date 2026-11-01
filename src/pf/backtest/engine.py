"""Silnik backtestu – krok daty rebalansowania t: modele, scenariusze, optymalizacja, prognozy.

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
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pf import ARTIFACTS_DIR
from pf.artifacts import write_parquet
from pf.backtest.data import config_digest, eligible_assets, estimation_window, sample_label
from pf.models.dependence import DependenceModel, fit_model, load_model, save_model
from pf.models.gof import compare_with_reference, oos_log_score, pit_panel
from pf.models.margins import MarginFit, fit_margins, residual_panel
from pf.models.rvine import pseudo_observations
from pf.opt.mv import mv_two_level
from pf.opt.profiles import TwoLevelResult, equal_weight, load_profiles, optimize_two_level, subfund_data
from pf.opt.utility import certainty_equivalent
from pf.sim.scenarios import ScenarioSet, control_portfolios, es_with_se, model_seed, simulate_adaptive

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


PANEL_VERSION = "trim_leading-v2+eps_absmax"   # zmiana panelu/marginesów unieważnia pamięć podręczną


def margins_key(window: pd.DataFrame, cfg: dict) -> str:
    return config_digest({"window": window_digest(window), "margins": cfg["margins"],
                          "min_obs": cfg["rebalance"]["min_obs"], "panel": PANEL_VERSION})[:16]


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
                   latest: bool = False, extra_lags: tuple[int, ...] = ()) -> dict[str, ScenarioSet]:
    """Scenariusze miesięczne każdego modelu (adaptacyjne N, wspólne W w dacie t)."""
    return {name: simulate_model(dm, name, cfg, bcfg, subfunds, latest, extra_lags) for name in names or list(dm.models)}


def simulate_model(dm: DateModels, name: str, cfg: dict, bcfg: dict, subfunds: dict, latest: bool = False,
                   extra_lags: tuple[int, ...] = ()) -> ScenarioSet:
    rcfg, scfg = cfg["rebalance"], bcfg["scenarios"]
    lag = int(bcfg.get("execution_lag", 0))
    portfolios = control_portfolios(dm.assets, subfunds)
    margins = [dm.margins[a] for a in dm.assets]
    w_rng = np.random.default_rng(model_seed(rcfg["seed"], dm.t))
    m_rng = np.random.default_rng(model_seed(rcfg["seed"], dm.t, name))
    sset = simulate_adaptive(margins, dm.models[name], rcfg["horizon_days"], portfolios, scfg,
                             rcfg["es_alpha"], w_rng, m_rng, skip=lag,
                             rel_se=scfg["rel_se_last"] if latest else scfg["rel_se"], extra_skips=extra_lags,
                             eps_cap=cfg["margins"].get("sim_eps_cap"))
    dm.timings[f"sim_{name}"] = sset.seconds
    return sset


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


# =========================================================================== krok daty (Etap 4)
#
# Dla daty t: modele zależności, scenariusze (adaptacyjne N), optymalizacja wszystkich strategii, profili
# i wariantów (analiza wrażliwości tylko w próbie kalibracyjnej), benchmarki, prognozy ex ante
# (VaR, ES, kwantyle i ogon rozkładu strat portfela), ocena modeli (AIC/BIC/Vuong) i log-score OOS.
# Wyniki zapisywane są w <run_dir>/dates/<data>/ (plik done.json kończy datę – przebieg wznawialny).


KEY_COLS = ["model", "strategy", "profile", "variant"]
BENCH_MODEL = "-"


def series_id(model: str, strategy: str, profile: str, variant: str) -> str:
    return f"{model}|{strategy}|{profile}|{variant}"


def variant_specs(t: pd.Timestamp, bcfg: dict, pcfg: dict) -> list[dict]:
    """Warianty optymalizacji w dacie t: bazowy oraz (próba kalibracyjna) warianty wrażliwości."""
    base_lag = int(bcfg["execution_lag"])
    base = {"variant": "base", "lag": base_lag, "measure": pcfg["dependence_measure"], "q": float(pcfg["tail_q"]),
            "kappa": None, "strategies": list(bcfg["grid"]["strategies"]), "benchmarks": True}
    out = [base]
    sens = bcfg.get("sensitivity", {})
    if sens.get("enabled") and sample_label(pd.Timestamp(t), bcfg) == sens.get("sample", "calibration"):
        for name, v in sens.get("variants", {}).items():
            out.append({**base, "variant": name, "lag": int(v.get("execution_lag", base_lag)),
                        "measure": v.get("dependence_measure", base["measure"]),
                        "q": float(v.get("tail_q", base["q"])), "strategies": list(v.get("strategies", [])),
                        "benchmarks": bool(v.get("benchmarks", False))})
        kg = sens.get("kappa_grid")
        if kg:
            for k in kg["values"]:
                out.append({**base, "variant": f"kappa={float(k):g}", "kappa": float(k),
                            "strategies": list(kg["strategies"]), "benchmarks": False})
    return out


def loss_forecast(losses: np.ndarray, alpha: float, grid: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Statystyki rozkładu strat L = −Rᵀx: VaR, ES, SE(ES), E[L]; ogon (najgorsze ⌈(1−α)N⌉ strat) i kwantyle."""
    losses = np.asarray(losses, dtype=float)
    var, es, se = es_with_se(losses, alpha)
    m = max(int(np.ceil((1.0 - alpha) * len(losses))), 1)
    tail = -np.sort(-np.partition(losses, len(losses) - m)[-m:])
    return ({"n": int(len(losses)), "var": var, "es": es, "es_se": se, "mean_loss": float(losses.mean())},
            tail.astype(np.float32), np.quantile(losses, grid).astype(np.float32))


def _weights_rows(t, key: tuple, res: TwoLevelResult) -> list[dict]:
    rows = []
    for k, wv in res.W.items():
        rows.append({"date": t, **dict(zip(KEY_COLS, key)), "level": "W", "subfund": k, "ticker": "", "weight": wv})
    for k, w in res.w.items():
        for a, wv in w.items():
            if wv > 1e-12:
                rows.append({"date": t, **dict(zip(KEY_COLS, key)), "level": "sf", "subfund": k, "ticker": a,
                             "weight": wv})
    for a, wv in res.x.items():
        if wv > 1e-12:
            rows.append({"date": t, **dict(zip(KEY_COLS, key)), "level": "x", "subfund": "", "ticker": a,
                         "weight": wv})
    return rows


def run_date(t: pd.Timestamp, t_next: pd.Timestamp | None, returns: pd.DataFrame, cfg: dict, bcfg: dict,
             pcfg: dict, subfunds_all: dict, run_dir: Path, cache_dir: Path | None = CACHE_DIR,
             latest: bool = False, force: bool = False) -> dict:
    """Pełny krok daty t; zwraca podsumowanie (czasy, N). Wyniki: <run_dir>/dates/<t>/."""
    t = pd.Timestamp(t)
    tag = t.strftime("%Y-%m-%d")
    out = Path(run_dir) / "dates" / tag
    if (out / "done.json").exists() and not force:
        return json.loads((out / "done.json").read_text())
    t_start = time.time()
    rcfg = cfg["rebalance"]
    alpha = float(rcfg["es_alpha"])
    grid_cfg = bcfg["grid"]
    model_names = list(grid_cfg["models"])
    opt = pcfg.get("optimizer", {})
    profiles = {k: v for k, v in load_profiles(pcfg).items() if k in grid_cfg["profiles"]}
    qgrid = np.arange(bcfg["pit_grid_step"], 1.0, bcfg["pit_grid_step"])

    dm = fit_margins_step(t, returns, cfg, jobs=1, cache_dir=cache_dir)
    fit_models_step(dm, cfg, model_names)
    subfunds = {sf: spec for sf, spec in subfunds_all.items() if set(spec["tickers"]) & set(dm.assets)}
    sfs_tickers = {sf: [a for a in spec["tickers"] if a in dm.assets] for sf, spec in subfunds.items()}
    variants = variant_specs(t, bcfg, pcfg)
    base_lag = int(bcfg["execution_lag"])
    lags = sorted({v["lag"] for v in variants})
    extra = tuple(L for L in lags if L != base_lag)
    pos = {a: i for i, a in enumerate(dm.assets)}

    # benchmarki bez kopuły (wagi niezależne od scenariuszy)
    bench: list[tuple[tuple, int, TwoLevelResult]] = []
    t0 = time.time()
    window = estimation_window(returns, t, rcfg["window_days"])[dm.assets]
    bench_variants = [v for v in variants if v["benchmarks"]]
    if "ew" in grid_cfg["benchmarks"]:
        ew = equal_weight(sfs_tickers)
        bench += [((BENCH_MODEL, "ew", "ALL", v["variant"]), v["lag"], ew) for v in bench_variants]
    if "mv_hist" in grid_cfg["benchmarks"]:
        for p in profiles.values():
            mv = mv_two_level(window, sfs_tickers, p, rcfg["horizon_days"], opt)
            bench += [((BENCH_MODEL, "mv_hist", p.id, v["variant"]), v["lag"], mv) for v in bench_variants]
    dm.timings["benchmarks"] = time.time() - t0

    weights, flags, fc_rows, tails, qrows, scen_rows = [], [], [], [], [], []

    def add_forecasts(items: list[tuple[tuple, int, TwoLevelResult]], sset: ScenarioSet, fmodel: str):
        for L in sorted({L for _, L, _ in items}):
            group = [(k, r) for k, LL, r in items if LL == L]
            R = sset.for_lag(L)
            X = np.zeros((len(dm.assets), len(group)))
            for j, (_, r) in enumerate(group):
                x = r.x
                X[[pos[a] for a in x.index], j] = x.to_numpy()
            losses = -(R @ X)
            for j, (key, r) in enumerate(group):
                st, tail, qq = loss_forecast(losses[:, j], alpha, qgrid)
                gamma = profiles[key[2]].gamma if key[2] in profiles else np.nan
                ce = certainty_equivalent(-losses[:, j], None, gamma) if np.isfinite(gamma) else np.nan
                fc_rows.append({"date": t, **dict(zip(KEY_COLS, key)), "lag": L, "forecast_model": fmodel,
                                **st, "exp_return": -st["mean_loss"], "ce_exante": ce})
                tails.append({"date": t, **dict(zip(KEY_COLS, key)), "tail": tail})
                qrows.append({"date": t, **dict(zip(KEY_COLS, key)), "quantiles": qq})

    def add_result(key, L, res: TwoLevelResult):
        weights.extend(_weights_rows(t, key, res))
        l2 = res.info.get("level2", {})
        flags.append({"date": t, **dict(zip(KEY_COLS, key)), "lag": L, "lt_relaxed": bool(res.flags.get("lt_relaxed")),
                      "lt_factor": float(res.flags.get("lt_factor", 1.0)),
                      "wmax_relaxed": bool(res.flags.get("wmax_relaxed")),
                      "infeasible_es": bool(res.flags.get("infeasible_es")),
                      "level2_success": bool(l2.get("success", l2.get("feasible", True))),
                      "level1_failures": int(sum(not v.get("success", v.get("feasible", True))
                                                 for v in res.info.get("level1", {}).values())),
                      "n_subfunds": int(len(res.W))})

    for key, L, r in bench:
        add_result(key, L, r)

    fm = bcfg.get("forecast_model", model_names[0])
    order = ([fm] if fm in model_names else []) + [m for m in model_names if m != fm]
    for name in order:
        sset = simulate_model(dm, name, cfg, bcfg, subfunds, latest, extra)
        scen_rows.append(scenario_stats_table(t, {name: sset}))
        t0 = time.time()
        sfs_by_lag = {L: subfund_data(sset.for_lag(L), dm.assets, subfunds) for L in lags}
        caches = {L: {} for L in lags}
        items = []
        for v in variants:
            for strat in v["strategies"]:
                for p in profiles.values():
                    res = optimize_two_level(sfs_by_lag[v["lag"]], p, strat, alpha, opt, v["measure"], v["q"],
                                             v["kappa"], caches[v["lag"]])
                    key = (name, strat, p.id, v["variant"])
                    add_result(key, v["lag"], res)
                    items.append((key, v["lag"], res))
        dm.timings[f"opt_{name}"] = time.time() - t0
        t0 = time.time()
        add_forecasts(items, sset, name)
        if name == fm and bench:
            add_forecasts(bench, sset, name)
        dm.timings[f"forecast_{name}"] = time.time() - t0
        del sset, sfs_by_lag

    # ocena modeli: okno estymacji i log-score poza próbą (sesje [t, t_next))
    t0 = time.time()
    dens = {n: m for n, m in dm.models.items() if m.has_density}
    dcfg = cfg["dependence"]
    gof = compare_with_reference(dm.models, dm.U, dcfg["reference"], dcfg["vuong_level"]).assign(date=t)
    oos = returns.loc[(returns.index >= t) & ((returns.index < t_next) if t_next is not None else True), dm.assets]
    ls_rows = []
    if len(oos):
        U_oos = pit_panel(dm.margins, oos)
        for n, m in dens.items():
            ls = oos_log_score(m, U_oos)
            ls_rows.append(pd.DataFrame({"date": t, "session": U_oos.index, "model": n, "logscore": ls}))
    dm.timings["gof"] = time.time() - t0

    out.mkdir(parents=True, exist_ok=True)
    write_parquet(pd.DataFrame(weights), out / "weights.parquet")
    write_parquet(pd.DataFrame(flags), out / "flags.parquet")
    write_parquet(pd.DataFrame(fc_rows), out / "forecasts.parquet")
    write_parquet(pd.DataFrame(qrows), out / "quantiles.parquet")
    write_parquet(pd.concat(scen_rows, ignore_index=True), out / "scenario_stats.parquet")
    write_parquet(gof, out / "gof.parquet")
    if ls_rows:
        write_parquet(pd.concat(ls_rows, ignore_index=True), out / "logscore.parquet")
    tail_dir = Path(run_dir) / "tails"
    write_parquet(pd.DataFrame(tails), tail_dir / f"{tag}.parquet")
    summary = {"date": tag, "seconds": time.time() - t_start, "n_assets": len(dm.assets),
               "subfunds": list(subfunds), "n_series": len(fc_rows), "timings": dm.timings,
               "cached": dm.cached, "latest": bool(latest), "t_next": None if t_next is None else str(t_next.date()),
               "n_scenarios": {r["model"].iloc[0]: int(r["n"].iloc[0]) for r in scen_rows}}
    (out / "done.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary
