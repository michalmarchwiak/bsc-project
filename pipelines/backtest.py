"""Backtest strategii opartych na modelach zależności (Etap 4).

Dla każdej daty rebalansowania t (pierwsza sesja miesiąca; ``config/backtest.yaml``):
    okno 1260 sesji ≤ t−1 → marginesy (pamięć podręczna) → 7 modeli zależności → scenariusze
    (adaptacyjne N) → optymalizacja strategii × profili (+ warianty wrażliwości w próbie kalibracyjnej)
    i benchmarków → prognozy ex ante (VaR, ES, ogon i kwantyle rozkładu strat) → AIC/BIC/Vuong i
    log-score OOS. Zapis po każdej dacie w artifacts/backtest/<run_id>/dates/<t>/ (wznawialne);
    równoległość po datach (``--jobs``, estymacja GARCH jobs = 1, wątki BLAS = 1).
Konsolidacja: zrealizowane stopy dzienne i okresowe (brutto/netto, dryf wag, koszty), szeregi SF,
PIT zrealizowanych strat, testy VaR/ES osobno dla kalibracji i hold-out.

Zamrożenie: przed pierwszą datą hold-out zapisywany jest ``frozen_params.json`` (skrót konfiguracji
parametrów meta); przebieg hold-out z inną konfiguracją jest odrzucany. Modele (GARCH, kopuły) są nadal
estymowane co miesiąc w oknie kroczącym – to część strategii.

Uruchomienie:
    python -m pipelines.backtest --sample calibration --jobs 4
    python -m pipelines.backtest --sample holdout --jobs 4
    python -m pipelines.backtest --consolidate-only
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pf import ARTIFACTS_DIR, ROOT, config
from pf.artifacts import update_manifest, write_parquet
from pf.backtest import engine
from pf.backtest.data import config_digest, eligible_dates, in_range, load_returns, run_id, sample_label
from pf.backtest.realized import add_sid, collect, collect_tails, evaluate_forecasts, realize
from pf.backtest.risk_tests import backtest_series
from pf.data import riskfree
from pf.portfolio import load_subfunds

log = logging.getLogger("backtest")
BACKTEST_DIR = ARTIFACTS_DIR / "backtest"


def load_all() -> tuple[dict, dict, dict, dict]:
    return config.load("models"), config.load("backtest"), config.load("profiles"), load_subfunds()


def current_run_id(cfg, bcfg, pcfg, subfunds) -> str:
    return run_id({"models": cfg, "backtest": bcfg, "profiles": pcfg, "subfunds": subfunds},
                  ROOT / cfg["returns"]["path"], bcfg.get("run_id_length", 12))


def all_dates(returns, cfg, bcfg, subfunds) -> pd.DatetimeIndex:
    d = eligible_dates(returns, cfg["rebalance"], subfunds, bcfg.get("require_all_subfunds", True),
                       bcfg.get("min_subfunds", 1))
    return in_range(d, [bcfg["calibration"][0], bcfg["holdout"][1]])


def freeze(run_dir: Path, cfg, bcfg, pcfg, subfunds) -> dict:
    """Zapis (lub weryfikacja) zamrożonych parametrów meta przed hold-out."""
    meta = {"profiles": pcfg, "backtest": {k: v for k, v in bcfg.items() if k != "riskfree"},
            "rvine": cfg["rvine"], "dependence": cfg["dependence"], "margins": cfg["margins"],
            "rebalance": cfg["rebalance"], "subfunds": subfunds}
    digest = config_digest(meta)
    path = run_dir / "frozen_params.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["digest"] != digest:
            raise RuntimeError("Konfiguracja różni się od zamrożonej przed hold-out – przebieg odrzucony")
        return old
    rec = {"digest": digest, "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "note": "Parametry meta (γ, κ, limity, q, ε, obcięcie vine'a) zamrożone przed hold-out; wartości "
                   "wyjściowe z config/profiles.yaml (bez dostrajania na kalibracji).", "params": meta}
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return rec


def _worker(args):
    t, t_next, run_dir, latest, force = args
    cfg, bcfg, pcfg, subfunds = load_all()
    returns = load_returns(ROOT / cfg["returns"]["path"])
    logging.getLogger("pf").setLevel(logging.WARNING)
    return engine.run_date(t, t_next, returns, cfg, bcfg, pcfg, subfunds, run_dir, engine.CACHE_DIR, latest, force)


def run_dates(dates: pd.DatetimeIndex, dates_all: pd.DatetimeIndex, run_dir: Path, jobs: int, force: bool) -> None:
    nxt = {t: (dates_all[i + 1] if i + 1 < len(dates_all) else None) for i, t in enumerate(dates_all)}
    todo = [t for t in dates if force or not (run_dir / "dates" / t.strftime("%Y-%m-%d") / "done.json").exists()]
    log.info("Daty do policzenia: %d z %d (%s – %s)", len(todo), len(dates),
             todo[0].date() if todo else "-", todo[-1].date() if todo else "-")
    if not todo:
        return
    tasks = [(t, nxt[t], run_dir, nxt[t] is None, force) for t in todo]
    t0, done = time.time(), 0
    if jobs <= 1:
        for task in tasks:
            s = _worker(task)
            done += 1
            log.info("%s: %.0f s, %d aktywów, %d szeregów (%d/%d)", s["date"], s["seconds"], s["n_assets"],
                     s["n_series"], done, len(tasks))
        return
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_worker, task): task[0] for task in tasks}
        for f in as_completed(futs):
            s = f.result()
            done += 1
            el = time.time() - t0
            log.info("%s: %.0f s, %d aktywów, N = %s (%d/%d, upłynęło %.1f h, ETA %.1f h)", s["date"], s["seconds"],
                     s["n_assets"], s["n_scenarios"], done, len(tasks), el / 3600,
                     el / done * (len(tasks) - done) / 3600)


def consolidate(run_dir: Path, returns: pd.DataFrame, cfg: dict, bcfg: dict, pcfg: dict, subfunds: dict,
                rf: pd.Series | None = None) -> dict:
    """Tabele zbiorcze: stopy zrealizowane, prognozy z PIT, testy VaR/ES, oceny modeli."""
    alpha = float(cfg["rebalance"]["es_alpha"])
    done = sorted(p.parent.name for p in (run_dir / "dates").glob("*/done.json"))
    dates = pd.DatetimeIndex(done)
    log.info("Konsolidacja %d dat (%s – %s)", len(dates), dates.min().date(), dates.max().date())
    weights = add_sid(collect(run_dir, "weights"))
    flags = add_sid(collect(run_dir, "flags"))
    lags = flags.drop_duplicates("sid").set_index("sid")["lag"]
    rf = riskfree.load(ROOT / bcfg["riskfree"]["path"]) if rf is None else rf
    cost = float(bcfg["transaction_cost"])

    wx = weights[weights["level"] == "x"][["date", "sid", "ticker", "weight"]]
    daily, periods = realize(returns, wx, lags, dates, cost, rf, bcfg["riskfree"]["day_count"])
    meta = flags.drop_duplicates("sid").set_index("sid")[["model", "strategy", "profile", "variant"]]
    periods = periods.join(meta, on="sid")
    periods["sample"] = [sample_label(t, bcfg) for t in periods["date"]]

    wsf = weights[(weights["level"] == "sf") & (weights["variant"] == "base")]
    wsf = add_sid(wsf, "subfund")[["date", "sid", "ticker", "weight"]]
    lags_sf = pd.Series({s: lags[s.rsplit("|", 1)[0]] for s in wsf["sid"].unique()})
    daily_sf, periods_sf = realize(returns, wsf, lags_sf, dates, cost, rf, bcfg["riskfree"]["day_count"])
    periods_sf["subfund"] = periods_sf["sid"].str.rsplit("|", n=1).str[1]
    periods_sf["series"] = periods_sf["sid"].str.rsplit("|", n=1).str[0]
    periods_sf = periods_sf.join(meta, on="series")
    periods_sf["sample"] = [sample_label(t, bcfg) for t in periods_sf["date"]]

    fc = collect(run_dir, "forecasts")
    grid = np.arange(bcfg["pit_grid_step"], 1.0, bcfg["pit_grid_step"])
    ev = evaluate_forecasts(fc, periods, collect_tails(run_dir), collect(run_dir, "quantiles"), grid)
    ev["sample"] = [sample_label(t, bcfg) for t in ev["date"]]

    rt_cfg = bcfg["risk_tests"]
    rows = []
    base = ev[(ev["variant"] == "base") & ev["loss"].notna()]
    for (sid, smp), g in base.groupby(["sid", "sample"]):
        g = g.sort_values("date")
        res = backtest_series(g["loss"], g["var"], g["es"], g["pit"], list(g["tail"]), alpha,
                              rt_cfg["z2_simulations"], tuple(rt_cfg["de_lags"]), seed=len(rows),
                              level=rt_cfg["level"])
        rows.append({"sid": sid, "sample": smp, **g.iloc[0][["model", "strategy", "profile", "variant"]].to_dict(),
                     **res})
    tests = pd.DataFrame(rows)

    out = {"periods": periods, "periods_sf": periods_sf, "tests": tests,
           "forecasts": ev.drop(columns=["tail", "quantiles"]), "flags": flags,
           "scenario_stats": collect(run_dir, "scenario_stats"), "gof": collect(run_dir, "gof"),
           "logscore": collect(run_dir, "logscore"),
           "weights_W": weights[weights["level"] == "W"].drop(columns=["ticker", "level"])}
    for name, df in out.items():
        write_parquet(df, run_dir / f"{name}.parquet")
    write_parquet(daily, run_dir / "daily.parquet")
    write_parquet(daily_sf, run_dir / "daily_sf.parquet")
    write_parquet(wx, run_dir / "weights_x.parquet")
    timings = pd.DataFrame([json.loads((run_dir / "dates" / d / "done.json").read_text()) for d in done])
    write_parquet(timings.drop(columns=["timings", "cached", "n_scenarios", "subfunds"]).assign(
        n_subfunds=timings["subfunds"].map(len)), run_dir / "timings.parquet")
    update_manifest(f"backtest/{run_dir.name}/periods", run_dir / "periods.parquet", n_rows=len(periods), params={
        "first_date": str(dates.min().date()), "last_date": str(dates.max().date()), "n_dates": len(dates),
        "n_series": int(periods["sid"].nunique()), "transaction_cost": cost, "execution_lag": bcfg["execution_lag"]})
    log.info("Zapisano tabele zbiorcze w %s (%d szeregów, %d okresów)", run_dir, periods["sid"].nunique(), len(periods))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", choices=["calibration", "holdout", "all"], default="calibration")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dates", nargs="*", type=pd.Timestamp, default=None)
    parser.add_argument("--max-dates", type=int, default=None, help="tylko pierwsze N dat (test)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--no-consolidate", action="store_true")
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, bcfg, pcfg, subfunds = load_all()
    rid = current_run_id(cfg, bcfg, pcfg, subfunds)
    run_dir = BACKTEST_DIR / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"models": cfg, "backtest": bcfg, "profiles": pcfg,
                                                     "subfunds": subfunds}, indent=1, ensure_ascii=False,
                                                    default=str), encoding="utf-8")
    log.info("run_id = %s", rid)
    returns = load_returns(ROOT / cfg["returns"]["path"])
    dates_all = all_dates(returns, cfg, bcfg, subfunds)
    if not args.consolidate_only:
        if args.dates:
            dates = pd.DatetimeIndex([dates_all[dates_all.searchsorted(d)] for d in args.dates])
        elif args.sample == "all":
            dates = dates_all
        else:
            dates = in_range(dates_all, bcfg[args.sample])
        if args.max_dates:
            dates = dates[:args.max_dates]
        if any(sample_label(t, bcfg) == "holdout" for t in dates):
            rec = freeze(run_dir, cfg, bcfg, pcfg, subfunds)
            log.info("Parametry zamrożone (%s, %s)", rec["digest"][:12], rec["frozen_utc"])
        run_dates(dates, dates_all, run_dir, args.jobs, args.force)
    if not args.no_consolidate:
        consolidate(run_dir, returns, cfg, bcfg, pcfg, subfunds)


if __name__ == "__main__":
    main()
