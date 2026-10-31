"""Pomiar kosztu kroku modelowego backtestu i prognoza czasu pełnego przebiegu (Etap 2).

Dla wybranych dat rebalansowania (domyślnie: pierwsza, środkowa i ostatnia data spełniająca
``min_obs`` i ``covers_all_subfunds`` w próbie kalibracja ∪ hold-out) mierzone są:
    * estymacja marginesów (``jobs = 1`` – jak przy równoległości po datach),
    * estymacja każdego modelu zależności z ``dependence.models``,
    * symulacja adaptacyjna (``config/backtest.yaml: scenarios``; N, ÊS, SE per portfel kontrolny),
    * log-score poza próbą (filtr GARCH naprzód, sesje do następnej daty),
oraz mikro-benchmark optymalizacji (LP Rockafellara–Uryaseva – strategie ``maxret_es``/``min_es``,
SLSQP z gradientem analitycznym dla CE – strategie ``ce``/``ce_dep``) dla typowych N.

Prognoza: T_daty = średni czas kroku modelowego + koszt optymalizacji (7 modeli × 5 profili × strategie
× (9 SF + poziom 2)); czas całkowity = n_dat × T_daty / jobs (skalowanie idealne – górna granica
przyspieszenia). Liczba dat: (a) dla obecnego pliku danych, (b) dla docelowej próby od ``calibration[0]``
do ``holdout[1]`` (pierwsza sesja każdego miesiąca).

Wyjście: artifacts/reports/scenario_timing.md, scenario_timing.csv (czasy), scenario_stats.csv (N, ÊS, SE).

Uruchomienie:
    python -m pipelines.scenario_timing [--dates 2014-02-03 2017-07-03 2020-12-01] [--jobs 3]
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from pf import ARTIFACTS_DIR, ROOT, config
from pf.backtest import engine
from pf.backtest.data import eligible_dates, in_range, load_returns
from pf.models.gof import oos_log_score, pit_panel
from pf.portfolio import load_subfunds

log = logging.getLogger("scenario_timing")
REPORT_DIR = ARTIFACTS_DIR / "reports"


def measure_date(t: pd.Timestamp, next_t: pd.Timestamp | None, models: list[str]) -> tuple[dict, pd.DataFrame]:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    cfg, bcfg = config.load("models"), config.load("backtest")
    returns = load_returns(ROOT / cfg["returns"]["path"])
    subfunds = load_subfunds()
    t0 = time.time()
    dm = engine.fit_margins_step(t, returns, cfg, jobs=1, cache_dir=None)
    engine.fit_models_step(dm, cfg, models)
    scen = engine.scenarios_step(dm, cfg, bcfg, subfunds, models)
    stats = engine.scenario_stats_table(t, scen)
    t1 = time.time()
    oos = returns.loc[(returns.index >= t) & ((returns.index < next_t) if next_t is not None else True), dm.assets]
    U_oos = pit_panel(dm.margins, oos)
    ls = {m: float(oos_log_score(dm.models[m], U_oos).sum()) for m in models if dm.models[m].has_density}
    dm.timings["log_score"] = time.time() - t1
    row = {"date": t, "n_assets": len(dm.assets), "panel_obs": len(dm.panel), "total": time.time() - t0,
           **dm.timings, **{f"n_{m}": scen[m].n for m in models}, **{f"ls_{m}": v for m, v in ls.items()}}
    return row, stats


def _measure(args):
    return measure_date(*args)


def bench_optimization(n_assets: int, n_scen: list[int], alpha: float, seed: int = 0) -> pd.DataFrame:
    """Czas LP (max E[R] przy ES ≤ L oraz min ES) i SLSQP (max CE − κ wᵀMw) na losowych scenariuszach."""
    from scipy import optimize

    from pf.opt.cvar import _solve, empirical_es

    rng = np.random.default_rng(seed)
    rows = []
    for N in n_scen:
        R = rng.standard_t(5, size=(N, n_assets)) * rng.uniform(0.02, 0.08, n_assets) + rng.uniform(0, 0.01, n_assets)
        w_ew = np.full(n_assets, 1.0 / n_assets)
        L = empirical_es(-R @ w_ew, alpha)
        cap = max(0.3, 1.0 / n_assets)
        t0 = time.time()
        _solve(R, alpha, L * (1 + 1e-10), cap, minimize_es=False)
        t_max = time.time() - t0
        t0 = time.time()
        _solve(R, alpha, None, cap, minimize_es=True)
        t_min = time.time() - t0
        gamma, kappa = 5.0, 0.35
        M = np.corrcoef(R, rowvar=False)

        def f(w):
            g = 1.0 + R @ w
            p = g ** (1.0 - gamma)
            m = p.mean()
            ce = m ** (1.0 / (1.0 - gamma)) - 1.0
            dce = m ** (1.0 / (1.0 - gamma) - 1.0) * (R * (g ** (-gamma))[:, None]).mean(axis=0)
            return -(ce - kappa * w @ M @ w), -(dce - 2 * kappa * M @ w)

        t0 = time.time()
        optimize.minimize(f, w_ew, jac=True, method="SLSQP", bounds=[(0, cap)] * n_assets,
                          constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}],
                          options={"maxiter": 200, "ftol": 1e-12})
        t_slsqp = time.time() - t0
        rows.append({"n_assets": n_assets, "N": N, "lp_maxret_es": t_max, "lp_min_es": t_min, "slsqp_ce": t_slsqp})
    return pd.DataFrame(rows)


def target_dates(bcfg: dict) -> int:
    idx = pd.bdate_range(bcfg["calibration"][0], bcfg["holdout"][1])
    return int(pd.Series(idx, index=idx).groupby(idx.to_period("M")).min().shape[0])


def report(times: pd.DataFrame, stats: pd.DataFrame, bench: pd.DataFrame, models: list[str], n_dates_data: int,
           n_dates_target: int, jobs_forecast: list[int], bcfg: dict, opt_counts: dict) -> str:
    scfg = bcfg["scenarios"]
    lines = ["# Koszt kroku modelowego backtestu i prognoza czasu (Etap 2)", "",
             f"Scenariusze: partie {scfg['batch']}, N ∈ [{scfg['n_min']}, {scfg['n_max']}], stop przy "
             f"max_k SE/|ÊS| ≤ {scfg['rel_se']} (portfele kontrolne: EW w każdym SF i EW między SF), "
             f"opóźnienie wykonania {bcfg['execution_lag']} sesja (H + L dni symulacji).",
             "Pomiar na jednym rdzeniu na datę (jobs = 1 w estymacji GARCH, wątki BLAS = 1).", "",
             "## Czasy kroku modelowego [s]", "",
             "| data | aktywa | panel | marginesy | " + " | ".join(f"fit {m}" for m in models) + " | "
             + " | ".join(f"sym. {m}" for m in models) + " | log-score | razem |",
             "|---|" + "---:|" * (5 + 2 * len(models))]
    for r in times.itertuples():
        d = r._asdict()
        lines.append(f"| {r.date.date()} | {r.n_assets} | {r.panel_obs} | {r.margins:.1f} | "
                     + " | ".join(f"{d[f'fit_{m}']:.1f}" for m in models) + " | "
                     + " | ".join(f"{d[f'sim_{m}']:.1f}" for m in models) + f" | {r.log_score:.1f} | {r.total:.1f} |")
    lines += ["", "## Liczba scenariuszy N i względny błąd ÊS (max po portfelach kontrolnych)", "",
              "| data | " + " | ".join(models) + " |", "|---|" + "---:|" * len(models)]
    agg = stats.groupby(["date", "model"]).agg(n=("n", "first"), rel=("rel_se", "max"), conv=("converged", "first"))
    for t in sorted(stats["date"].unique()):
        cells = []
        for m in models:
            a = agg.loc[(t, m)]
            cells.append(f"{int(a.n)} ({100 * a.rel:.1f}%{'' if a.conv else ', n_max'})")
        lines.append(f"| {pd.Timestamp(t).date()} | " + " | ".join(cells) + " |")
    worst = stats.loc[stats.groupby(["date", "model"])["rel_se"].idxmax(), ["date", "model", "portfolio"]]
    lines += ["", "Portfel wyznaczający N (max SE/|ÊS|): " + ", ".join(
        f"{pd.Timestamp(r.date).date()}/{r.model}: {r.portfolio}" for r in worst.itertuples()) + ".", ""]

    lines += ["## Mikro-benchmark optymalizacji [s na problem]", "",
              "| aktywa | N | LP max E[R] przy ES ≤ L | LP min ES | SLSQP CE − κD |", "|---:|---:|---:|---:|---:|"]
    lines += [f"| {r.n_assets} | {r.N} | {r.lp_maxret_es:.2f} | {r.lp_min_es:.2f} | {r.slsqp_ce:.3f} |"
              for r in bench.itertuples()]
    n_mean = float(np.mean([times[f"n_{m}"].mean() for m in models]))

    def at_n(col: str) -> float:  # interpolacja log-log czasu problemu przy średnim N
        return float(np.exp(np.interp(np.log(n_mean), np.log(bench["N"]), np.log(bench[col]))))

    per_problem_lp = at_n("lp_maxret_es") + at_n("lp_min_es")
    per_problem_ce = 2 * at_n("slsqp_ce")
    n_problems = opt_counts["models"] * opt_counts["profiles"] * opt_counts["levels"]
    opt_per_date = n_problems * (per_problem_lp + per_problem_ce)
    model_step = float(times["total"].mean())
    per_date = model_step + opt_per_date
    lines += ["", f"Średnie N ≈ {n_mean:.0f}; koszt optymalizacji na datę ≈ {opt_counts['models']} modeli × "
              f"{opt_counts['profiles']} profili × {opt_counts['levels']} problemów (9 SF + poziom 2) × "
              f"(2 LP + 2 SLSQP) ≈ {opt_per_date:.0f} s (czasy interpolowane log-log przy średnim N; "
              f"LP {per_problem_lp:.2f} s, SLSQP {per_problem_ce:.3f} s na parę problemów; SLSQP to prototyp "
              "funkcji celu z Etapu 3). Czas pełnego LP rośnie z N w przybliżeniu kwadratowo.", "",
              "## Prognoza czasu pełnego przebiegu", "",
              f"Krok modelowy: średnio {model_step:.0f} s/datę; z optymalizacją ≈ {per_date:.0f} s/datę.", "",
              "| próba | liczba dat | " + " | ".join(f"jobs = {j} [h]" for j in jobs_forecast) + " |",
              "|---|---:|" + "---:|" * len(jobs_forecast)]
    for label, n in (("obecny plik danych", n_dates_data), ("docelowa (kalibracja + hold-out)", n_dates_target)):
        lines.append(f"| {label} | {n} | " + " | ".join(f"{n * per_date / j / 3600:.1f}" for j in jobs_forecast) + " |")
    lines += ["", "Uwagi: czas symulacji rośnie liniowo z N i (dla vine'ów) z liczbą krawędzi z kopułą t; "
              "daty wcześniejsze mają mniej aktywów. Prognoza dla próby docelowej zakłada koszt daty jak w pomiarze "
              "(≈ 80 aktywów). Analizy wrażliwości (Etap 4) i MC ex post (Etap 5) nie są wliczone."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dates", nargs="*", type=pd.Timestamp, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--jobs", type=int, default=1, help="równoległe daty pomiaru")
    parser.add_argument("--forecast-jobs", nargs="*", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--report-only", action="store_true",
                        help="odtwórz raport z zapisanych scenario_timing.csv i scenario_stats.csv (bez pomiaru)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, bcfg = config.load("models"), config.load("backtest")
    models = args.models or cfg["dependence"]["models"]
    returns = load_returns(ROOT / cfg["returns"]["path"])
    all_dates = eligible_dates(returns, cfg["rebalance"], load_subfunds())
    sample = in_range(all_dates, [bcfg["calibration"][0], bcfg["holdout"][1]])
    dates = pd.DatetimeIndex(args.dates) if args.dates else sample[[0, len(sample) // 2, len(sample) - 1]]
    nxt = {t: (all_dates[all_dates > t].min() if (all_dates > t).any() else None) for t in dates}
    if args.report_only:
        times = pd.read_csv(REPORT_DIR / "scenario_timing.csv", parse_dates=["date"])
        stats = pd.read_csv(REPORT_DIR / "scenario_stats.csv", parse_dates=["date"])
    else:
        log.info("Daty pomiaru: %s; modele: %s", [str(t.date()) for t in dates], models)
        tasks = [(t, nxt[t], models) for t in dates]
        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                out = list(ex.map(_measure, tasks))
        else:
            out = [_measure(x) for x in tasks]
        times = pd.DataFrame([o[0] for o in out])
        stats = pd.concat([o[1] for o in out], ignore_index=True)
    bench = bench_optimization(10, [5000, 10000, 20000, 50000], cfg["rebalance"]["es_alpha"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    times.to_csv(REPORT_DIR / "scenario_timing.csv", index=False)
    stats.to_csv(REPORT_DIR / "scenario_stats.csv", index=False)
    bench.to_csv(REPORT_DIR / "optimization_benchmark.csv", index=False)
    opt_counts = {"models": len(models), "profiles": 5, "levels": len(load_subfunds()) + 1}
    text = report(times, stats, bench, models, len(sample), target_dates(bcfg), args.forecast_jobs, bcfg, opt_counts)
    (REPORT_DIR / "scenario_timing.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
