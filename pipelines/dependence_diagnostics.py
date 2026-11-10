"""Diagnostyka zależności w horyzoncie miesiąca: scenariusze modelu a dane okna estymacji.

Kryterium doboru agregacji panelu reszt (``dependence.aggregation`` w ``config/models.yaml``): zależność
H-dniowych stóp sub-funduszy w scenariuszach ma odtwarzać zależność obserwowaną w danych. Dla każdej daty t
(okno ≤ t−1, skład sub-funduszy z daty t) i każdego wariantu k (sumy reszt z k sesji):

    dane:        stopy H-dniowe (nakładające się) portfeli równych wag sub-funduszy z okna estymacji,
    scenariusze: stopy H-dniowe tych samych portfeli z N scenariuszy modelu (marginesy GARCH + kopuła na panelu k),
    miary:       ρ_S (korelacja rang Spearmana) i Λ(q) = P(V_i ≤ q, V_j ≤ q)/q (dolny ogon) dla par sub-funduszy.

Raport: średni błąd bezwzględny i średnia różnica (scenariusze − dane) po parach i datach, osobno dla par
między rynkami (GPW × USA) i w obrębie rynku. Marginesy są wspólne dla wszystkich k (pamięć podręczna backtestu),
a scenariusze korzystają ze wspólnych liczb losowych, więc różnice między wariantami wynikają tylko z panelu.

Ograniczenia: okno 1260 sesji zawiera ok. 60 rozłącznych okresów 21-dniowych, więc pojedyncza korelacja z danych
ma błąd standardowy rzędu 0,1 – wnioski należy opierać na średnich po parach i datach. Dane dają zależność
bezwarunkową w oknie, scenariusze – warunkową względem stanu GARCH w dacie t.

Uruchomienie:
    python -m pipelines.dependence_diagnostics                       # co 12. data backtestu, k ∈ {1, 2, 5, 10}
    python -m pipelines.dependence_diagnostics --dates 2012-01-02 2020-04-01 --days 1 5 --models gauss rvine
Wynik: artifacts/reports/dependence_diagnostics.csv i .md.
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd

from pf import ARTIFACTS_DIR, config, universe
from pf.backtest import engine
from pf.backtest.data import estimation_window
from pf.models.margins import default_market
from pf.opt.dependence_penalty import dependence_matrix
from pf.portfolio import active_universe, deterministic_tickers, load_subfunds
from pipelines.backtest import all_dates, prepare_returns

log = logging.getLogger("dependence_diagnostics")
REPORT_DIR = ARTIFACTS_DIR / "reports"


def subfund_members(assets: list[str], subfunds: dict, exclude=()) -> dict[str, list[int]]:
    """SF → pozycje aktywów stochastycznych w ``assets`` (bez SF deterministycznych i pustych)."""
    pos = {a: i for i, a in enumerate(assets)}
    out = {}
    for sf, spec in subfunds.items():
        idx = [pos[a] for a in spec["tickers"] if a in pos and a not in set(exclude)]
        if idx and not spec.get("deterministic", False):
            out[sf] = idx
    return out


def empirical_horizon_returns(window: pd.DataFrame, members: dict[str, list[int]], horizon: int) -> np.ndarray:
    """Nakładające się H-dniowe log-stopy portfeli równych wag SF w oknie (brak notowania → stopa 0)."""
    simple = np.expm1(window.fillna(0.0).to_numpy())
    sf = np.column_stack([simple[:, idx].mean(axis=1) for idx in members.values()])
    return pd.DataFrame(np.log1p(sf)).rolling(horizon).sum().dropna().to_numpy()


def scenario_horizon_returns(R: np.ndarray, members: dict[str, list[int]]) -> np.ndarray:
    return np.column_stack([R[:, idx].mean(axis=1) for idx in members.values()])


def pair_table(X_emp: np.ndarray, X_scen: np.ndarray, names: list[str], q: float) -> pd.DataFrame:
    rho_e, rho_s = dependence_matrix(X_emp, "rho_S"), dependence_matrix(X_scen, "rho_S")
    lam_e, lam_s = dependence_matrix(X_emp, "lambda_L", q), dependence_matrix(X_scen, "lambda_L", q)
    rows = [{"sf_i": names[i], "sf_j": names[j], "rho_emp": rho_e[i, j], "rho_scen": rho_s[i, j],
             "lam_emp": lam_e[i, j], "lam_scen": lam_s[i, j]} for i, j in combinations(range(len(names)), 2)]
    return pd.DataFrame(rows)


def diagnose_date(args) -> pd.DataFrame:
    t, days, models, n_scen, q = args
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    cfg, bcfg = config.load("models"), config.load("backtest")
    pool = load_subfunds()
    returns = prepare_returns(cfg, {**bcfg, "options_overlay": {"enabled": False}}, pool)
    returns, subfunds = universe.build(pool, returns, cfg["rebalance"]).view(t)
    det = deterministic_tickers(subfunds)
    horizon = int(cfg["rebalance"]["horizon_days"])
    fixed = {**bcfg, "scenarios": {**bcfg["scenarios"], "batch": n_scen, "n_min": n_scen, "n_max": n_scen}}
    out = []
    for k in days:
        cfg_k = {**cfg, "dependence": {**cfg["dependence"], "aggregation": {"days": int(k), "overlap": True}}}
        dm = engine.fit_margins_step(t, returns, cfg_k, jobs=1, deterministic=det)
        engine.fit_models_step(dm, cfg_k, models)
        members = subfund_members(dm.assets, subfunds, exclude=det)
        names = list(members)
        market = {sf: {default_market(dm.assets[i]) for i in idx} for sf, idx in members.items()}
        window = estimation_window(returns, t, cfg["rebalance"]["window_days"])[dm.assets]
        X_emp = empirical_horizon_returns(window, members, horizon)
        for name in models:
            sset = engine.simulate_model(dm, name, cfg_k, fixed, subfunds)
            tab = pair_table(X_emp, scenario_horizon_returns(sset.R, members), names, q)
            tab["cross_market"] = [market[a] != market[b] for a, b in zip(tab["sf_i"], tab["sf_j"])]
            out.append(tab.assign(date=t, model=name, days=int(k), panel_obs=len(dm.panel)))
    return pd.concat(out, ignore_index=True)


def summarize(pairs: pd.DataFrame) -> pd.DataFrame:
    d = pairs.assign(rho_err=pairs["rho_scen"] - pairs["rho_emp"], lam_err=pairs["lam_scen"] - pairs["lam_emp"],
                     group=np.where(pairs["cross_market"], "GPW × USA", "ten sam rynek"))
    both = pd.concat([d, d.assign(group="wszystkie pary")], ignore_index=True)
    g = both.groupby(["model", "days", "group"])
    return g.agg(n_pairs=("rho_err", "size"), rho_emp=("rho_emp", "mean"), rho_scen=("rho_scen", "mean"),
                 rho_bias=("rho_err", "mean"), rho_mae=("rho_err", lambda x: x.abs().mean()),
                 lam_emp=("lam_emp", "mean"), lam_scen=("lam_scen", "mean"), lam_bias=("lam_err", "mean"),
                 lam_mae=("lam_err", lambda x: x.abs().mean())).reset_index()


def markdown(summary: pd.DataFrame, pairs: pd.DataFrame, horizon: int, q: float, n_scen: int) -> str:
    dates = pd.DatetimeIndex(pairs["date"].unique()).sort_values()
    lines = ["# Diagnostyka zależności w horyzoncie miesiąca", "",
             f"Daty: {len(dates)} ({dates.min().date()} – {dates.max().date()}), uniwersum: "
             f"{pairs.attrs.get('universe', '')}, H = {horizon} sesji, N = {n_scen} scenariuszy, q = {q:g}. "
             "Porównanie stóp H-dniowych portfeli równych wag sub-funduszy: dane okna estymacji a scenariusze modelu "
             "z kopułą szacowaną na sumach reszt z k sesji (k = 1: panel dzienny). ρ_S – korelacja rang, "
             "Λ(q) – zależność w dolnym ogonie; błąd = scenariusze − dane, średnie po parach SF i datach.", "",
             "| model | k | pary | ρ_S dane | ρ_S scen. | średni błąd ρ_S | MAE ρ_S | Λ dane | Λ scen. | średni błąd Λ | MAE Λ |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary.sort_values(["model", "group", "days"]).itertuples():
        lines.append(f"| {r.model} | {r.days} | {r.group} | {r.rho_emp:.3f} | {r.rho_scen:.3f} | {r.rho_bias:+.3f} | "
                     f"{r.rho_mae:.3f} | {r.lam_emp:.3f} | {r.lam_scen:.3f} | {r.lam_bias:+.3f} | {r.lam_mae:.3f} |")
    lines += ["", "Pojedyncza korelacja z danych ma błąd standardowy rzędu 0,1 (ok. 60 rozłącznych okresów w oknie), "
              "więc MAE nie spada do zera nawet dla poprawnego modelu; miarodajne jest porównanie wariantów k."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dates", nargs="*", type=pd.Timestamp, default=None)
    parser.add_argument("--every", type=int, default=12, help="co która data backtestu (gdy brak --dates)")
    parser.add_argument("--days", nargs="*", type=int, default=[1, 2, 5, 10], help="warianty agregacji k")
    parser.add_argument("--models", nargs="*", default=["gauss"], help="modele zależności (np. gauss rvine)")
    parser.add_argument("--n", type=int, default=5000, help="liczba scenariuszy")
    parser.add_argument("--q", type=float, default=0.10, help="poziom dolnego ogona Λ(q)")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, bcfg = config.load("models"), config.load("backtest")
    pool = load_subfunds()
    returns = prepare_returns(cfg, {**bcfg, "options_overlay": {"enabled": False}}, pool)
    calendar = all_dates(returns, cfg, bcfg, pool)
    if args.dates:
        dates = pd.DatetimeIndex([calendar[min(calendar.searchsorted(d), len(calendar) - 1)] for d in args.dates])
    else:
        dates = calendar[::max(args.every, 1)]
    tasks = [(t, tuple(args.days), tuple(args.models), args.n, args.q) for t in dates]
    log.info("Daty: %d, k: %s, modele: %s", len(tasks), args.days, args.models)
    if args.jobs <= 1:
        parts = [diagnose_date(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            parts = list(ex.map(diagnose_date, tasks))
    pairs = pd.concat(parts, ignore_index=True)
    pairs.attrs["universe"] = active_universe()
    summary = summarize(pairs)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(REPORT_DIR / "dependence_diagnostics.csv", index=False, float_format="%.5f")
    (REPORT_DIR / "dependence_diagnostics.md").write_text(
        markdown(summary, pairs, int(cfg["rebalance"]["horizon_days"]), args.q, args.n), encoding="utf-8")
    log.info("Zapisano %s\n%s", REPORT_DIR / "dependence_diagnostics.md",
             summary[summary["group"] == "wszystkie pary"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
