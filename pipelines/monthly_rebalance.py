"""Comiesięczne rebalansowanie wag sub-funduszy (pierwsza sesja miesiąca).

Dla każdej daty rebalansowania t (pierwsza sesja miesiąca w danych):
    1. okno estymacji: ostatnie ``window_days`` sesji przed t (dane ≤ t−1, bez look-ahead);
    2. marginesy AR(1)–GJR-GARCH(1,1) skew-t i R-vine (3 drzewa) na resztach;
    3. scenariusze stóp zwrotu aktywów w horyzoncie H (start ze stanu GARCH na t−1),
       agregacja do sub-funduszy macierzą A (wagi równe w sub-funduszu);
    4. LP: max E[R] przy ES_97,5% ≤ L (portfel min-ES, gdy limit nieosiągalny);
    5. (dla dat historycznych) zrealizowane stopy zwrotu sub-funduszy i portfela do
       następnego rebalansowania (kupno i trzymanie w miesiącu).

Przebieg jest przyrostowy: daty już policzone są pomijane (``--force`` liczy ponownie).
Daty historyczne używają ``backtest_scenarios``, najnowsza – ``n_scenarios`` scenariuszy.

Wyjście (artifacts/rebalance/):
    weights.parquet            – wagi, E[R], ES, VaR, wykonalność, rodziny kopuł, AIC
    realized.parquet           – zrealizowane stopy zwrotu sub-funduszy i portfela
    vines/<data>.json.gz       – R-vine dla daty; margins/<data>.parquet – parametry GARCH
    scenarios_latest.parquet   – scenariusze sub-funduszy dla ostatniej daty

Uruchomienie:
    python -m pipelines.monthly_rebalance                 # wszystkie brakujące daty
    python -m pipelines.monthly_rebalance --date 2020-12-01
    python -m pipelines.monthly_rebalance --start 2019-01-01 --scenarios 1000 --jobs 4
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from pf import ARTIFACTS_DIR, ROOT, config
from pf.artifacts import update_manifest, write_parquet
from pf.opt.cvar import max_return_es
from pf.portfolio import load_subfunds, lookthrough_matrix, subfund_returns
from pf.sim.scenarios import eligible_assets, estimate, simulate

log = logging.getLogger("monthly_rebalance")

OUT_DIR = ARTIFACTS_DIR / "rebalance"
REPORT_DIR = ARTIFACTS_DIR / "reports"


def rebalance_dates(index: pd.DatetimeIndex, min_history: int) -> pd.DatetimeIndex:
    """Pierwsza sesja każdego miesiąca, przed którą jest co najmniej ``min_history`` sesji."""
    s = pd.Series(index, index=index)
    firsts = pd.DatetimeIndex(s.groupby(index.to_period("M")).min().to_numpy())
    pos = index.get_indexer(firsts)
    return firsts[pos >= min_history]


def covers_all_subfunds(returns: pd.DataFrame, t: pd.Timestamp, rcfg: dict, subfunds: dict) -> bool:
    """Czy w oknie przed t każdy sub-fundusz ma co najmniej jedno aktywo z ``min_obs`` obserwacjami."""
    assets = set(eligible_assets(estimation_window(returns, t, rcfg["window_days"]), rcfg["min_obs"]))
    return all(assets & set(spec["tickers"]) for spec in subfunds.values())


def estimation_window(returns: pd.DataFrame, t: pd.Timestamp, window_days: int) -> pd.DataFrame:
    return returns.loc[returns.index < t].iloc[-window_days:]


def realized_returns(returns: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp | None) -> pd.Series:
    """Proste stopy zwrotu aktywów w okresie [t0, t1) (kupno i trzymanie; brak notowań = 0)."""
    period = returns.loc[(returns.index >= t0) & ((returns.index < t1) if t1 is not None else True)]
    return np.expm1(period.fillna(0.0).sum())


def summary_report(weights: pd.DataFrame, realized: pd.DataFrame, subfunds: dict, rcfg: dict) -> str:
    """Raport: statystyki zrealizowane portfela i sub-funduszy, średnie wagi, przekroczenia VaR."""
    m = realized.merge(weights[["date", "var", "es", "feasible"]], on="date")
    loss = -m["portfolio"]
    n = len(m)

    def stats(r: pd.Series) -> tuple[float, float, float, float]:
        wealth = (1.0 + r).cumprod()
        ann = wealth.iloc[-1] ** (12.0 / len(r)) - 1.0
        vol = r.std() * np.sqrt(12.0)
        mdd = (wealth / wealth.cummax() - 1.0).min()
        return 100 * ann, 100 * vol, ann / vol if vol > 0 else np.nan, 100 * mdd

    lines = [
        "# Comiesięczne rebalansowanie – podsumowanie",
        "",
        f"Okres: {m['date'].min().date()} – {m['end'].max().date()} ({n} miesięcy). Okno estymacji "
        f"{rcfg['window_days']} sesji, horyzont {rcfg['horizon_days']} sesji, ES_{rcfg['es_alpha']} ≤ "
        f"{100 * rcfg['es_limit']:.1f}%.",
        f"Limit ES nieosiągalny (portfel min-ES): {int((~m['feasible']).sum())} mies.",
        f"Przekroczenia VaR_{rcfg['es_alpha']}: {int((loss > m['var']).sum())} (oczekiwane "
        f"{n * (1 - rcfg['es_alpha']):.1f}); strata > prognozowany ES: {int((loss > m['es']).sum())}.",
        "",
        "| strategia | zwrot roczny [%] | zmienność [%] | Sharpe (r_f = 0) | max DD [%] | średnia waga |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    ann, vol, sh, mdd = stats(m["portfolio"])
    lines.append(f"| portfel (ES-LP) | {ann:.2f} | {vol:.2f} | {sh:.2f} | {mdd:.2f} | – |")
    for sf, spec in subfunds.items():
        if f"r_{sf}" in m:
            ann, vol, sh, mdd = stats(m[f"r_{sf}"])
            lines.append(f"| {sf} {spec['name']} | {ann:.2f} | {vol:.2f} | {sh:.2f} | {mdd:.2f} | "
                         f"{weights[f'w_{sf}'].mean():.3f} |")
    return "\n".join(lines) + "\n"


def rebalance_one(t: pd.Timestamp, returns: pd.DataFrame, cfg: dict, subfunds: dict, n_scen: int,
                  jobs: int | None) -> tuple[dict, object, pd.DataFrame, pd.DataFrame]:
    rcfg = cfg["rebalance"]
    window = estimation_window(returns, t, rcfg["window_days"])
    model = estimate(window, cfg, rcfg["min_obs"], jobs=jobs)
    rng = np.random.default_rng([rcfg["seed"], int(t.strftime("%Y%m%d"))])
    scen_assets = simulate(model, n_scen, rcfg["horizon_days"], rng)
    A = lookthrough_matrix(subfunds, model.assets)
    R = subfund_returns(scen_assets, A)
    sol = max_return_es(R, rcfg["es_alpha"], rcfg["es_limit"], rcfg["max_weight"])
    families = pd.Series([e.copula.label() for e in model.vine.edges()]).value_counts()
    row = {"date": t, "window_start": window.index.min(), "window_end": window.index.max(),
           "n_assets": len(model.assets), "panel_obs": model.panel_obs, "n_scenarios": n_scen,
           "exp_return": sol.expected_return, "es": sol.es, "var": sol.var, "feasible": sol.feasible,
           "vine_aic": model.vine.aic(), **{f"w_{sf}": 0.0 for sf in subfunds},
           **{f"w_{sf}": w for sf, w in zip(A.index, sol.weights)},
           **{f"n_{fam}": int(c) for fam, c in families.items()}}
    margins = pd.DataFrame([f.to_dict() for f in model.margins.values()])
    return row, model, margins, pd.DataFrame(R, columns=A.index)


def main() -> None:
    cfg = config.load("models")
    rcfg = cfg["rebalance"]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", type=pd.Timestamp, default=None, help="pojedyncza data rebalansowania")
    parser.add_argument("--start", type=pd.Timestamp, default=None)
    parser.add_argument("--end", type=pd.Timestamp, default=None)
    parser.add_argument("--scenarios", type=int, default=None, help="liczba scenariuszy dla dat historycznych")
    parser.add_argument("--jobs", type=int, default=None, help="procesy do estymacji GARCH")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    returns = pd.read_csv(ROOT / cfg["returns"]["path"], index_col=0, parse_dates=True).sort_index()
    subfunds = load_subfunds()
    all_dates = rebalance_dates(returns.index, rcfg["min_obs"])
    all_dates = pd.DatetimeIndex([t for t in all_dates if covers_all_subfunds(returns, t, rcfg, subfunds)])
    dates = all_dates
    if args.date is not None:
        dates = all_dates[all_dates == all_dates[all_dates.searchsorted(args.date)]]
    if args.start is not None:
        dates = dates[dates >= args.start]
    if args.end is not None:
        dates = dates[dates <= args.end]

    weights_path = OUT_DIR / "weights.parquet"
    done = pd.read_parquet(weights_path) if weights_path.exists() else pd.DataFrame(columns=["date"])
    if not args.force:
        dates = dates[~dates.isin(pd.DatetimeIndex(done["date"]))]
    log.info("Daty do policzenia: %d (%s – %s)", len(dates), dates.min().date() if len(dates) else "-",
             dates.max().date() if len(dates) else "-")

    rows = []
    latest_date = all_dates.max()
    for t in dates:
        t0 = time.time()
        n_scen = rcfg["n_scenarios"] if t == latest_date else (args.scenarios or rcfg["backtest_scenarios"])
        row, model, margins, R = rebalance_one(t, returns, cfg, subfunds, n_scen, args.jobs)
        tag = t.strftime("%Y-%m-%d")
        model.vine.save(OUT_DIR / "vines" / f"{tag}.json.gz")
        write_parquet(margins, OUT_DIR / "margins" / f"{tag}.parquet")
        if t == latest_date:
            write_parquet(R.astype("float32"), OUT_DIR / "scenarios_latest.parquet")
        rows.append(row)
        w = {k[2:]: round(float(v), 3) for k, v in row.items() if k.startswith("w_") and v > 1e-4}
        log.info("%s: E[R] = %.2f%%, ES = %.2f%%%s, wagi %s (%.0f s)", tag, 100 * row["exp_return"],
                 100 * row["es"], "" if row["feasible"] else " (limit nieosiągalny – min ES)", w, time.time() - t0)
        # zapis po każdej dacie – przerwany przebieg można wznowić
        out = pd.concat([done[~done["date"].isin(pd.DatetimeIndex([r["date"] for r in rows]))],
                         pd.DataFrame(rows)], ignore_index=True).sort_values("date")
        write_parquet(out.reset_index(drop=True), weights_path)

    if not weights_path.exists():
        return
    weights = pd.read_parquet(weights_path)
    realized = []
    reb = pd.DatetimeIndex(weights["date"])
    for i, t in enumerate(reb):
        t_next = reb[i + 1] if i + 1 < len(reb) else None
        if t_next is None:
            continue  # okres jeszcze się nie zakończył
        assets = eligible_assets(estimation_window(returns, t, rcfg["window_days"]), rcfg["min_obs"])
        A = lookthrough_matrix(subfunds, assets)
        r_sf = A.to_numpy() @ realized_returns(returns, t, t_next).reindex(A.columns).to_numpy()
        w = weights.loc[i, [f"w_{sf}" for sf in A.index]].to_numpy(dtype=float)
        realized.append({"date": t, "end": t_next, "portfolio": float(w @ r_sf),
                         **{f"r_{sf}": float(r) for sf, r in zip(A.index, r_sf)}})
    if realized:
        realized = pd.DataFrame(realized)
        write_parquet(realized, OUT_DIR / "realized.parquet")
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "rebalance_summary.md").write_text(summary_report(weights, realized, subfunds, rcfg),
                                                         encoding="utf-8")
    update_manifest("rebalance/weights", weights_path, n_rows=len(weights), params={
        "rebalance": rcfg, "rvine": cfg["rvine"], "margins": cfg["margins"], "returns": cfg["returns"]["path"],
        "first_date": str(reb.min().date()), "last_date": str(reb.max().date())})


if __name__ == "__main__":
    main()
