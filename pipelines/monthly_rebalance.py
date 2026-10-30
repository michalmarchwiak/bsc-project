"""Comiesięczne rebalansowanie składu sub-funduszy (pierwsza sesja miesiąca).

Optymalizowane są wagi tickerów wewnątrz każdego sub-funduszu; wagi między sub-funduszami (W)
nie są tu wyznaczane – pipeline zapisuje skład i scenariusze stóp zwrotu sub-funduszy, na których
W dobierane jest później (profil ryzyka).

Dla każdej daty rebalansowania t (pierwsza sesja miesiąca w danych):
    1. okno estymacji: ostatnie ``window_days`` sesji przed t (dane ≤ t−1, bez look-ahead);
    2. marginesy AR(1)–GJR-GARCH(1,1) skew-t i R-vine (3 drzewa) na resztach wszystkich aktywów
       (model zapisany dla daty t jest używany ponownie, ``--refit`` estymuje od nowa);
    3. scenariusze stóp zwrotu aktywów w horyzoncie H (start ze stanu GARCH na t−1);
    4. dla każdego sub-funduszu k z aktywami i ∈ I_k (LP Rockafellara–Uryaseva):
           max E[R_kᵀ w]  p.w.  ES_α(−R_kᵀ w) ≤ ES_α(−R_kᵀ w_EW),  Σ w_i = 1,
           0 ≤ w_i ≤ max(w_max, 1/|I_k|),
       gdzie w_EW – wagi równe; portfel równych wag jest zawsze dopuszczalny;
    5. (dla dat historycznych) zrealizowane stopy zwrotu sub-funduszy (skład optymalny i równe
       wagi, kupno i trzymanie do następnego rebalansowania), opcjonalnie netto po kosztach
       transakcyjnych c·Σ|w_t − w̃_t| (``transaction_cost``, w̃ – wagi po dryfie cen).

Przebieg jest przyrostowy: daty już policzone są pomijane (``--force`` liczy ponownie).
Daty historyczne używają ``backtest_scenarios``, najnowsza – ``n_scenarios`` scenariuszy.

Wyjście (artifacts/rebalance/):
    composition.parquet        – skład: date, subfund, ticker, weight, weight_ew
    subfunds.parquet           – E[R], ES, VaR sub-funduszy (skład optymalny i równe wagi)
    models.parquet             – okno estymacji, liczba aktywów, AIC i rodziny kopuł R-vine
    realized.parquet           – zrealizowane stopy zwrotu sub-funduszy (brutto; netto i koszty, gdy c > 0)
    turnover.parquet           – obrót ½Σ|w_t − w̃_t| względem wag po dryfie cen (Opt i EW)
    vines/<data>.json.gz       – R-vine dla daty; margins/<data>.parquet – parametry GARCH
    scenarios_latest.parquet   – scenariusze stóp zwrotu sub-funduszy (skład optymalny), ostatnia data
Raport: artifacts/reports/rebalance_summary.md

Uruchomienie:
    python -m pipelines.monthly_rebalance                 # wszystkie brakujące daty
    python -m pipelines.monthly_rebalance --date 2020-12-01
    python -m pipelines.monthly_rebalance --start 2019-01-01 --scenarios 1000 --jobs 4 --force
    python -m pipelines.monthly_rebalance --cost 0           # raport bez kosztów transakcyjnych
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from pf import ARTIFACTS_DIR, ROOT, config
from pf.artifacts import update_manifest, write_parquet
from pf.opt.cvar import empirical_es, max_return_es
from pf.portfolio import load_subfunds
from pf.sim.scenarios import Model, eligible_assets, estimate, load_model, simulate

log = logging.getLogger("monthly_rebalance")

OUT_DIR = ARTIFACTS_DIR / "rebalance"
REPORT_DIR = ARTIFACTS_DIR / "reports"
TABLES = ("composition", "subfunds", "models")


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


def optimize_subfund(R: pd.DataFrame, alpha: float, max_weight: float) -> tuple[pd.Series, dict]:
    """Skład sub-funduszu: max E[R] przy ES_α ≤ ES_α portfela równych wag (scenariusze × aktywa)."""
    n = R.shape[1]
    X = R.to_numpy()
    w_ew = np.full(n, 1.0 / n)
    es_ew = empirical_es(-X @ w_ew, alpha)
    cap = max(max_weight, 1.0 / n)
    if n == 1:
        w, sol = w_ew, None
    else:
        sol = max_return_es(X, alpha, es_ew + 1e-10 * max(1.0, abs(es_ew)), cap)
        w = sol.weights
    stats = {"n_assets": n, "cap": cap, "exp_return": float(X.mean(axis=0) @ w),
             "es": empirical_es(-X @ w, alpha), "var": float(sol.var) if sol else float(np.quantile(-X @ w, alpha)),
             "exp_return_ew": float(X.mean(axis=0) @ w_ew), "es_ew": es_ew,
             "feasible": bool(sol.feasible) if sol else True}
    return pd.Series(w, index=R.columns), stats


def rebalance_one(t: pd.Timestamp, returns: pd.DataFrame, cfg: dict, subfunds: dict, n_scen: int,
                  jobs: int | None, model: Model | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame, dict, Model, pd.DataFrame]:
    """Zwraca (skład, statystyki sub-funduszy, wiersz modelu, model, scenariusze sub-funduszy)."""
    rcfg = cfg["rebalance"]
    window = estimation_window(returns, t, rcfg["window_days"])
    if model is None:
        model = estimate(window, cfg, rcfg["min_obs"], jobs=jobs)
    rng = np.random.default_rng([rcfg["seed"], int(t.strftime("%Y%m%d"))])
    scen = simulate(model, n_scen, rcfg["horizon_days"], rng)
    comp, stats, sf_scen = [], [], {}
    available = set(model.assets)
    for sf, spec in subfunds.items():
        members = [a for a in spec["tickers"] if a in available]
        if not members:
            continue
        w, st = optimize_subfund(scen[members], rcfg["es_alpha"], rcfg["asset_max_weight"])
        comp += [{"date": t, "subfund": sf, "ticker": a, "weight": float(w[a]), "weight_ew": 1.0 / len(members)}
                 for a in members]
        stats.append({"date": t, "subfund": sf, **st})
        sf_scen[sf] = scen[members].to_numpy() @ w.to_numpy()
    families = pd.Series([e.copula.label() for e in model.vine.edges()]).value_counts()
    model_row = {"date": t, "window_start": window.index.min(), "window_end": window.index.max(),
                 "n_assets": len(model.assets), "panel_obs": model.panel_obs, "n_scenarios": n_scen,
                 "vine_aic": model.vine.aic(), **{f"n_{fam}": int(c) for fam, c in families.items()}}
    return pd.DataFrame(comp), pd.DataFrame(stats), model_row, model, pd.DataFrame(sf_scen)


def upsert(path, new: pd.DataFrame) -> pd.DataFrame:
    """Zastępuje w tabeli wiersze dat obecnych w ``new`` i zapisuje wynik."""
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old[~old["date"].isin(new["date"])], new], ignore_index=True)
    sort = [c for c in ("date", "subfund", "ticker") if c in new.columns]
    out = new.sort_values(sort).reset_index(drop=True)
    write_parquet(out, path)
    return out


def realized_table(returns: pd.DataFrame, comp: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dates = pd.DatetimeIndex(sorted(comp["date"].unique()))
    for t, t_next in zip(dates[:-1], dates[1:]):
        r = realized_returns(returns, t, t_next)
        for sf, g in comp[comp["date"] == t].groupby("subfund"):
            ra = r.reindex(g["ticker"]).fillna(0.0).to_numpy()
            rows.append({"date": t, "end": t_next, "subfund": sf, "r_opt": float(g["weight"].to_numpy() @ ra),
                         "r_ew": float(g["weight_ew"].to_numpy() @ ra)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.merge(stats[["date", "subfund", "var"]], on=["date", "subfund"], how="left")


def apply_costs(realized: pd.DataFrame, to: pd.DataFrame, cost: float) -> pd.DataFrame:
    """Stopy zwrotu netto po kosztach transakcyjnych w dniu rebalansowania t:

        k_t = c · Σ_i |w_{i,t} − w̃_{i,t}| = 2c · TO_t,    r_net = (1 − k_t)(1 + r) − 1,

    c – koszt jako ułamek wartości transakcji (kupna i sprzedaży). Przy zakupie początkowym
    (brak pozycji z poprzedniej daty, w̃ = 0) Σ|w_t| = 1, więc k_t = c.
    """
    out = realized.copy()
    key = pd.MultiIndex.from_frame(out[["date", "subfund"]])
    traded = 2.0 * to.set_index(["date", "subfund"])[["turnover_opt", "turnover_ew"]]
    for v in ("opt", "ew"):
        tr = traded[f"turnover_{v}"].reindex(key).to_numpy()
        out[f"cost_{v}"] = cost * np.where(np.isnan(tr), 1.0, tr)
        out[f"r_{v}_net"] = (1.0 - out[f"cost_{v}"]) * (1.0 + out[f"r_{v}"]) - 1.0
    return out


def drifted_weights(w_prev: pd.Series, r: pd.Series) -> pd.Series:
    """Wagi po okresie kupna i trzymania: w̃_i = w_i (1 + r_i) / Σ_j w_j (1 + r_j)."""
    g = w_prev * (1.0 + r.reindex(w_prev.index).fillna(0.0))
    return g / g.sum()


def turnover(comp: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Obrót w dniu rebalansowania t: TO_t = ½ Σ_i |w_{i,t} − w̃_{i,t}|.

    w̃_t – wagi z t−1 po dryfie cen w okresie [t−1, t) (portfel trzymany bez zmian), więc TO_t
    mierzy rzeczywistą transakcję z w̃_t do w_t. Liczone dla składu optymalnego i równych wag
    (które również wymagają przywrócenia po dryfie). Pierwsza data (zakup początkowy) jest pomijana.
    """
    dates = pd.DatetimeIndex(sorted(comp["date"].unique()))
    by_date = {t: g for t, g in comp.groupby("date")}
    rows = []
    for t_prev, t in zip(dates[:-1], dates[1:]):
        r = realized_returns(returns, t_prev, t)
        prev_sf, cur_sf = by_date[t_prev].groupby("subfund"), dict(tuple(by_date[t].groupby("subfund")))
        for sf, prev in prev_sf:
            if sf not in cur_sf:
                continue
            prev, cur = prev.set_index("ticker"), cur_sf[sf].set_index("ticker")
            row = {"date": t, "subfund": sf}
            for col, name in (("weight", "turnover_opt"), ("weight_ew", "turnover_ew")):
                drift = drifted_weights(prev[col], r)
                idx = drift.index.union(cur.index)
                row[name] = 0.5 * float((cur[col].reindex(idx, fill_value=0.0)
                                         - drift.reindex(idx, fill_value=0.0)).abs().sum())
            rows.append(row)
    return pd.DataFrame(rows, columns=["date", "subfund", "turnover_opt", "turnover_ew"])


def summary_report(comp: pd.DataFrame, stats: pd.DataFrame, realized: pd.DataFrame, to: pd.DataFrame,
                   subfunds: dict, rcfg: dict) -> str:
    def perf(r: pd.Series) -> tuple[float, float, float, float]:
        wealth = (1.0 + r).cumprod()
        ann = wealth.iloc[-1] ** (12.0 / len(r)) - 1.0
        vol = r.std() * np.sqrt(12.0)
        mdd = (wealth / wealth.cummax() - 1.0).min()
        return 100 * ann, 100 * vol, ann / vol if vol > 0 else np.nan, 100 * mdd

    holdings = comp[comp["weight"] > 1e-3].groupby(["subfund", "date"]).size().groupby(level="subfund").mean()
    n = realized["date"].nunique()
    lines = [
        "# Comiesięczne rebalansowanie składu sub-funduszy – podsumowanie",
        "",
        f"Okres: {realized['date'].min().date()} – {realized['end'].max().date()} ({n} miesięcy). "
        f"Okno estymacji {rcfg['window_days']} sesji, horyzont {rcfg['horizon_days']} sesji.",
        f"Skład: max E[R] przy ES_{rcfg['es_alpha']} ≤ ES portfela równych wag w sub-funduszu, "
        f"waga tickera ≤ {100 * rcfg['asset_max_weight']:.0f}% (lub 1/n przy mniejszej liczbie aktywów).",
        "Opt – skład optymalizowany, EW – równe wagi; zwrot i zmienność roczne [%], Sharpe przy r_f = 0, "
        "obrót – średni miesięczny ½Σ|w_t − w̃_t| [%] (w̃_t – wagi z poprzedniego rebalansowania po dryfie "
        "cen), VaR – liczba miesięcy ze stratą powyżej prognozowanego "
        f"VaR_{rcfg['es_alpha']} (oczekiwane {n * (1 - rcfg['es_alpha']):.1f}).",
        "",
        "| sub-fundusz | zwrot Opt | zwrot EW | zmienn. Opt | zmienn. EW | Sharpe Opt | Sharpe EW | "
        "max DD Opt | max DD EW | śr. liczba składników | obrót Opt | obrót EW | VaR |",
        "|---|" + "---:|" * 12,
    ]
    for sf, spec in subfunds.items():
        g = realized[realized["subfund"] == sf]
        if g.empty:
            continue
        a_o, v_o, s_o, d_o = perf(g["r_opt"])
        a_e, v_e, s_e, d_e = perf(g["r_ew"])
        breaches = int((-g["r_opt"] > g["var"]).sum())
        t_sf = to[to["subfund"] == sf]
        lines.append(f"| {sf} {spec['name']} | {a_o:.2f} | {a_e:.2f} | {v_o:.2f} | {v_e:.2f} | {s_o:.2f} | "
                     f"{s_e:.2f} | {d_o:.2f} | {d_e:.2f} | {holdings.get(sf, np.nan):.1f} | "
                     f"{100 * t_sf['turnover_opt'].mean():.1f} | {100 * t_sf['turnover_ew'].mean():.1f} | "
                     f"{breaches} |")
    cost = rcfg.get("transaction_cost") or 0.0
    if cost > 0 and "r_opt_net" in realized:
        lines += ["", f"## Po kosztach transakcyjnych ({100 * cost:.2f}% wartości transakcji)", "",
                  "Koszt w dniu rebalansowania c·Σ|w_t − w̃_t| (zakup początkowy: c); zwrot i koszt roczne [%].",
                  "", "| sub-fundusz | zwrot netto Opt | zwrot netto EW | Sharpe netto Opt | Sharpe netto EW | "
                  "koszt Opt | koszt EW |", "|---|" + "---:|" * 6]
        for sf, spec in subfunds.items():
            g = realized[realized["subfund"] == sf]
            if g.empty:
                continue
            a_o, _, s_o, _ = perf(g["r_opt_net"])
            a_e, _, s_e, _ = perf(g["r_ew_net"])
            lines.append(f"| {sf} {spec['name']} | {a_o:.2f} | {a_e:.2f} | {s_o:.2f} | {s_e:.2f} | "
                         f"{1200 * g['cost_opt'].mean():.3f} | {1200 * g['cost_ew'].mean():.3f} |")
    last = comp[comp["date"] == comp["date"].max()]
    lines += ["", f"## Skład na {last['date'].max().date()}", "",
              "| sub-fundusz | skład (waga > 0,1%) |", "|---|---|"]
    for sf, g in last.groupby("subfund"):
        g = g[g["weight"] > 1e-3].sort_values("weight", ascending=False)
        lines.append(f"| {sf} | " + ", ".join(f"{r.ticker} {100 * r.weight:.1f}%" for r in g.itertuples()) + " |")
    return "\n".join(lines) + "\n"


def cached_model(t: pd.Timestamp, models: pd.DataFrame | None) -> Model | None:
    tag = t.strftime("%Y-%m-%d")
    vine_path, margins_path = OUT_DIR / "vines" / f"{tag}.json.gz", OUT_DIR / "margins" / f"{tag}.parquet"
    if models is None or not (vine_path.exists() and margins_path.exists()):
        return None
    row = models[models["date"] == t]
    if row.empty:
        return None
    return load_model(vine_path, margins_path, row["window_end"].iloc[0], row["panel_obs"].iloc[0])


def main() -> None:
    cfg = config.load("models")
    rcfg = cfg["rebalance"]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", type=pd.Timestamp, default=None, help="pojedyncza data rebalansowania")
    parser.add_argument("--start", type=pd.Timestamp, default=None)
    parser.add_argument("--end", type=pd.Timestamp, default=None)
    parser.add_argument("--scenarios", type=int, default=None, help="liczba scenariuszy dla dat historycznych")
    parser.add_argument("--jobs", type=int, default=None, help="procesy do estymacji GARCH")
    parser.add_argument("--force", action="store_true", help="przelicz daty już obecne w wynikach")
    parser.add_argument("--refit", action="store_true", help="estymuj modele od nowa zamiast wczytywać zapisane")
    parser.add_argument("--cost", type=float, default=None,
                        help="koszt transakcyjny jako ułamek wartości transakcji (np. 0.001; 0 = bez kosztów)")
    args = parser.parse_args()
    if args.cost is not None:
        rcfg = {**rcfg, "transaction_cost": args.cost}
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

    paths = {name: OUT_DIR / f"{name}.parquet" for name in TABLES}
    models = pd.read_parquet(paths["models"]) if paths["models"].exists() else None
    if not args.force and paths["subfunds"].exists():
        dates = dates[~dates.isin(pd.DatetimeIndex(pd.read_parquet(paths["subfunds"])["date"]))]
    log.info("Daty do policzenia: %d (%s – %s)", len(dates), dates.min().date() if len(dates) else "-",
             dates.max().date() if len(dates) else "-")

    latest_date = all_dates.max()
    for t in dates:
        t0 = time.time()
        n_scen = rcfg["n_scenarios"] if t == latest_date else (args.scenarios or rcfg["backtest_scenarios"])
        model = None if args.refit else cached_model(t, models)
        reused = model is not None
        comp, stats, model_row, model, sf_scen = rebalance_one(t, returns, cfg, subfunds, n_scen, args.jobs, model)
        tag = t.strftime("%Y-%m-%d")
        if not reused:
            model.vine.save(OUT_DIR / "vines" / f"{tag}.json.gz")
            write_parquet(pd.DataFrame([f.to_dict() for f in model.margins.values()]),
                          OUT_DIR / "margins" / f"{tag}.parquet")
        if t == latest_date:
            write_parquet(sf_scen.astype("float32"), OUT_DIR / "scenarios_latest.parquet")
        # zapis po każdej dacie – przerwany przebieg można wznowić
        upsert(paths["composition"], comp)
        upsert(paths["subfunds"], stats)
        models = upsert(paths["models"], pd.DataFrame([model_row]))
        gain = 100 * (stats["exp_return"] - stats["exp_return_ew"]).mean()
        log.info("%s: %d sub-funduszy, śr. ΔE[R] vs EW = %+.2f p.p.%s (%.0f s)", tag, len(stats), gain,
                 " [model z pliku]" if reused else "", time.time() - t0)

    if not paths["subfunds"].exists():
        return
    comp, stats = pd.read_parquet(paths["composition"]), pd.read_parquet(paths["subfunds"])
    realized = realized_table(returns, comp, stats)
    if not realized.empty:
        to = turnover(comp, returns)
        if rcfg.get("transaction_cost"):
            realized = apply_costs(realized, to, rcfg["transaction_cost"])
        write_parquet(realized, OUT_DIR / "realized.parquet")
        write_parquet(to, OUT_DIR / "turnover.parquet")
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "rebalance_summary.md").write_text(
            summary_report(comp, stats, realized, to, subfunds, rcfg), encoding="utf-8")
    reb = pd.DatetimeIndex(stats["date"].unique())
    update_manifest("rebalance/composition", paths["composition"], n_rows=len(comp), params={
        "rebalance": rcfg, "rvine": cfg["rvine"], "margins": cfg["margins"], "returns": cfg["returns"]["path"],
        "first_date": str(reb.min().date()), "last_date": str(reb.max().date())})


if __name__ == "__main__":
    main()
