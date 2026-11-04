"""Miesięczna nakładka opcyjna na tickery sub-funduszy: rejestr opcji, polityki P0–P3, NAV, portfele, walidacja.

Jedna komenda (offline, na artefaktach Etapu 1 i danych z ``data/processed``):

    python -m pipelines.options_monthly --jobs 4

Etapy (``--stages``, domyślnie wszystkie, w tej kolejności):
    ledger       – krok każdej daty t_j (równolegle, wznawialnie): model GARCH-kopuła z okna ≤ t_j − 1, σ_imp,
                   rejestr opcji całej siatki, decyzje P3 (warianty λ i wrażliwości);
                   artifacts/options/dates/<t>/ (poza repozytorium), pamięć podręczna modeli artifacts/options/cache/;
    consolidate  – rejestr ``artifacts/options/ledger.parquet``, sygnały i pozycje P0–P2, decyzje
                   ``artifacts/options/overlay_decisions.parquet``, NAV ``artifacts/subfunds_overlay_nav.parquet``;
    portfolio    – optymalizacja dwupoziomowa bez i z kolumnami SFk_OV (profile i strategie z
                   ``monthly_overlay.portfolio``), realizacja stóp ``artifacts/options/portfolio_periods.parquet``;
    validation   – replikacja Cboe BXM i PPUT silnikiem modułu ``artifacts/reports/options_validation.{csv,md}``
                   (brak szeregów → wpis ``status: skipped`` w manifeście), kalibracja atm_adj na zrzutach opcji;
    annual       – scenariusze roczne SF z nakładką miesiąc po miesiącu dla ostatniej daty (``--annual-date``);
    report       – ``artifacts/reports/options_monthly_backtest.md``.

``--refresh-data`` (sieć): indeksy zmienności, instrumenty bazowe, stopa USD i indeksy strategii Cboe przed etapami.
Pojedyncze daty: ``--dates 2008-10-01 2020-03-02``; ``--max-dates N`` – pierwsze N dat (test).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from pf import ARTIFACTS_DIR, ROOT, config
from pf.artifacts import update_manifest, write_parquet
from pf.backtest import overlay as bto
from pf.backtest.data import config_digest, sample_label
from pf.data.riskfree import period_rate
from pf.options import calendar, engine, paths, scenarios, validation
from pf.options.market import MarketData, load_market
from pf.portfolio import deterministic_tickers, load_subfunds

log = logging.getLogger("options_monthly")
OUT_DIR = ARTIFACTS_DIR / "options"
STAGES = ("ledger", "consolidate", "portfolio", "validation", "annual", "report")
PARTS = ("ledger", "state", "tickers", "decisions", "lp_info")
PORT_PARTS = ("weights", "level1", "exante")


# --------------------------------------------------------------------------- konfiguracja i daty

def load_configs() -> tuple[dict, dict, dict, dict, dict]:
    return (config.load("models"), config.load("backtest"), config.load("profiles"), load_subfunds(),
            config.load("options")["monthly_overlay"])


def overlay_periods(market: MarketData, cfg: dict, bcfg: dict, subfunds: dict) -> pd.DataFrame:
    """Okresy opcji na datach rebalansowania backtestu (``pipelines.backtest.all_dates``)."""
    from pipelines.backtest import all_dates

    dates = all_dates(market.r_pln, cfg, bcfg, subfunds)
    return calendar.overlay_periods(market.index, dates, int(bcfg["execution_lag"]),
                                    int(cfg["rebalance"]["horizon_days"]))


def manifest_root(out_dir: Path) -> Path:
    out_dir = Path(out_dir).resolve()
    return ARTIFACTS_DIR if out_dir.is_relative_to(ARTIFACTS_DIR.resolve()) else out_dir


def report_dir(out_dir: Path) -> Path:
    root = manifest_root(out_dir)
    return root / "reports"


def _hash_frame(df) -> str:
    h = pd.util.hash_pandas_object(df, index=True).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


STEP_KEYS_EXCLUDED = ("portfolio", "annual_mc", "validation", "stress_windows", "selected")


def step_digest(period: dict, market: MarketData, cfg: dict, ocfg: dict, subfunds: dict, seed: int) -> str:
    """Skrót konfiguracji i danych ≤ t_close (zmiana danych po t_close nie unieważnia daty)."""
    end = period["close"] if pd.notna(period["close"]) else period["open"]
    cut = lambda x: x.loc[x.index <= end]  # noqa: E731
    data = {"r_local": _hash_frame(cut(market.r_local)), "r_pln": _hash_frame(cut(market.r_pln)),
            "fx": _hash_frame(cut(market.fx)), "vol": _hash_frame(cut(market.vol)),
            "rates": _hash_frame(pd.concat([cut(market.rate_pln), cut(market.rate_usd)], axis=1)),
            "spx": _hash_frame(cut(market.spx)) if market.spx is not None else None}
    conf = {"ocfg": {k: v for k, v in ocfg.items() if k not in STEP_KEYS_EXCLUDED},
            "models": {k: cfg[k] for k in ("margins", "rebalance", "dependence")}, "subfunds": subfunds,
            "period": {k: str(v) for k, v in period.items()}, "seed": seed}
    return config_digest({"data": data, "conf": conf})[:16]


# --------------------------------------------------------------------------- etap: ledger (krok daty)

def run_date(period: dict, market: MarketData, cfg: dict, ocfg: dict, subfunds: dict, out_dir: Path, seed: int,
             force: bool = False) -> dict:
    tag = pd.Timestamp(period["date"]).strftime("%Y-%m-%d")
    folder = Path(out_dir) / "dates" / tag
    digest = step_digest(period, market, cfg, ocfg, subfunds, seed)
    done = folder / "done.json"
    if done.exists() and not force:
        old = json.loads(done.read_text())
        if old.get("digest") == digest:
            return old
    res = engine.date_step(period, market, cfg, ocfg, subfunds, seed, Path(out_dir) / "cache")
    folder.mkdir(parents=True, exist_ok=True)
    for name in PARTS:
        write_parquet(res[name], folder / f"{name}.parquet")
    st = res["state"].iloc[0]
    summary = {"date": tag, "digest": digest, "n_tickers": int(st["n_tickers"]), "seconds": float(st["sec_total"]),
               "n_p3_positions": int(len(res["decisions"]))}
    done.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


_CTX: dict = {}


def _init_worker() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    logging.getLogger("pf").setLevel(logging.WARNING)
    cfg, bcfg, pcfg, subfunds, ocfg = load_configs()
    _CTX.update(cfg=cfg, bcfg=bcfg, pcfg=pcfg, subfunds=subfunds, ocfg=ocfg, market=load_market(ocfg, bcfg, ROOT))


def _ledger_worker(args):
    period, out_dir, force = args
    c = _CTX
    return run_date(period, c["market"], c["cfg"], c["ocfg"], c["subfunds"], out_dir,
                    int(c["cfg"]["rebalance"]["seed"]), force)


def _parallel(fn, tasks: list, jobs: int, label: str) -> list:
    out, failed, t0 = [], {}, time.time()
    if not tasks:
        return out
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn"), initializer=_init_worker) as ex:
        futs = {ex.submit(fn, t): t[0]["date"] for t in tasks}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                out.append(f.result())
            except Exception as exc:  # noqa: BLE001 – błąd jednej daty nie przerywa przebiegu
                failed[str(pd.Timestamp(futs[f]).date())] = repr(exc)
                log.error("%s %s: BŁĄD %r", label, futs[f], exc)
                continue
            el = time.time() - t0
            log.info("%s %s (%d/%d, %.1f min, ETA %.1f min)", label, out[-1].get("date"), k, len(tasks), el / 60,
                     el / k * (len(tasks) - k) / 60)
    if failed:
        raise RuntimeError(f"Nieudane daty ({label}): {sorted(failed)}")
    return out


def run_ledger(periods: pd.DataFrame, out_dir: Path, jobs: int, force: bool, ctx: dict | None = None) -> None:
    rows = periods.to_dict("records")
    if jobs <= 1 or ctx is not None:
        c = ctx or _ensure_ctx()
        for k, per in enumerate(rows, 1):
            s = run_date(per, c["market"], c["cfg"], c["ocfg"], c["subfunds"], out_dir,
                         int(c["cfg"]["rebalance"]["seed"]), force)
            log.info("ledger %s (%d/%d, %.1f s)", s["date"], k, len(rows), s.get("seconds", 0.0))
        return
    _parallel(_ledger_worker, [(per, out_dir, force) for per in rows], jobs, "ledger")


def _ensure_ctx() -> dict:
    if not _CTX:
        _init_worker()
    return _CTX


def collect(out_dir: Path, sub: str, names, dates=None) -> dict[str, pd.DataFrame]:
    base = Path(out_dir) / sub
    tags = None if dates is None else {pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates}
    out = {}
    for name in names:
        files = sorted(f for f in base.glob(f"*/{name}.parquet") if tags is None or f.parent.name in tags)
        frames = [pd.read_parquet(f) for f in files]
        frames = [f for f in frames if len(f)]
        out[name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out


# --------------------------------------------------------------------------- etap: consolidate

def consolidate(periods: pd.DataFrame, market: MarketData, cfg: dict, bcfg: dict, ocfg: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    parts = collect(out_dir, "dates", PARTS, periods["date"])
    ledger, state, tick = parts["ledger"], parts["state"], parts["tickers"]
    if ledger.empty:
        raise SystemExit("Brak policzonych dat – uruchom najpierw etap ledger")
    state = state.sort_values("date").reset_index(drop=True)
    sig = engine.signals_table(state, tick, market, ocfg)
    dec = pd.concat([engine.p012_decisions(ledger, sig, ocfg), parts["decisions"]], ignore_index=True)
    nav_p, dec_out = engine.nav_table(ledger, dec, ocfg)
    opens = state.set_index("date")[["open", "close"]]
    rf = {d: period_rate(market.rate_pln, o, c, bcfg["riskfree"]["day_count"]) if pd.notna(c) else np.nan
          for d, (o, c) in opens.iterrows()}
    nav_p["rf"] = nav_p["date"].map(rf)
    nav_p["sample"] = [sample_label(t, bcfg) or "other" for t in nav_p["date"]]
    for c in ("subfund", "policy", "variant", "kind"):
        nav_p[c] = nav_p[c].astype(str)
    led = ledger.copy()
    for c in ("subfund", "ticker", "currency", "type", "iv_source"):
        led[c] = led[c].astype("category")
    root = manifest_root(out_dir)
    paths_out = {"ledger": out_dir / "ledger.parquet", "decisions": out_dir / "overlay_decisions.parquet",
                 "signals": out_dir / "signals.parquet", "state": out_dir / "market_state.parquet",
                 "lp_info": out_dir / "lp_info.parquet", "nav": root / "subfunds_overlay_nav.parquet"}
    write_parquet(led, paths_out["ledger"])
    write_parquet(dec_out, paths_out["decisions"])
    write_parquet(sig, paths_out["signals"])
    write_parquet(state, paths_out["state"])
    write_parquet(parts["lp_info"], paths_out["lp_info"])
    write_parquet(nav_p, paths_out["nav"])
    m_bar = float(np.nanmedian(state["m"]))
    params = manifest_params(ocfg, state, m_bar)
    update_manifest("options/ledger", paths_out["ledger"], params=params, n_rows=len(led), root=root)
    update_manifest("options/overlay_decisions", paths_out["decisions"], n_rows=len(dec_out), root=root,
                    params={"policies": ocfg["policies"], "rules": ocfg["rules"], "lp": ocfg["lp"]})
    update_manifest("subfunds_overlay_nav", paths_out["nav"], n_rows=len(nav_p), root=root,
                    params={"kappa": ocfg["kappa"], "variants": [v["variant"] for v in engine.variant_specs(ocfg)],
                            "weights": "równe w SF (reguła macierzy A)", "m_bar": m_bar})
    log.info("Konsolidacja: %d wierszy rejestru, %d decyzji, %d okresów NAV", len(led), len(dec_out), len(nav_p))
    return {"ledger": ledger, "state": state, "tickers": tick, "signals": sig, "decisions": dec, "decisions_out": dec_out,
            "nav": nav_p, "m_bar": m_bar}


def manifest_params(ocfg: dict, state: pd.DataFrame, m_bar: float) -> dict:
    return {"moneyness": ocfg["moneyness"], "band": ocfg["band"], "kappa": ocfg["kappa"],
            "implied_vol": ocfg["implied_vol"], "m_bar": m_bar, "atm_adj": ocfg["implied_vol"].get("atm_adj", 1.0),
            "first_date": str(pd.Timestamp(state["date"].min()).date()),
            "last_date": str(pd.Timestamp(state["date"].max()).date()), "n_dates": int(len(state)),
            "dividends": ocfg.get("dividends"), "note": "opcje hipotetyczne, wycena modelowa"}


# --------------------------------------------------------------------------- etap: portfolio

def run_portfolio_date(period: dict, market: MarketData, cfg: dict, ocfg: dict, pcfg: dict, subfunds: dict,
                       decisions: pd.DataFrame, out_dir: Path, seed: int, force: bool = False) -> dict:
    tag = pd.Timestamp(period["date"]).strftime("%Y-%m-%d")
    folder = Path(out_dir) / "portfolio" / tag
    pos_dec = decisions[decisions["date"] == period["date"]]
    digest = config_digest({"step": step_digest(period, market, cfg, ocfg, subfunds, seed), "port": ocfg["portfolio"],
                            "profiles": pcfg, "selected": ocfg["selected"], "kappa": ocfg["kappa"],
                            "dec": _hash_frame(pos_dec.reset_index(drop=True)) if len(pos_dec) else ""})[:16]
    done = folder / "done.json"
    if done.exists() and not force and json.loads(done.read_text()).get("digest") == digest:
        return json.loads(done.read_text())
    all_tickers = [t for spec in subfunds.values() for t in spec["tickers"]]
    model = scenarios.fit_overlay_model(market, pd.Timestamp(period["date"]), cfg, ocfg, all_tickers,
                                        deterministic_tickers(subfunds), Path(out_dir) / "cache")
    positions = {sf: g[["ticker", "type", "moneyness", "h", "premium_rel", "delta"]].reset_index(drop=True)
                 for sf, g in pos_dec.groupby("subfund")}
    res = engine.portfolio_step(period, market, model, cfg, ocfg, pcfg, subfunds, positions, seed)
    folder.mkdir(parents=True, exist_ok=True)
    for name in PORT_PARTS:
        write_parquet(res[name], folder / f"{name}.parquet")
    summary = {"date": tag, "digest": digest, "seconds": res["seconds"]}
    done.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def _portfolio_worker(args):
    period, out_dir, force, dec_path = args
    c = _CTX
    if "decisions" not in c or c.get("dec_path") != dec_path:
        c["decisions"], c["dec_path"] = selected_decisions(pd.read_parquet(dec_path), c["ocfg"]), dec_path
    return run_portfolio_date(period, c["market"], c["cfg"], c["ocfg"], c["pcfg"], c["subfunds"], c["decisions"],
                              out_dir, int(c["cfg"]["rebalance"]["seed"]), force)


def selected_decisions(decisions_out: pd.DataFrame, ocfg: dict) -> pd.DataFrame:
    d = decisions_out[(decisions_out["policy"] == ocfg["selected"]) & (decisions_out["variant"] == "base")]
    return d.reset_index(drop=True)


def run_portfolio(periods: pd.DataFrame, market: MarketData, cfg: dict, bcfg: dict, ocfg: dict, pcfg: dict,
                  subfunds: dict, out_dir: Path, jobs: int, force: bool, ctx: dict | None = None) -> pd.DataFrame:
    dec_path = Path(out_dir) / "overlay_decisions.parquet"
    if not dec_path.exists():
        raise SystemExit("Brak overlay_decisions.parquet – uruchom etap consolidate")
    rows = periods.to_dict("records")
    if jobs <= 1 or ctx is not None:
        dec = selected_decisions(pd.read_parquet(dec_path), ocfg)
        for k, per in enumerate(rows, 1):
            s = run_portfolio_date(per, market, cfg, ocfg, pcfg, subfunds, dec, out_dir,
                                   int(cfg["rebalance"]["seed"]), force)
            log.info("portfolio %s (%d/%d, %.1f s)", s["date"], k, len(rows), s.get("seconds", 0.0))
    else:
        _parallel(_portfolio_worker, [(per, out_dir, force, dec_path) for per in rows], jobs, "portfolio")
    parts = collect(out_dir, "portfolio", PORT_PARTS, periods["date"])
    dec = selected_decisions(pd.read_parquet(dec_path), ocfg)
    pos_by_date = {d: {sf: g.reset_index(drop=True) for sf, g in gd.groupby("subfund")} for d, gd in dec.groupby("date")}
    pp = engine.portfolio_realize(periods, market, parts["weights"], parts["level1"], parts["exante"], pos_by_date, ocfg,
                                  float(bcfg["transaction_cost"]), int(bcfg["execution_lag"]),
                                  int(bcfg["riskfree"]["day_count"]))
    pp["sample"] = [sample_label(t, bcfg) or "other" for t in pp["date"]]
    root = manifest_root(out_dir)
    write_parquet(parts["weights"], Path(out_dir) / "portfolio_weights.parquet")
    write_parquet(parts["exante"], Path(out_dir) / "portfolio_exante.parquet")
    write_parquet(pp, Path(out_dir) / "portfolio_periods.parquet")
    update_manifest("options/portfolio_periods", Path(out_dir) / "portfolio_periods.parquet", n_rows=len(pp), root=root,
                    params={"portfolio": ocfg["portfolio"], "selected": ocfg["selected"],
                            "transaction_cost": bcfg["transaction_cost"], "execution_lag": bcfg["execution_lag"]})
    return pp


# --------------------------------------------------------------------------- etap: validation

def atm_adj_estimate() -> dict:
    """atm_adj z kwotowań SPY (DoltHub / zrzuty yfinance), jeśli dostępne."""
    try:
        from pipelines import calibrate_surface as cs
        from pf.options import implied_vol, surface

        oc = config.load("options")
        quotes = cs.load_quotes("SPY")
        und = pd.read_parquet(ARTIFACTS_DIR / "market" / "underlyings.parquet")
        vol = pd.read_parquet(ARTIFACTS_DIR / "market" / "vol_indices.parquet")
        obs = surface.prepare_quotes(quotes, vol, und[("IRX", "Close")].dropna(), oc["surface"], oc["iv_filters"])
        return implied_vol.calibrate_atm_adj(obs)
    except (SystemExit, Exception) as exc:  # noqa: BLE001
        return {"status": "unavailable", "reason": str(exc)}


def run_validation(ocfg: dict, out_dir: Path, root: Path = ROOT) -> pd.DataFrame | None:
    d = ocfg["data"]
    mroot = manifest_root(out_dir)
    paths_needed = [root / d["strategy_indices"], root / d["underlyings"], root / d["vol_indices"]]
    if not all(p.exists() for p in paths_needed):
        update_manifest("reports/options_validation", root=mroot,
                        params={"status": "skipped", "reason": "brak indeksów BXM/PPUT lub danych S&P 500"})
        log.warning("Walidacja BXM/PPUT pominięta (brak danych)")
        return None
    idx = pd.read_parquet(paths_needed[0])
    und = pd.read_parquet(paths_needed[1])
    vol = pd.read_parquet(paths_needed[2])
    y_path = root / d["usd_rate"]
    y = pd.read_parquet(y_path)["rate"] if y_path.exists() else und[("IRX", "Close")].dropna() / 100.0
    est = atm_adj_estimate()
    icfg = ocfg["implied_vol"]
    variants = [("base", icfg, None)]
    if est.get("status") == "ok":
        variants.append((f"atm_adj={est['atm_adj']:.3f}", {**icfg, "atm_adj": est["atm_adj"]}, None))
    sb = ocfg.get("sensitivity", {}).get("beta")
    if sb is not None:
        variants.append((f"beta={float(sb):g}", icfg, float(sb)))
    rows, status = [], {}
    for name, spec in ocfg["validation"].items():
        if name not in idx.columns or idx[name].dropna().empty:
            status[name] = "skipped"
            continue
        for vname, ic, beta in variants:
            rep = validation.replicate_monthly(und[("SPX", "Close")], und[("SPXTR", "Close")], vol, y, spec, ic,
                                               int(ocfg.get("trading_days", 252)), beta)
            tab = validation.tracking(rep, idx[name], ocfg["stress_windows"])
            rows.append(tab.assign(index=name, variant=vname))
        status[name] = "ok"
    if not rows:
        update_manifest("reports/options_validation", root=mroot, params={"status": "skipped", "indices": status})
        return None
    rep = pd.concat(rows, ignore_index=True)
    rdir = report_dir(out_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    rep.to_csv(rdir / "options_validation.csv", index=False)
    lines = ["# Walidacja wyceny nakładki: replikacja Cboe BXM i PPUT", "",
             "Silnik modułu (BSM, σ = atm_adj·σ_VIX(τ) + β ln(K/F), τ = n/252, rolowanie w 3. piątek, rozliczenie po "
             "zamknięciu). TE = σ(r_rep − r_off)·√12 na stopach między rolowaniami.", "",
             f"Estymata atm_adj z kwotowań SPY: {json.dumps(est, ensure_ascii=False)}", "",
             "| indeks | wariant | okres | mies. | TE [% p.a.] | śr. różnica [% p.a.] | replika [% p.a.] | indeks [% p.a.] | korelacja |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r.index} | {r.variant} | {r.period} | {r.n_months} | {r.te_ann_pct:.2f} | {r.mean_diff_ann_pct:.2f} | "
              f"{r.ret_replica_ann_pct:.2f} | {r.ret_official_ann_pct:.2f} | {r.corr:.3f} |" for r in rep.itertuples()]
    (rdir / "options_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_manifest("reports/options_validation", rdir / "options_validation.csv", n_rows=len(rep), root=mroot,
                    params={"status": "ok", "indices": status, "atm_adj_estimate": est,
                            "validation": ocfg["validation"]})
    return rep


# --------------------------------------------------------------------------- etap: annual

def frozen_p3(decisions: pd.DataFrame, ledger: pd.DataFrame, t: pd.Timestamp, policy_name: str = "P3") -> pd.DataFrame:
    """Średnie hedge ratio P3 (wariant bazowy) z dat ≤ t: Σ_t h / liczba dat z tickerem w rejestrze."""
    d = decisions[(decisions["policy"] == policy_name) & (decisions["variant"] == "base") & (decisions["date"] <= t)]
    n = ledger[ledger["date"] <= t].groupby("ticker", observed=True)["date"].nunique()
    if d.empty:
        return pd.DataFrame(columns=["ticker", "type", "moneyness", "h"])
    s = d.groupby(["ticker", "type", "moneyness"])["h"].sum().reset_index()
    s["h"] = s["h"] / s["ticker"].map(n).astype(float)
    return s


def run_annual(cons: dict, market: MarketData, cfg: dict, ocfg: dict, subfunds: dict, out_dir: Path,
               date: pd.Timestamp | None = None) -> pd.DataFrame:
    state, ledger, tick = cons["state"], cons["ledger"], cons["tickers"]
    t = pd.Timestamp(date) if date is not None else state["date"].max()
    t = state.loc[state["date"] <= t, "date"].max()
    acfg = ocfg["annual_mc"]
    all_tickers = [tk for spec in subfunds.values() for tk in spec["tickers"]]
    model = scenarios.fit_overlay_model(market, t, cfg, ocfg, all_tickers, deterministic_tickers(subfunds),
                                        Path(out_dir) / "cache")
    hist_state = state[state["date"] <= t]
    m_bar = float(np.nanmedian(hist_state["m"]))
    rates = ledger[ledger["date"] == t].drop_duplicates("ticker").set_index("ticker")["r"]
    rates = rates.set_axis(rates.index.astype(str))
    sf = {k: [x for x in v if x in rates.index] for k, v in engine.overlay_subfunds(subfunds, ocfg).items()}
    sf = {k: v for k, v in sf.items() if v}
    opens = hist_state.set_index("date")["open"]
    tickers = sorted({x for v in sf.values() for x in v})
    prices = pd.DataFrame({d: market.spot(o, tickers) for d, o in opens.items()}).T
    th = tick[tick["date"] <= t]
    history = {"m": hist_state["m"], "sigma_hat": th.pivot_table(index="date", columns="ticker", values="sigma_hat"),
               "prices": prices}
    pols = {}
    for name in acfg["policies"]:
        pols[name] = {"kind": "frozen"} if name == "P3_frozen" else ocfg["policies"][name]
    R = paths.simulate_overlay_paths(model, sf, pols, rates, m_bar, int(acfg["months"]), int(acfg["sessions_per_month"]),
                                     int(acfg["n_scenarios"]), int(cfg["rebalance"]["seed"]), ocfg, history,
                                     frozen_p3(cons["decisions"], ledger, t), cfg["margins"].get("sim_eps_cap"))
    stats = paths.summary(R, float(ocfg["lp"]["alpha"])).assign(date=t, m_bar=m_bar)
    write_parquet(stats, Path(out_dir) / "annual_mc.parquet")
    update_manifest("options/annual_mc", Path(out_dir) / "annual_mc.parquet", n_rows=len(stats),
                    root=manifest_root(out_dir), params={"annual_mc": acfg, "date": str(t.date()), "m_bar": m_bar})
    return stats


# --------------------------------------------------------------------------- etap: report

def write_report(cons: dict, pp: pd.DataFrame | None, val: pd.DataFrame | None, annual: pd.DataFrame | None,
                 ocfg: dict, pcfg: dict, out_dir: Path) -> Path:
    alpha = float(ocfg["lp"]["alpha"])
    nav_p = cons["nav"]
    sfm = bto.sf_metrics(nav_p[nav_p["r_ov"].notna()], alpha)
    stress = bto.stress_table(nav_p, ocfg["stress_windows"])
    gam = {k: float(v["gamma"]) for k, v in pcfg["profiles"].items()}
    pfm = bto.portfolio_metrics(pp, gam, alpha) if pp is not None and len(pp) else None
    st = cons["state"]
    meta = {"first_date": str(pd.Timestamp(st["date"].min()).date()), "last_date": str(pd.Timestamp(st["date"].max()).date()),
            "n_dates": len(st), "kappa": ocfg["kappa"], "band": ocfg["band"], "m_bar": cons["m_bar"],
            "atm_adj": ocfg["implied_vol"].get("atm_adj", 1.0), "selected": ocfg["selected"],
            "annual_date": str(pd.Timestamp(annual["date"].iloc[0]).date()) if annual is not None and len(annual) else None}
    md = bto.report(sfm, stress, pfm, val, annual, meta)
    rdir = report_dir(out_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / "options_monthly_backtest.md"
    path.write_text(md, encoding="utf-8")
    write_parquet(sfm, Path(out_dir) / "sf_metrics.parquet")
    if pfm is not None:
        write_parquet(pfm, Path(out_dir) / "portfolio_metrics.parquet")
    update_manifest("reports/options_monthly_backtest", path, root=manifest_root(out_dir),
                    params={"n_series_sf": int(len(sfm)), "portfolio": pfm is not None})
    return path


def load_consolidated(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    root = manifest_root(out_dir)
    cons = {"ledger": pd.read_parquet(out_dir / "ledger.parquet"), "state": pd.read_parquet(out_dir / "market_state.parquet"),
            "nav": pd.read_parquet(root / "subfunds_overlay_nav.parquet"),
            "decisions": pd.read_parquet(out_dir / "overlay_decisions.parquet")}
    cons["tickers"] = collect(out_dir, "dates", ["tickers"])["tickers"]   # decyzje: wariant bazowy wszystkich polityk
    for c in ("ticker", "subfund"):
        cons["ledger"][c] = cons["ledger"][c].astype(str)
    cons["m_bar"] = float(np.nanmedian(cons["state"]["m"]))
    return cons


# --------------------------------------------------------------------------- main

def refresh_data() -> None:
    from pf.data import cboe_strategy_indices, underlyings, usd_rate, vol_indices

    for step in (vol_indices.main, underlyings.main, usd_rate.main, cboe_strategy_indices.main):
        step()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--stages", nargs="*", choices=STAGES, default=list(STAGES))
    parser.add_argument("--dates", nargs="*", type=pd.Timestamp, default=None)
    parser.add_argument("--max-dates", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--annual-date", type=pd.Timestamp, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("pf").setLevel(logging.WARNING)
    if args.refresh_data:
        refresh_data()
    cfg, bcfg, pcfg, subfunds, ocfg = load_configs()
    market = load_market(ocfg, bcfg, ROOT)
    periods = overlay_periods(market, cfg, bcfg, subfunds)
    if args.dates:
        idx = pd.DatetimeIndex(periods["date"])
        periods = periods[periods["date"].isin([idx[min(idx.searchsorted(d), len(idx) - 1)] for d in args.dates])]
    if args.max_dates:
        periods = periods.iloc[:args.max_dates]
    log.info("Daty: %d (%s – %s), źródło stopy USD: %s", len(periods), periods["date"].min().date(),
             periods["date"].max().date(), market.info.get("usd_rate_source"))
    out_dir = args.out_dir
    cons = pp = val = annual = None
    if "ledger" in args.stages:
        run_ledger(periods, out_dir, args.jobs, args.force)
    if "consolidate" in args.stages:
        cons = consolidate(periods, market, cfg, bcfg, ocfg, out_dir)
    if "portfolio" in args.stages and ocfg["portfolio"].get("enabled", True):
        pp = run_portfolio(periods, market, cfg, bcfg, ocfg, pcfg, subfunds, out_dir, args.jobs, args.force)
    if "validation" in args.stages:
        val = run_validation(ocfg, out_dir)
    if cons is None and ({"annual", "report"} & set(args.stages)):
        cons = load_consolidated(out_dir)
    if "annual" in args.stages:
        annual = run_annual(cons, market, cfg, ocfg, subfunds, out_dir, args.annual_date)
    if "report" in args.stages:
        if pp is None and (Path(out_dir) / "portfolio_periods.parquet").exists():
            pp = pd.read_parquet(Path(out_dir) / "portfolio_periods.parquet")
        if val is None and (report_dir(out_dir) / "options_validation.csv").exists():
            val = pd.read_csv(report_dir(out_dir) / "options_validation.csv")
        if annual is None and (Path(out_dir) / "annual_mc.parquet").exists():
            annual = pd.read_parquet(Path(out_dir) / "annual_mc.parquet")
        path = write_report(cons, pp, val, annual, ocfg, pcfg, out_dir)
        log.info("Raport: %s", path)


if __name__ == "__main__":
    main()
