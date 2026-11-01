"""MC ex post na wynikach backtestu (Etap 5): miesięczne VaR_97,5% i ES_97,5% z SE dla każdego szeregu.

Szeregi: portfele (wariant bazowy: 7 modeli × 4 strategie × 5 profili + ew + mv_hist) i ich sub-fundusze,
dzienne stopy netto, osobno dla próby kalibracyjnej, hold-out i całości. Metody: FHS (GJR-GARCH skew-t,
bootstrap reszt, stan startowy z losowej daty) i bootstrap stacjonarny Politisa–Romano; dla strategii
głównej (``expost_mc.main_model`` × ``expost_mc.main_strategy``) także wielowymiarowe MC (marginesy GARCH
+ R-vine na 9 dziennych szeregach SF; wagi W z ostatniej daty próby i średnie W okresu).
N adaptacyjnie (SE(ÊS)/|ÊS| ≤ ``rel_se``, N ≤ ``n_max``). Porównanie: ES ex ante (średnia prognoz) i ES
historyczny (miesięczne stopy netto).

Wyjście: artifacts/backtest/<run_id>/es_mc.parquet; raport artifacts/reports/es_mc.md.

Uruchomienie:
    python -m pipelines.expost_mc --jobs 4
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from pf import ARTIFACTS_DIR, ROOT, config
from pf.artifacts import update_manifest, write_parquet
from pf.backtest.data import sample_label
from pf.backtest.expost_mc import FHS, MultiMC, adaptive, block_length, stationary_monthly_losses
from pf.opt.cvar import empirical_es
from pipelines.backtest import BACKTEST_DIR, current_run_id, load_all

log = logging.getLogger("expost_mc")
REPORT_DIR = ARTIFACTS_DIR / "reports"
SAMPLES = ("calibration", "holdout", "all")


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8"))


def univariate_task(args) -> list[dict]:
    sid, kind, sample, r, ecfg, alpha = args
    rows = []
    if len(r) < 2 * ecfg["horizon"] + 50:
        return rows
    H = ecfg["horizon"]
    kw = dict(alpha=alpha, batch=ecfg["batch"], n_min=ecfg["n_min"], n_max=ecfg["n_max"], rel_se=ecfg["rel_se"])
    t0 = time.time()
    try:
        fhs = FHS(r)
        rng = np.random.default_rng(_seed(sid, sample, "fhs"))
        res = adaptive(lambda b: fhs.monthly_losses(b, H, rng), **kw)
        rows.append({"sid": sid, "kind": kind, "sample": sample, "method": "fhs", "T_days": len(r),
                     "garch_converged": fhs.converged, "seconds": time.time() - t0, **res})
    except Exception as exc:  # noqa: BLE001
        rows.append({"sid": sid, "kind": kind, "sample": sample, "method": "fhs", "T_days": len(r), "error": str(exc)})
    t0 = time.time()
    b = block_length(r)
    seed = _seed(sid, sample, "sb")
    state = {"k": 0}

    def draw(n):
        state["k"] += 1
        return stationary_monthly_losses(r, n, H, b, seed + state["k"])

    res = adaptive(draw, **kw)
    rows.append({"sid": sid, "kind": kind, "sample": sample, "method": "stationary_bootstrap", "T_days": len(r),
                 "block": b, "seconds": time.time() - t0, **res})
    return rows


def multivariate_task(args) -> list[dict]:
    sid, sample, r_sf, W_last, W_mean, cfg, ecfg, alpha = args
    rows = []
    t0 = time.time()
    mm = MultiMC(r_sf, cfg)
    rng = np.random.default_rng(_seed(sid, sample, "mv"))
    kw = dict(alpha=alpha, batch=ecfg["batch"], n_min=ecfg["n_min"], n_max=ecfg["n_max"], rel_se=ecfg["rel_se"])
    for label, W in (("multivariate_W_last", W_last), ("multivariate_W_mean", W_mean)):
        w = W.reindex(mm.names).fillna(0.0).to_numpy()
        w = w / w.sum()
        res = adaptive(lambda b: -(mm.monthly_returns(b, ecfg["horizon"], rng) @ w), **kw)
        rows.append({"sid": sid, "kind": "portfolio", "sample": sample, "method": label, "T_days": len(r_sf),
                     "seconds": time.time() - t0, **res})
    return rows


def build_tasks(run_dir, bcfg, ecfg, alpha, cfg):
    daily = pd.read_parquet(run_dir / "daily.parquet")
    daily_sf = pd.read_parquet(run_dir / "daily_sf.parquet")
    base = daily[daily["sid"].str.endswith("|base")]
    labels = {t: sample_label(t, bcfg) for t in pd.DatetimeIndex(daily["date"].unique())}
    uni, multi = [], []
    for kind, df in (("portfolio", base), ("subfund", daily_sf)):
        df = df.assign(sample=df["date"].map(labels))
        for sid, g in df.groupby("sid"):
            g = g.sort_values("session")
            for smp in SAMPLES:
                x = g if smp == "all" else g[g["sample"] == smp]
                uni.append((sid, kind, smp, x["r_net"].to_numpy(dtype=float), ecfg, alpha))
    W = pd.read_parquet(run_dir / "weights_W.parquet")
    main = W[(W["model"] == ecfg["main_model"]) & (W["strategy"] == ecfg["main_strategy"]) & (W["variant"] == "base")]
    sf = daily_sf.assign(series=daily_sf["sid"].str.rsplit("|", n=1).str[0],
                         subfund=daily_sf["sid"].str.rsplit("|", n=1).str[1], sample=daily_sf["date"].map(labels))
    for prof, wp in main.groupby("profile"):
        sid = f"{ecfg['main_model']}|{ecfg['main_strategy']}|{prof}|base"
        s = sf[sf["series"] == sid]
        for smp in SAMPLES:
            x = s if smp == "all" else s[s["sample"] == smp]
            r_sf = x.pivot_table(index="session", columns="subfund", values="r_net").dropna(how="any").astype(float)
            wd = wp if smp == "all" else wp[wp["date"].map(lambda t: sample_label(t, bcfg)) == smp]
            if r_sf.empty or wd.empty:
                continue
            last = wd[wd["date"] == wd["date"].max()].set_index("subfund")["weight"]
            mean = wd.pivot_table(index="date", columns="subfund", values="weight").fillna(0.0).mean()
            multi.append((sid, smp, r_sf[sorted(r_sf.columns)], last, mean, cfg, ecfg, alpha))
    return uni, multi


def comparison(es: pd.DataFrame, run_dir, bcfg) -> pd.DataFrame:
    """ES ex ante (średnia prognoz) i historyczny (miesięczne stopy netto) obok wyników MC."""
    per = pd.read_parquet(run_dir / "periods.parquet")
    per_sf = pd.read_parquet(run_dir / "periods_sf.parquet")
    fc = pd.read_parquet(run_dir / "forecasts.parquet")
    alpha = 0.975
    rows = []
    for kind, p in (("portfolio", per), ("subfund", per_sf)):
        for (sid, smp), g in p.assign(sample=p["sample"]).groupby(["sid", "sample"]):
            rows.append({"sid": sid, "kind": kind, "sample": smp, "es_hist": empirical_es(-g["r_net"].to_numpy(), alpha),
                         "T_months": len(g)})
        for sid, g in p.groupby("sid"):
            rows.append({"sid": sid, "kind": kind, "sample": "all", "es_hist": empirical_es(-g["r_net"].to_numpy(), alpha),
                         "T_months": len(g)})
    hist = pd.DataFrame(rows)
    fcs = fc[fc["loss"].notna()]
    ex = pd.concat([fcs.groupby(["sid", "sample"])["es"].mean().rename("es_exante").reset_index(),
                    fcs.groupby("sid")["es"].mean().rename("es_exante").reset_index().assign(sample="all")])
    return es.merge(hist, on=["sid", "kind", "sample"], how="left").merge(ex, on=["sid", "sample"], how="left")


def report(tab: pd.DataFrame, ecfg: dict) -> str:
    port = tab[tab["kind"] == "portfolio"]
    lines = ["# MC ex post: miesięczne VaR i ES 97,5% zrealizowanych szeregów", "",
             f"Dzienne stopy netto, horyzont {ecfg['horizon']} sesji, N adaptacyjnie (SE(ÊS)/|ÊS| ≤ {ecfg['rel_se']}, "
             f"N ≤ {ecfg['n_max']}). FHS – GJR-GARCH(1,1) skew-t z bootstrapem reszt i losowym stanem startowym; "
             "SB – bootstrap stacjonarny Politisa–Romano; MV – marginesy GARCH + R-vine na 9 szeregach SF "
             "(strategia główna). ES ex ante – średnia prognoz modelu; ES hist. – z miesięcznych stóp netto. "
             "Wartości w %. Szereg strategii o zmiennych wagach nie jest ściśle stacjonarny.", ""]
    for smp in SAMPLES:
        p = port[port["sample"] == smp]
        if p.empty:
            continue
        piv = p.pivot_table(index="sid", columns="method", values="es")
        se = p.pivot_table(index="sid", columns="method", values="es_se")
        meta = p.drop_duplicates("sid").set_index("sid")[["es_exante", "es_hist", "T_months"]]
        lines += [f"## Próba: {smp}", "", "| szereg | ES FHS (SE) | ES SB (SE) | ES MV W_last | ES MV W_mean | "
                  "ES ex ante | ES hist. | T [mies.] |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for sid in sorted(piv.index):
            def cell(m):
                if m not in piv.columns or pd.isna(piv.loc[sid, m]):
                    return ""
                return f"{100 * piv.loc[sid, m]:.2f} ({100 * se.loc[sid, m]:.2f})"
            mm = meta.loc[sid]
            lines.append(f"| {sid.replace('|', ' / ')} | {cell('fhs')} | {cell('stationary_bootstrap')} | "
                         f"{cell('multivariate_W_last')} | {cell('multivariate_W_mean')} | "
                         f"{100 * mm.es_exante:.2f} | {100 * mm.es_hist:.2f} | {int(mm.T_months)} |")
        lines.append("")
    sub = tab[(tab["kind"] == "subfund") & (tab["sample"] == "all")]
    if not sub.empty:
        sub = sub.assign(subfund=sub["sid"].str.rsplit("|", n=1).str[1])
        agg = sub.pivot_table(index="subfund", columns="method", values="es", aggfunc="median")
        lines += ["## Sub-fundusze (całość, mediana po strategiach i profilach) – ES [%]", "",
                  "| SF | FHS | SB |", "|---|---:|---:|"]
        for k, r in agg.iterrows():
            lines.append(f"| {k} | {100 * r.get('fhs', np.nan):.2f} | {100 * r.get('stationary_bootstrap', np.nan):.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, bcfg, pcfg, subfunds = load_all()
    run_dir = BACKTEST_DIR / current_run_id(cfg, bcfg, pcfg, subfunds)
    ecfg = bcfg["expost_mc"]
    alpha = float(cfg["rebalance"]["es_alpha"])
    out_path = run_dir / "es_mc.parquet"
    if not args.report_only:
        uni, multi = build_tasks(run_dir, bcfg, ecfg, alpha, cfg)
        log.info("Zadania: %d jednowymiarowych, %d wielowymiarowych", len(uni), len(multi))
        rows, t0 = [], time.time()
        with ProcessPoolExecutor(max_workers=max(args.jobs, 1), mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(multivariate_task, m) for m in multi] + [ex.submit(univariate_task, u) for u in uni]
            for i, f in enumerate(as_completed(futs), 1):
                rows += f.result()
                if i % 200 == 0 or i == len(futs):
                    log.info("%d/%d zadań (%.1f min)", i, len(futs), (time.time() - t0) / 60)
        es = pd.DataFrame(rows)
        write_parquet(comparison(es, run_dir, bcfg), out_path)
    tab = pd.read_parquet(out_path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "es_mc.md").write_text(report(tab, ecfg), encoding="utf-8")
    update_manifest(f"backtest/{run_dir.name}/es_mc", out_path, n_rows=len(tab), params={"expost_mc": ecfg})


if __name__ == "__main__":
    main()
