"""Ranking strategii, profili i modeli zależności; raporty i wykresy backtestu (Etap 6).

Wejście: tabele zbiorcze artifacts/backtest/<run_id>/ (``pipelines.backtest``) i ``es_mc.parquet``
(``pipelines.expost_mc``, opcjonalnie). Wszystkie miary na miesięcznych stopach netto w PLN, osobno dla
próby kalibracyjnej i hold-out (wnioski końcowe – hold-out).

Wyjście (artifacts/reports/): backtest_summary.md, ranking_calibration.md, ranking_holdout.md,
dependence_models.md; tabele ranking_<próba>.parquet w katalogu przebiegu; wykresy figures/backtest/.

Uruchomienie:
    python -m pipelines.ranking
"""

from __future__ import annotations

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pf import ARTIFACTS_DIR, ROOT  # noqa: E402
from pf.artifacts import update_manifest, write_parquet  # noqa: E402
from pf.backtest import metrics  # noqa: E402
from pf.backtest.ranking import add_ranks, crra_loss, mcs, monotone_kappa, series_metrics, sharpe_diff_test  # noqa: E402
from pf.opt.profiles import load_profiles  # noqa: E402
from pipelines.backtest import BACKTEST_DIR, current_run_id, load_all  # noqa: E402

log = logging.getLogger("ranking")
REPORT_DIR = ARTIFACTS_DIR / "reports"
FIG_DIR = ROOT / "figures" / "backtest"
SAMPLES = ("calibration", "holdout")
LABEL = {"calibration": "kalibracyjna", "holdout": "hold-out"}


def pct(x, d=2):
    return "" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}"


def num(x, d=2):
    return "" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def short(sid: str) -> str:
    m, s, p, _ = sid.split("|")
    return f"{s}" if m == "-" else f"{m}/{s}"


# --------------------------------------------------------------------------- ranking strategii

def rank_sample(periods, tests, gammas, rcfg, alpha, es_expost, sample) -> pd.DataFrame:
    base = periods[(periods["variant"] == "base") & (periods["sample"] == sample)]
    met = series_metrics(base, gammas, alpha, es_expost)
    bench = rcfg["sharpe_test"]["benchmark"]
    xb = base[base["sid"] == bench].set_index("date")
    xb = (xb["r_net"] - xb["rf"])
    out = []
    for p, gam in gammas.items():
        tab = met[met["eval_profile"] == p].copy()
        sids = tab["sid"].tolist()
        piv = base[base["sid"].isin(sids)].pivot_table(index="date", columns="sid", values="r_net").dropna()
        rf = base.drop_duplicates("date").set_index("date")["rf"].reindex(piv.index)
        losses = pd.DataFrame({s: crra_loss(piv[s], rf, gam) for s in piv.columns}, index=piv.index)
        m = mcs(losses, rcfg["mcs_size"], rcfg["mcs_reps"], seed=hash(p) % 2 ** 31)
        tab = tab.join(m, on="sid")
        sh = []
        for s in tab["sid"]:
            if s == bench:
                sh.append({"sid": s, "p_sharpe_vs_ew": np.nan, "sr_diff_ann": 0.0})
                continue
            xs = base[base["sid"] == s].set_index("date")
            xs = (xs["r_net"] - xs["rf"]).reindex(xb.index).dropna()
            r = sharpe_diff_test(xs.to_numpy(), xb.reindex(xs.index).to_numpy(), rcfg["sharpe_test"]["block"],
                                 rcfg["sharpe_test"]["reps"], seed=len(sh))
            sh.append({"sid": s, **r})
        tab = tab.merge(pd.DataFrame(sh), on="sid", how="left")
        t = tests[tests["sample"] == sample].set_index("sid")
        for c in ("p_uc", "p_binom", "p_ind", "p_cc", "p_z2", "p_de_uc", "p_de_cc5", "exceedances", "expected"):
            if c in t.columns:
                tab[c] = tab["sid"].map(t[c])
        out.append(add_ranks(tab, rcfg["composite_weights"]).sort_values("ce_ann", ascending=False))
    return pd.concat(out, ignore_index=True)


def ranking_markdown(tab: pd.DataFrame, sample: str, profiles, rcfg, n_months: int) -> str:
    lines = [f"# Ranking strategii – próba {LABEL[sample]}", "",
             f"Miesięczne stopy netto w PLN ({n_months} miesięcy), koszt 0,1% wartości transakcji. CE – ekwiwalent "
             "pewności CRRA (γ profilu) stóp nadwyżkowych nad WIBOR 1M, w skali rocznej. MCS – przynależność do "
             f"Model Confidence Set (α = {rcfg['mcs_size']}, strata −u_γ(1 + r̃)); p(SR) – test Ledoita–Wolfa "
             "różnicy Sharpe'a względem ew; p VaR/ES – Kupiec (uc), Christoffersen (cc), Acerbi–Szekely Z₂, "
             "Du–Escanciano (DE). Rangi: 1 – najlepsza; ranga złożona – średnia ważona rang "
             f"({', '.join(f'{k}: {v:g}' for k, v in rcfg['composite_weights'].items())}).", ""]
    if sample == "holdout":
        lines += ["Hold-out obejmuje ok. 69 miesięcy – oczekiwana liczba przekroczeń VaR 97,5% to ok. 1,7, więc testy "
                  "VaR/ES mają niską moc.", ""]
    for p, prof in profiles.items():
        t = tab[tab["eval_profile"] == p]
        if t.empty:
            continue
        lines += [f"## {p} – {prof.name} (γ = {prof.gamma:g}, κ = {prof.kappa:g})", "",
                  "| # | szereg | CE [%] | zwrot [%] | zmienn. [%] | Sharpe (SE) | p(SR) | Sortino | max DD [%] | "
                  "Calmar | Omega | STARR | obrót [%] | MCS | przekr. | p uc | p cc | p Z₂ | p DE | r. Sharpe | "
                  "r. DD | r. złożona |", "|---:|---|" + "---:|" * 15 + "---:|---:|---:|---:|"]
        for i, r in enumerate(t.itertuples(), 1):
            lines.append(
                f"| {i} | {short(r.sid)} | {pct(r.ce_ann)} | {pct(r.ann_return)} | {pct(r.ann_vol)} | "
                f"{num(r.sharpe)} ({num(r.sharpe_se)}) | {num(getattr(r, 'p_sharpe_vs_ew', np.nan), 3)} | "
                f"{num(r.sortino)} | {pct(r.max_dd, 1)} | {num(r.calmar)} | {num(r.omega)} | "
                f"{num(getattr(r, 'starr', np.nan), 3)} | {pct(r.turnover, 1)} | {'tak' if r.in_mcs else ''} | "
                f"{'' if pd.isna(getattr(r, 'exceedances', np.nan)) else int(r.exceedances)} | "
                f"{num(getattr(r, 'p_uc', np.nan), 3)} | {num(getattr(r, 'p_cc', np.nan), 3)} | "
                f"{num(getattr(r, 'p_z2', np.nan), 3)} | {num(getattr(r, 'p_de_uc', np.nan), 3)} | "
                f"{int(r.rank_sharpe)} | {int(r.rank_max_dd)} | {num(r.rank_composite, 1)} |")
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- modele zależności

def dependence_ranking(gof, logscore, tests, ranking_tabs, rcfg, bcfg) -> tuple[pd.DataFrame, str]:
    from pf.backtest.data import sample_label

    level = rcfg["test_level"]
    rows, lines = [], ["# Ranking modeli zależności", "",
                       "AIC/BIC – średnie po datach (okno estymacji); Vuong – liczba dat, w których R-vine jest "
                       "istotnie lepszy / gorszy (poziom 5%, poprawka Schwarza); MCS – Model Confidence Set na dziennej "
                       f"stracie −log-score poza próbą (α = {rcfg['mcs_size']}); testy – odsetek szeregów (strategie × "
                       f"profile) bez odrzucenia na poziomie {level} w teście Kupca, Christoffersena (cc), Z₂ i "
                       "Du–Escanciano; śr. ranga – średnia ranga CE portfeli modelu w każdej parze (strategia, profil) "
                       "spośród 7 modeli (1 – najlepszy).", ""]
    for smp in SAMPLES:
        g = gof[gof["date"].map(lambda t: sample_label(t, bcfg)) == smp]
        ls = logscore[logscore["date"].map(lambda t: sample_label(t, bcfg)) == smp]
        piv = ls.pivot_table(index="session", columns="model", values="logscore").dropna()
        m = mcs(-piv, rcfg["mcs_size"], rcfg["mcs_reps"], seed=7) if piv.shape[1] > 1 else None
        t = tests[(tests["sample"] == smp) & (tests["model"] != "-")]
        rt = ranking_tabs[smp]
        rt = rt[(rt["model"] != "-") & (rt["profile"] == rt["eval_profile"])]
        rt = rt.assign(r=rt.groupby(["strategy", "profile"])["ce_ann"].rank(ascending=False))
        for model in sorted(set(g["model"]) | set(t["model"])):
            gm = g[g["model"] == model]
            tm = t[t["model"] == model]
            passed = {k: float((tm[c] >= level).mean()) if len(tm) else np.nan
                      for k, c in (("kupiec", "p_uc"), ("christoffersen", "p_cc"), ("z2", "p_z2"), ("de", "p_de_uc"))}
            rows.append({"sample": smp, "model": model, "aic": gm["aic"].mean(), "bic": gm["bic"].mean(),
                         "vuong_rvine_better": int((gm.get("vuong_winner") == "rvine").sum()),
                         "vuong_rvine_worse": int((gm.get("vuong_winner") == model).sum()) if model != "rvine" else 0,
                         "ls_mean": float(piv[model].mean()) if model in piv.columns else np.nan,
                         "in_mcs_ls": bool(m.loc[model, "in_mcs"]) if m is not None and model in m.index else None,
                         "mcs_p_ls": float(m.loc[model, "mcs_pvalue"]) if m is not None and model in m.index else np.nan,
                         **{f"pass_{k}": v for k, v in passed.items()},
                         "avg_rank_ce": float(rt.loc[rt["model"] == model, "r"].mean())})
        d = pd.DataFrame([r for r in rows if r["sample"] == smp]).sort_values("avg_rank_ce")
        lines += [f"## Próba {LABEL[smp]}", "",
                  "| model | śr. AIC | śr. BIC | Vuong: R-vine lepszy | Vuong: R-vine gorszy | śr. log-score/dzień | "
                  "MCS log-score (p) | Kupiec | Christoffersen | Z₂ | DE | śr. ranga CE |",
                  "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
        for r in d.itertuples():
            mc = "" if r.in_mcs_ls is None else ("tak" if r.in_mcs_ls else "nie")
            lines.append(f"| {r.model} | {num(r.aic, 0)} | {num(r.bic, 0)} | {r.vuong_rvine_better} | "
                         f"{r.vuong_rvine_worse} | {num(r.ls_mean, 2)} | {mc} ({num(r.mcs_p_ls, 3)}) | "
                         f"{pct(r.pass_kupiec, 0)}% | {pct(r.pass_christoffersen, 0)}% | {pct(r.pass_z2, 0)}% | "
                         f"{pct(r.pass_de, 0)}% | {num(r.avg_rank_ce, 2)} |")
        lines.append("")
    return pd.DataFrame(rows), "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- podsumowanie i wrażliwość

def sensitivity_section(periods, gammas, profiles) -> list[str]:
    cal = periods[(periods["sample"] == "calibration") & (periods["model"] != "-")]
    if cal["variant"].nunique() <= 1:
        return []
    lines = ["## Wrażliwość (próba kalibracyjna, strategia ce_dep)", "",
             "CE zrealizowany [% rocznie] przy γ profilu; wartości średnie po 7 modelach zależności "
             "(kolumna base – parametry wyjściowe).", ""]
    rows = []
    for (sid, var), g in cal.groupby(["sid", "variant"]):
        st, p = g["strategy"].iloc[0], g["profile"].iloc[0]
        if st != "ce_dep" or p not in gammas:
            continue
        ce = metrics.annualize_ce(metrics.realized_ce(g["r_net"], g["rf"], gammas[p]))
        rows.append({"model": g["model"].iloc[0], "profile": p, "variant": var, "ce": ce})
    d = pd.DataFrame(rows)
    piv = d.pivot_table(index="profile", columns="variant", values="ce", aggfunc="mean")
    cols = ["base"] + [c for c in ("lag0", "q10", "rhoS") if c in piv.columns]
    lines += ["| profil | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    lines += [f"| {p} | " + " | ".join(pct(piv.loc[p, c]) for c in cols) + " |" for p in piv.index]
    kcols = sorted([c for c in piv.columns if c.startswith("kappa=")], key=lambda c: float(c.split("=")[1]))
    if kcols:
        kp = piv[kcols]
        kp.columns = [float(c.split("=")[1]) for c in kcols]
        order = [p for p in profiles if p in kp.index]
        sel = monotone_kappa(kp.loc[order], order)
        lines += ["", "Siatka κ (CE [% rocznie], średnia po modelach) i wybór κ maksymalizujący Σ CE przy "
                  "ograniczeniu κ_P1 ≥ … ≥ κ_P5 (informacyjnie – hold-out liczony z κ wyjściowymi):", "",
                  "| profil | " + " | ".join(f"κ = {k:g}" for k in kp.columns) + " | κ wyjściowe | κ wybrane |",
                  "|---|" + "---:|" * (len(kp.columns) + 2)]
        for p in order:
            lines.append(f"| {p} | " + " | ".join(pct(kp.loc[p, k]) for k in kp.columns)
                         + f" | {profiles[p].kappa:g} | {sel[p]:g} |")
    return lines + [""]


def summary_markdown(run_dir, periods, tests, flags, scen, timings, rank_tabs, profiles, gammas, bcfg) -> str:
    n = {s: periods.loc[(periods["sample"] == s) & (periods["variant"] == "base"), "date"].nunique() for s in SAMPLES}
    base = periods[periods["variant"] == "base"]
    lines = ["# Backtest – podsumowanie", "",
             f"Przebieg `{run_dir.name}`; daty rebalansowania {timings['date'].min()} – {timings['date'].max()} "
             f"({len(timings)}), w tym okresy realizacji: kalibracja {n['calibration']} mies., hold-out "
             f"{n['holdout']} mies. Opóźnienie wykonania {bcfg['execution_lag']} sesja, koszt "
             f"{100 * bcfg['transaction_cost']:.1f}% wartości transakcji, r_f – WIBOR 1M.",
             f"Szeregi bazowe: {base['sid'].nunique()} (7 modeli × 4 strategie × 5 profili + ew + 5 × mv_hist).",
             f"Scenariusze: mediana N = {int(scen['n'].median())}, maks. {int(scen['n'].max())}; odsetek par "
             f"(data, model) z N = n_max: {pct((scen.drop_duplicates(['date', 'model'])['n'] >= bcfg['scenarios']['n_max']).mean(), 1)}%.",
             f"Czas: średnio {timings['seconds'].mean():.0f} s/datę (1 proces), łącznie {timings['seconds'].sum() / 3600:.1f} h CPU.",
             f"Luzowanie c_LT: {pct(flags['lt_relaxed'].mean(), 2)}% problemów poziomu 2; W_max: "
             f"{pct(flags['wmax_relaxed'].mean(), 2)}%; maxret_es bez rozwiązania przy ES ≤ ES(EW) (min ES): "
             f"{pct(flags['infeasible_es'].mean(), 2)}%.", ""]
    for smp in SAMPLES:
        tab = rank_tabs[smp]
        lines += [f"## Najlepsze szeregi wg CE – próba {LABEL[smp]}", "",
                  "| profil | 1. | 2. | 3. | ew (CE) | rvine/ce_dep (CE, miejsce) | w MCS |", "|---|---|---|---|---:|---|---:|"]
        for p in profiles:
            t = tab[tab["eval_profile"] == p].reset_index(drop=True)
            if t.empty:
                continue
            top = [f"{short(s)} ({pct(c)})" for s, c in zip(t["sid"][:3], t["ce_ann"][:3])]
            ew = t[t["strategy"] == "ew"]
            main = t[(t["model"] == "rvine") & (t["strategy"] == "ce_dep")]
            lines.append(f"| {p} | " + " | ".join(top) + f" | {pct(ew['ce_ann'].iloc[0]) if len(ew) else ''} | "
                         + (f"{pct(main['ce_ann'].iloc[0])} ({main.index[0] + 1}.)" if len(main) else "")
                         + f" | {int(t['in_mcs'].sum())} |")
        lines.append("")
    tb = tests[tests["model"] != "-"]
    if not tb.empty:
        lines += ["## Testy VaR/ES (szeregi modelowe, odsetek bez odrzucenia na poziomie 5%)", "",
                  "| próba | T (mies.) | oczekiwane przekr. | śr. przekr. | Kupiec | Christoffersen cc | Z₂ | DE uc | DE cc(5) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for smp in SAMPLES:
            t = tb[tb["sample"] == smp]
            if t.empty:
                continue
            lines.append(f"| {LABEL[smp]} | {int(t['T'].median())} | {t['expected'].median():.1f} | "
                         f"{t['exceedances'].mean():.1f} | {pct((t['p_uc'] >= .05).mean(), 0)}% | "
                         f"{pct((t['p_cc'] >= .05).mean(), 0)}% | {pct((t['p_z2'] >= .05).mean(), 0)}% | "
                         f"{pct((t['p_de_uc'] >= .05).mean(), 0)}% | {pct((t['p_de_cc5'] >= .05).mean(), 0)}% |")
        lines += ["", "Przy próbie miesięcznej oczekiwana liczba przekroczeń jest mała, więc moc testów jest niska.", ""]
    lines += sensitivity_section(periods, gammas, profiles)
    lines += ["Szczegóły: `ranking_calibration.md`, `ranking_holdout.md`, `dependence_models.md`, `es_mc.md`; "
              "wykresy: `figures/backtest/`.", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- wykresy

def figures(periods, weights_W, forecasts, profiles, rcfg, bcfg) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    main_m, main_s = rcfg["main_series"]["model"], rcfg["main_series"]["strategy"]
    base = periods[periods["variant"] == "base"]
    split = pd.Timestamp(bcfg["holdout"][0])
    files = []
    # krzywe majątku i obsunięcia
    fig, axes = plt.subplots(len(profiles), 2, figsize=(14, 3.2 * len(profiles)), constrained_layout=True)
    for i, p in enumerate(profiles):
        sel = base[(base["profile"] == p) & (base["model"].isin([main_m, "-"]))]
        sel = pd.concat([sel, base[base["sid"] == rcfg["sharpe_test"]["benchmark"]]])
        for sid, g in sel.groupby("sid"):
            g = g.sort_values("end")
            w = np.cumprod(1.0 + g["r_net"].to_numpy())
            axes[i, 0].plot(g["end"], w, lw=1.1, label=short(sid))
            axes[i, 1].plot(g["end"], 100 * metrics.drawdown_series(g["r_net"].to_numpy()), lw=1.0)
        for ax in axes[i]:
            ax.axvline(split, color="grey", ls="--", lw=0.8)
            ax.grid(alpha=0.3)
        axes[i, 0].set_ylabel(f"{p}: majątek")
        axes[i, 1].set_ylabel("obsunięcie [%]")
        axes[i, 0].legend(fontsize=7, ncol=3)
    fig.suptitle(f"Krzywe majątku (netto, PLN) i obsunięcia – {main_m} × strategie, benchmarki; linia: początek hold-out")
    files.append(_save(fig, "wealth_drawdown.png"))
    # skład W w czasie (strategia główna)
    fig, axes = plt.subplots(len(profiles), 1, figsize=(12, 2.6 * len(profiles)), constrained_layout=True, sharex=True)
    Wm = weights_W[(weights_W["model"] == main_m) & (weights_W["strategy"] == main_s) & (weights_W["variant"] == "base")]
    for i, p in enumerate(profiles):
        piv = Wm[Wm["profile"] == p].pivot_table(index="date", columns="subfund", values="weight").fillna(0.0)
        axes[i].stackplot(piv.index, piv.T.to_numpy(), labels=piv.columns, step="post")
        axes[i].set_ylabel(p)
        axes[i].set_ylim(0, 1)
        axes[i].axvline(split, color="k", ls="--", lw=0.8)
    axes[0].legend(ncol=9, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.35))
    fig.suptitle(f"Wagi sub-funduszy W w czasie – {main_m} × {main_s}")
    files.append(_save(fig, "composition_W.png"))
    # krocząca 12-miesięczna Sharpe'a
    fig, axes = plt.subplots(len(profiles), 1, figsize=(12, 2.4 * len(profiles)), constrained_layout=True, sharex=True)
    for i, p in enumerate(profiles):
        for sid in (f"{main_m}|{main_s}|{p}|base", f"-|mv_hist|{p}|base", rcfg["sharpe_test"]["benchmark"]):
            g = base[base["sid"] == sid].sort_values("end")
            if g.empty:
                continue
            x = (g["r_net"] - g["rf"]).rolling(12)
            axes[i].plot(g["end"], x.mean() / x.std() * np.sqrt(12), lw=1.0, label=short(sid))
        axes[i].axhline(0, color="k", lw=0.5)
        axes[i].axvline(split, color="grey", ls="--", lw=0.8)
        axes[i].set_ylabel(p)
        axes[i].legend(fontsize=7)
    fig.suptitle("Krocząca 12-miesięczna Sharpe'a (nadwyżka nad WIBOR 1M)")
    files.append(_save(fig, "rolling_sharpe.png"))
    # ES ex ante vs zrealizowane straty
    fig, axes = plt.subplots(len(profiles), 1, figsize=(12, 2.4 * len(profiles)), constrained_layout=True, sharex=True)
    fc = forecasts[(forecasts["model"] == main_m) & (forecasts["strategy"] == main_s) & (forecasts["variant"] == "base")]
    for i, p in enumerate(profiles):
        g = fc[fc["profile"] == p].sort_values("date")
        axes[i].plot(g["date"], 100 * g["es"], lw=1.0, label="ES 97,5% ex ante")
        axes[i].plot(g["date"], 100 * g["var"], lw=0.8, ls=":", label="VaR 97,5% ex ante")
        axes[i].bar(g["date"], 100 * g["loss"], width=20, color=np.where(g["hit"] == True, "C3", "0.6"),
                    label="strata zrealizowana")
        axes[i].axvline(split, color="grey", ls="--", lw=0.8)
        axes[i].set_ylabel(f"{p} [%]")
    axes[0].legend(fontsize=7, ncol=3)
    fig.suptitle(f"Prognozy ex ante a zrealizowane miesięczne straty – {main_m} × {main_s} (czerwone: przekroczenie VaR)")
    files.append(_save(fig, "es_vs_losses.png"))
    return files


def _save(fig, name) -> str:
    path = FIG_DIR / name
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, bcfg, pcfg, subfunds = load_all()
    run_dir = BACKTEST_DIR / current_run_id(cfg, bcfg, pcfg, subfunds)
    rcfg = bcfg["ranking"]
    alpha = float(cfg["rebalance"]["es_alpha"])
    profiles = {k: v for k, v in load_profiles(pcfg).items() if k in bcfg["grid"]["profiles"]}
    gammas = {k: v.gamma for k, v in profiles.items()}
    periods = pd.read_parquet(run_dir / "periods.parquet")
    tests = pd.read_parquet(run_dir / "tests.parquet")
    es_expost = None
    if (run_dir / "es_mc.parquet").exists():
        e = pd.read_parquet(run_dir / "es_mc.parquet")
        e = e[(e["kind"] == "portfolio") & (e["method"] == rcfg["es_expost_method"])]
        es_expost = e.set_index(["sid", "sample"])["es"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tabs = {}
    for smp in SAMPLES:
        tabs[smp] = rank_sample(periods, tests, gammas, rcfg, alpha, es_expost, smp)
        write_parquet(tabs[smp], run_dir / f"ranking_{smp}.parquet")
        n_m = periods.loc[(periods["sample"] == smp) & (periods["variant"] == "base"), "date"].nunique()
        (REPORT_DIR / f"ranking_{smp}.md").write_text(ranking_markdown(tabs[smp], smp, profiles, rcfg, n_m),
                                                     encoding="utf-8")
        log.info("Ranking %s: %d wierszy", smp, len(tabs[smp]))
    dep, text = dependence_ranking(pd.read_parquet(run_dir / "gof.parquet"), pd.read_parquet(run_dir / "logscore.parquet"),
                                   tests, tabs, rcfg, bcfg)
    write_parquet(dep, run_dir / "dependence_ranking.parquet")
    (REPORT_DIR / "dependence_models.md").write_text(text, encoding="utf-8")
    timings = pd.read_parquet(run_dir / "timings.parquet")
    (REPORT_DIR / "backtest_summary.md").write_text(summary_markdown(
        run_dir, periods, tests, pd.read_parquet(run_dir / "flags.parquet"),
        pd.read_parquet(run_dir / "scenario_stats.parquet"), timings, tabs, profiles, gammas, bcfg), encoding="utf-8")
    files = figures(periods, pd.read_parquet(run_dir / "weights_W.parquet"), pd.read_parquet(run_dir / "forecasts.parquet"),
                    profiles, rcfg, bcfg)
    update_manifest(f"backtest/{run_dir.name}/ranking", run_dir / "ranking_holdout.parquet",
                    params={"ranking": rcfg, "figures": files})
    log.info("Raporty: %s; wykresy: %s", REPORT_DIR, json.dumps(files))


if __name__ == "__main__":
    main()
