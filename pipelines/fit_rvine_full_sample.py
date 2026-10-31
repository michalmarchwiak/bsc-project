"""Estymacja marginesów AR(1)–GJR-GARCH(1,1) skew-t i modeli zależności na próbie kalibracyjnej (PLN).

Próba: ``calibration`` z ``config/backtest.yaml`` (przecięcie z zakresem danych). Marginesy estymowane są
raz; wszystkie modele zależności z ``dependence.models`` (``config/models.yaml``) – na tych samych
pseudo-obserwacjach panelu reszt. Porównanie: log-wiarygodność, AIC, BIC i test Vuonga (z poprawką
Schwarza) modelu ``dependence.reference`` (R-vine) względem pozostałych modeli z gęstością.

Wyjście:
    artifacts/models/rvine_full_sample.json.gz           – struktura i kopuły parowe R-vine
    artifacts/models/<model>_full_sample.json.gz         – pozostałe modele z gęstością (cvine, dvine, gauss, t)
    artifacts/models/margins_full_sample.csv             – parametry marginesów
    artifacts/reports/rvine_full_sample.csv / .md        – krawędzie drzew 1–L R-vine i podsumowanie
    artifacts/reports/dependence_models_full_sample.csv / .md – porównanie modeli zależności

Uruchomienie:
    python -m pipelines.fit_rvine_full_sample [--jobs 4] [--models rvine gauss]
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from pf import ARTIFACTS_DIR, ROOT, config
from pf.artifacts import update_manifest
from pf.backtest.data import eligible_assets, load_returns
from pf.models.dependence import fit_model, save_model
from pf.models.gof import compare_with_reference
from pf.models.margins import fit_margins, residual_panel
from pf.models.rvine import pseudo_observations
from pf.portfolio import load_subfunds

log = logging.getLogger("fit_rvine_full_sample")

MODEL_DIR = ARTIFACTS_DIR / "models"
REPORT_DIR = ARTIFACTS_DIR / "reports"


def to_markdown(edges: pd.DataFrame, vine, margins: pd.DataFrame, returns: pd.DataFrame, n_panel: int,
                sf_of: dict[str, str], estimation: str) -> str:
    est = "parametry z MNW (start z τ Kendalla)" if estimation == "mle" else "parametry z τ Kendalla"
    lines = [
        "# R-vine na próbie kalibracyjnej",
        "",
        f"Dane: dzienne log-stopy w PLN, {returns.index.min().date()} – {returns.index.max().date()}; "
        f"{vine.d} aktywów, {n_panel} wspólnych sesji panelu reszt.",
        f"Marginesy: AR(1)–GJR-GARCH(1,1) skew-t; kopuła: R-vine obcięta na {vine.truncation} drzewach, "
        f"{est}, rodzina z AIC.",
        f"Log-wiarygodność kopuły: {vine.loglik():.1f}, liczba parametrów: {vine.n_params()}, "
        f"AIC: {vine.aic():.1f}.",
        "",
        "## Rodziny kopuł w drzewach",
        "",
    ]
    lines += family_table(edges)
    t1 = edges[edges["tree"] == 1].copy()
    t1["sf_a"], t1["sf_b"] = t1["a"].map(sf_of), t1["b"].map(sf_of)
    within = (t1["sf_a"] == t1["sf_b"]).mean()
    lines += ["", f"Drzewo 1: {100 * within:.0f}% krawędzi łączy aktywa z tego samego sub-funduszu.", "",
              "## Drzewo 1 – krawędzie", "",
              "| a | b | rodzina | τ̂ | θ / ρ | ν | λ_L | λ_U |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in t1.sort_values("tau_emp", key=abs, ascending=False).itertuples():
        nu = f"{r.par2:.1f}" if pd.notna(r.par2) else ""
        lines.append(f"| {r.a} | {r.b} | {r.family} | {r.tau_emp:.3f} | {r.par:.3f} | {nu} | "
                     f"{r.lambda_L:.3f} | {r.lambda_U:.3f} |")
    lines += ["", "## Marginesy (parametry, stopy ×100)", "",
              "| ticker | μ | φ | ω | α | γ | β | η | λ | zbieżność |", "|---|" + "---:|" * 9]
    for r in margins.itertuples():
        lines.append(f"| {r.name} | {r.mu:.3f} | {r.phi:.3f} | {r.omega:.4f} | {r.alpha:.3f} | {r.gamma:.3f} | "
                     f"{r.beta:.3f} | {r.eta:.2f} | {r.lam:.3f} | {'tak' if r.converged else 'NIE'} |")
    return "\n".join(lines) + "\n"


def family_table(edges: pd.DataFrame) -> list[str]:
    counts = edges.pivot_table(index="family", columns="tree", values="a", aggfunc="count", fill_value=0)
    counts["razem"] = counts.sum(axis=1)
    lines = ["| rodzina | " + " | ".join(f"T{c}" for c in counts.columns[:-1]) + " | razem |",
             "|---|" + "---:|" * len(counts.columns)]
    lines += [f"| {f} | " + " | ".join(str(int(v)) for v in row) + " |" for f, row in counts.iterrows()]
    return lines


def models_markdown(table: pd.DataFrame, models: dict, returns: pd.DataFrame, n_panel: int, reference: str,
                    level: float) -> str:
    lines = [
        "# Modele zależności na próbie kalibracyjnej",
        "",
        f"Dane: dzienne log-stopy w PLN, {returns.index.min().date()} – {returns.index.max().date()}; "
        f"{next(iter(models.values())).d} aktywów, {n_panel} sesji panelu reszt (wspólne marginesy "
        "AR(1)–GJR-GARCH(1,1) skew-t, pseudo-obserwacje z rang).",
        f"AIC = 2k − 2ℓ, BIC = k ln n − 2ℓ. Test Vuonga z poprawką Schwarza: Z > 0 – lepszy {reference}; "
        f"rozstrzygnięcie przy |Z| > z_(1−{level}/2). Kopuła empiryczna nie ma gęstości (bez AIC/BIC i testu).",
        "",
        "| model | ℓ | k | AIC | BIC | Z Vuonga | p | lepszy | czas estymacji [s] |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in table.sort_values("aic", na_position="last").itertuples():
        z = f"{r.vuong_stat:.2f}" if pd.notna(getattr(r, "vuong_stat", np.nan)) else ""
        p = f"{r.vuong_pvalue:.3g}" if pd.notna(getattr(r, "vuong_pvalue", np.nan)) else ""
        win = r.vuong_winner if isinstance(getattr(r, "vuong_winner", None), str) else ""
        ll = f"{r.loglik:.1f}" if pd.notna(r.loglik) else ""
        aic = f"{r.aic:.1f}" if pd.notna(r.aic) else ""
        bic = f"{r.bic:.1f}" if pd.notna(r.bic) else ""
        lines.append(f"| {r.model} | {ll} | {r.n_params} | {aic} | {bic} | {z} | {p} | {win} | "
                     f"{r.fit_seconds:.1f} |")
    vines = {n: m for n, m in models.items() if hasattr(m, "vine")}
    if vines:
        lines += ["", "## Rodziny kopuł parowych (drzewa 1–L)", ""]
        for n, m in vines.items():
            edges = pd.DataFrame(m.vine.summary())
            lines += [f"### {n}", ""] + family_table(edges) + [""]
        if "cvine" in vines:
            v = vines["cvine"].vine
            roots = []
            for level_ in v.trees[:v.truncation]:
                common = set.intersection(*[{e.a, e.b} for e in level_]) if level_ else set()
                roots.append(", ".join(v.names[i] for i in sorted(common)))
            lines += ["C-vine – korzenie drzew 1–L: " + " → ".join(roots), ""]
        if "dvine" in vines:
            v = vines["dvine"].vine
            adj: dict[int, list[int]] = {}
            for e in v.trees[0]:
                adj.setdefault(e.a, []).append(e.b)
                adj.setdefault(e.b, []).append(e.a)
            start = [k for k, nb in adj.items() if len(nb) == 1][0]
            path, prev = [start], None
            while len(path) < v.d:
                nxt = [k for k in adj[path[-1]] if k != prev][0]
                prev = path[-1]
                path.append(nxt)
            lines += ["D-vine – porządek zmiennych: " + " – ".join(v.names[i] for i in path), ""]
    if "student_t" in models:
        lines += [f"Kopuła t: ν̂ = {models['student_t'].nu:.2f} (profil wiarygodności na siatce).", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--models", nargs="*", default=None, help="podzbiór modeli zależności")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config.load("models")
    bcfg = config.load("backtest")
    dcfg = cfg["dependence"]
    names_models = args.models or dcfg["models"]
    returns = load_returns(ROOT / cfg["returns"]["path"])
    lo, hi = (pd.Timestamp(x) for x in bcfg["calibration"])
    returns = returns.loc[(returns.index >= lo) & (returns.index <= hi)]
    assets = eligible_assets(returns, cfg["rebalance"]["min_obs"])

    t0 = time.time()
    margins, z = fit_margins(returns[assets], scale=cfg["margins"]["scale"], jobs=args.jobs)
    panel = residual_panel(z)
    U = pseudo_observations(panel.to_numpy())
    log.info("Marginesy: %.0f s, %d aktywów, panel %d sesji", time.time() - t0, len(assets), len(panel))

    models, times = {}, {}
    for name in names_models:
        t1 = time.time()
        models[name] = fit_model(name, U, assets, cfg)
        times[name] = time.time() - t1
        log.info("%s: ℓ = %.1f, k = %d, AIC = %.1f (%.0f s)", name, models[name].loglik, models[name].n_params,
                 models[name].aic(), times[name])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    margins_df = pd.DataFrame([f.to_dict() for f in margins.values()])
    margins_df.to_csv(MODEL_DIR / "margins_full_sample.csv", index=False)
    for name, m in models.items():
        if name == "rvine":
            m.vine.save(MODEL_DIR / "rvine_full_sample.json.gz")
        elif m.has_density and name != "indep":
            save_model(m, MODEL_DIR / f"{name}_full_sample.json.gz")

    table = compare_with_reference(models, U, dcfg["reference"], dcfg["vuong_level"])
    table["fit_seconds"] = table["model"].map(times)
    table.to_csv(REPORT_DIR / "dependence_models_full_sample.csv", index=False)
    (REPORT_DIR / "dependence_models_full_sample.md").write_text(
        models_markdown(table, models, returns, len(panel), dcfg["reference"], dcfg["vuong_level"]),
        encoding="utf-8")
    common = {"returns": cfg["returns"]["path"], "start": str(returns.index.min().date()),
              "end": str(returns.index.max().date()), "calibration": bcfg["calibration"],
              "rvine": cfg["rvine"], "margins": cfg["margins"], "panel_obs": int(len(panel))}
    if "rvine" in models:
        vine = models["rvine"].vine
        edges = pd.DataFrame(vine.summary())
        edges.to_csv(REPORT_DIR / "rvine_full_sample.csv", index=False)
        sf_of = {t: sf for sf, spec in load_subfunds().items() for t in spec["tickers"]}
        (REPORT_DIR / "rvine_full_sample.md").write_text(
            to_markdown(edges, vine, margins_df, returns, len(panel), sf_of, cfg["rvine"].get("estimation", "itau")),
            encoding="utf-8")
        update_manifest("models/rvine_full_sample", MODEL_DIR / "rvine_full_sample.json.gz", params={
            **common, "loglik": vine.loglik(), "aic": vine.aic()})
        log.info("AIC R-vine %.1f; rodziny: %s", vine.aic(), edges["family"].value_counts().to_dict())
    update_manifest("reports/dependence_models_full_sample", REPORT_DIR / "dependence_models_full_sample.csv",
                    n_rows=len(table), params={**common, "dependence": dcfg,
                                               "aic": table.set_index("model")["aic"].to_dict()})


if __name__ == "__main__":
    main()
