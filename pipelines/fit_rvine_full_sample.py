"""Estymacja marginesów AR(1)–GJR-GARCH(1,1) skew-t i kopuły R-vine na całej historii log-stóp (PLN).

Wyjście:
    artifacts/models/rvine_full_sample.json.gz     – struktura i kopuły parowe
    artifacts/models/margins_full_sample.csv       – parametry marginesów
    artifacts/reports/rvine_full_sample.csv / .md  – krawędzie drzew 1–L i podsumowanie

Uruchomienie:
    python -m pipelines.fit_rvine_full_sample [--jobs 4]
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from pf import ARTIFACTS_DIR, ROOT, config
from pf.artifacts import update_manifest
from pf.portfolio import load_subfunds
from pf.sim.scenarios import estimate

log = logging.getLogger("fit_rvine_full_sample")

MODEL_DIR = ARTIFACTS_DIR / "models"
REPORT_DIR = ARTIFACTS_DIR / "reports"


def to_markdown(edges: pd.DataFrame, vine, margins: pd.DataFrame, returns: pd.DataFrame, n_panel: int,
                sf_of: dict[str, str]) -> str:
    lines = [
        "# R-vine na pełnej próbie",
        "",
        f"Dane: dzienne log-stopy w PLN, {returns.index.min().date()} – {returns.index.max().date()}; "
        f"{vine.d} aktywów, {n_panel} wspólnych sesji panelu reszt.",
        f"Marginesy: AR(1)–GJR-GARCH(1,1) skew-t; kopuła: R-vine obcięta na {vine.truncation} drzewach, "
        f"parametry z τ Kendalla, rodzina z AIC.",
        f"Log-wiarygodność kopuły: {vine.loglik():.1f}, liczba parametrów: {vine.n_params()}, "
        f"AIC: {vine.aic():.1f}.",
        "",
        "## Rodziny kopuł w drzewach",
        "",
    ]
    counts = edges.pivot_table(index="family", columns="tree", values="a", aggfunc="count", fill_value=0)
    counts["razem"] = counts.sum(axis=1)
    lines += ["| rodzina | " + " | ".join(f"T{c}" for c in counts.columns[:-1]) + " | razem |",
              "|---|" + "---:|" * len(counts.columns)]
    lines += [f"| {f} | " + " | ".join(str(int(v)) for v in row) + " |" for f, row in counts.iterrows()]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config.load("models")
    returns = pd.read_csv(ROOT / cfg["returns"]["path"], index_col=0, parse_dates=True).sort_index()

    t0 = time.time()
    model = estimate(returns, cfg, min_obs=cfg["rebalance"]["min_obs"], jobs=args.jobs)
    log.info("Estymacja: %.0f s, %d aktywów, panel %d sesji", time.time() - t0, len(model.assets),
             model.panel_obs)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    vine_path = model.vine.save(MODEL_DIR / "rvine_full_sample.json.gz")
    margins = pd.DataFrame([f.to_dict() for f in model.margins.values()])
    margins.to_csv(MODEL_DIR / "margins_full_sample.csv", index=False)
    edges = pd.DataFrame(model.vine.summary())
    edges.to_csv(REPORT_DIR / "rvine_full_sample.csv", index=False)
    sf_of = {t: sf for sf, spec in load_subfunds().items() for t in spec["tickers"]}
    (REPORT_DIR / "rvine_full_sample.md").write_text(
        to_markdown(edges, model.vine, margins, returns, model.panel_obs, sf_of), encoding="utf-8")
    update_manifest("models/rvine_full_sample", vine_path, params={
        "returns": cfg["returns"]["path"], "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()), "rvine": cfg["rvine"], "margins": cfg["margins"],
        "loglik": model.vine.loglik(), "aic": model.vine.aic(), "panel_obs": model.panel_obs})
    log.info("AIC kopuły %.1f; rodziny: %s", model.vine.aic(), edges["family"].value_counts().to_dict())


if __name__ == "__main__":
    main()
