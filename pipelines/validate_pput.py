"""Walidacja wyceny syntetycznej: replikacja Cboe PPUT vs oficjalny szereg.

Wejście: artifacts/market/{underlyings,vol_indices,cboe_strategy_indices}.parquet,
         artifacts/surface/skew_params.json.
Wyjście: artifacts/reports/pput_tracking.csv / .md, wpis ``reports/pput_tracking`` w manifeście.
Gdy brak szeregu PPUT, walidacja jest oznaczana w manifeście jako pominięta.

Uwaga: parametry skośności skalibrowano na danych od 2019 r., więc dla wcześniejszych lat
(m.in. 2008) replikacja jest ekstrapolacją parametrów – raport służy walidacji wyceny,
nie jest backtestem (w backteście obowiązuje kalibracja wyłącznie na danych ≤ t).

Uruchomienie:
    python -m pipelines.validate_pput
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from pf import ARTIFACTS_DIR, config
from pf.artifacts import update_manifest
from pf.options import pput
from pf.options.surface import SkewParams, rate_from_irx

log = logging.getLogger("validate_pput")

MARKET = ARTIFACTS_DIR / "market"
REPORT_DIR = ARTIFACTS_DIR / "reports"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    strat_path = MARKET / "cboe_strategy_indices.parquet"
    if not strat_path.exists():
        update_manifest("reports/pput_tracking", params={"status": "skipped", "reason": "brak szeregu PPUT"})
        log.warning("Brak szeregu PPUT – walidacja pominięta")
        return
    cfg = config.load("options")["surface"]
    raw = json.loads((ARTIFACTS_DIR / "surface" / "skew_params.json").read_text(encoding="utf-8"))
    params = SkewParams(**{k: raw[k] for k in ("a", "b", "c", "d", "e", "use_skew", "min_sigma")})

    und = pd.read_parquet(MARKET / "underlyings.parquet")
    vol = pd.read_parquet(MARKET / "vol_indices.parquet")
    official = pd.read_parquet(strat_path)["PPUT"].dropna()
    spx, spx_tr = und[("SPX", "Close")].dropna(), und[("SPXTR", "Close")].dropna()
    rate = pd.Series(rate_from_irx(und[("IRX", "Close")]), index=und.index)
    q = pput.trailing_div_yield(spx, spx_tr).reindex(spx.index).fillna(cfg["default_div_yield"])

    replica = pput.replicate(spx, spx_tr, vol, params, rate, q, tenors_days=cfg["tenors_days"])
    start = max(replica.index[0], official.index[0])
    rep = replica.loc[start:] / replica.loc[start:].iloc[0]
    off = official.loc[start:] / official.loc[start:].iloc[0]
    report = pput.tracking_report(rep, off)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "pput_tracking.csv"
    report.to_csv(path, index=False)
    md = ["# Replikacja Cboe PPUT (BS + powierzchnia) vs szereg oficjalny", "",
          f"Wersja powierzchni: {raw.get('version')}. TE = σ(różnic dziennych log-stóp) · √252.", "",
          "| okres | dni | TE [% p.a.] | zwrot replika [%] | zwrot PPUT [%] | korelacja |",
          "|---|---:|---:|---:|---:|---:|"]
    md += [f"| {r.period} | {r.n_days} | {r.te_ann_pct:.2f} | {r.ret_replica_pct:.2f} | "
           f"{r.ret_official_pct:.2f} | {r.corr:.4f} |" for r in report.itertuples()]
    (REPORT_DIR / "pput_tracking.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    update_manifest("reports/pput_tracking", path, n_rows=len(report),
                    params={"status": "ok", "surface_version": raw.get("version")})
    log.info("TE całość: %.2f%% p.a.", report.loc[report["period"] == "all", "te_ann_pct"].iloc[0])


if __name__ == "__main__":
    main()
