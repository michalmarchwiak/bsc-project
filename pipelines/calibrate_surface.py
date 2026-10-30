"""Kalibracja parametrycznej powierzchni zmienności i walidacja poza próbą.

Wejście (artefakty Etapu 1):
    artifacts/market/vol_indices.parquet, artifacts/market/underlyings.parquet,
    artifacts/market/dolthub_options/SPY.parquet (jeśli dostępne),
    artifacts/raw/options_snapshots/date=*/SPY.parquet (jeśli dostępne).

Kroki: kwotowania → filtry i IV (OTM) → podział czasowy (``surface.oos_split``) →
kalibracja na części uczącej → RMSE OOS w % premii (puty ATM, 5% OTM, 10% OTM) →
kalibracja końcowa na całej próbie.

Wyjście:
    artifacts/surface/skew_params.json         – parametry końcowe (wersja = hash parametrów)
    artifacts/reports/surface_oos.csv / .md    – raport walidacji OOS
    wpis ``surface/skew_params`` w manifest.json

Uruchomienie:
    python -m pipelines.calibrate_surface
"""

from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

from pf import ARTIFACTS_DIR, config
from pf.artifacts import update_manifest
from pf.data.options_snapshot import OUT_DIR as SNAPSHOT_DIR
from pf.options import surface

log = logging.getLogger("calibrate_surface")

MARKET = ARTIFACTS_DIR / "market"
PARAMS_PATH = ARTIFACTS_DIR / "surface" / "skew_params.json"
REPORT_DIR = ARTIFACTS_DIR / "reports"
QUOTE_COLUMNS = ["quote_date", "expiration", "strike", "call_put", "bid", "ask", "underlying_price"]


def load_dolthub(symbol: str, spot: pd.Series) -> pd.DataFrame | None:
    path = MARKET / "dolthub_options" / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path).rename(columns={"date": "quote_date"})
    df["underlying_price"] = spot.reindex(df["quote_date"]).to_numpy()
    return df.dropna(subset=["underlying_price"]).assign(source="dolthub")[QUOTE_COLUMNS + ["source"]]


def load_snapshots(symbol: str) -> pd.DataFrame | None:
    files = sorted(SNAPSHOT_DIR.glob(f"date=*/{symbol.replace('^', '')}.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["quote_date"] = (df["fetched_at_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None)
                        .dt.normalize())
    df = df.sort_values("fetched_at_utc").drop_duplicates(["quote_date", "contractSymbol"], keep="last")
    return df.assign(source="yfinance")[QUOTE_COLUMNS + ["openInterest", "lastTradeDate", "source"]]


def load_quotes(symbol: str = "SPY") -> pd.DataFrame:
    und = pd.read_parquet(MARKET / "underlyings.parquet")
    frames = [x for x in (load_dolthub(symbol, und[(symbol, "Close")]), load_snapshots(symbol)) if x is not None]
    if not frames:
        raise SystemExit("Brak kwotowań opcji (DoltHub i zrzuty yfinance) – uruchom najpierw Etap 1")
    quotes = pd.concat(frames, ignore_index=True)
    return quotes.drop_duplicates(["quote_date", "expiration", "strike", "call_put"], keep="first")


def to_markdown(report: pd.DataFrame, params: dict, train_info: dict, split: str) -> str:
    lines = [
        "# Walidacja powierzchni zmienności poza próbą",
        "",
        f"Model: σ_imp = a + b·VIX_T + c·k + d·k·VIX_T{' + e·k·SKEW/100' if params['use_skew'] else ''}, "
        f"k = ln(K/F). Podział czasowy: kalibracja < {split} ≤ walidacja.",
        f"Kalibracja ({train_info['method']}): n = {train_info['n_obs']}, "
        f"RMSE IV w próbie = {100 * train_info['rmse_iv']:.2f} p.p.",
        "",
        "Błąd ceny = (model − mid) / mid, puty; przedziały K/S z `config/options.yaml`.",
        "",
        "| bucket | tenor | n | RMSE ceny [% premii] | obciążenie [% premii] | RMSE IV [p.p.] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in report.itertuples():
        lines.append(f"| {r.bucket} | {r.tenor} | {r.n} | {r.rmse_price_pct:.2f} | {r.bias_price_pct:.2f} "
                     f"| {r.rmse_iv_pp:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config.load("options")
    scfg, filters = cfg["surface"], cfg["iv_filters"]

    vol = pd.read_parquet(MARKET / "vol_indices.parquet")
    und = pd.read_parquet(MARKET / "underlyings.parquet")
    irx = und[("IRX", "Close")] if ("IRX", "Close") in und.columns else None
    quotes = load_quotes("SPY")
    obs = surface.prepare_quotes(quotes, vol, irx, scfg, filters)
    accepted = obs[obs["reject"] == ""]
    log.info("Kwotowania OTM: %d, przyjęte: %d; odrzucenia: %s", len(obs), len(accepted),
             obs.loc[obs["reject"] != "", "reject"].value_counts().to_dict())

    train, test = surface.time_split(obs, scfg["oos_split"])
    kw = dict(use_skew=scfg["use_skew"], robust=scfg["robust"], huber_c=scfg["huber_c"],
              min_sigma=scfg["min_sigma"])
    if len(train[train["reject"] == ""]) and len(test[test["reject"] == ""]):
        p_train = surface.fit_skew(train, **kw)
        report = surface.oos_report(test, p_train, scfg["buckets"])
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report.to_csv(REPORT_DIR / "surface_oos.csv", index=False)
        (REPORT_DIR / "surface_oos.md").write_text(
            to_markdown(report, p_train.to_dict(), p_train.fit_info, scfg["oos_split"]), encoding="utf-8")
        oos = {"split": scfg["oos_split"], "train_params": p_train.to_dict(),
               "report": report.to_dict(orient="records")}
    else:
        log.warning("Za mało danych po jednej ze stron podziału %s – walidacja OOS pominięta", scfg["oos_split"])
        oos = {"status": "skipped", "split": scfg["oos_split"]}

    params = surface.fit_skew(obs, **kw).to_dict()
    params["tenors_days"] = scfg["tenors_days"]
    params["sources"] = quotes["source"].value_counts().to_dict()
    version = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]
    params["version"] = version
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(params, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    update_manifest("surface/skew_params", PARAMS_PATH, params={"version": version, "oos": oos})
    log.info("Powierzchnia %s: %s", version, {k: params[k] for k in "abcde"})


if __name__ == "__main__":
    main()
