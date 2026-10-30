"""Import historycznych notowań opcji US z DoltHub ``post-no-preference/options`` (dane od 02.2019).

Dostęp:
    * SQL API DoltHub (domyślnie): ``GET https://www.dolthub.com/api/v1alpha1/<owner>/<repo>/<branch>?q=<SQL>``,
      odpowiedź JSON z polami ``query_execution_status``, ``schema``, ``rows``; wynik pojedynczego
      zapytania jest ograniczony, więc dane pobierane są dzień po dniu ze stronicowaniem LIMIT/OFFSET;
    * lokalny klon (``dolt clone post-no-preference/options``) – opcja ``--dolt-dir``; zapytania
      wykonywane przez ``dolt sql -r csv`` (zalecane przy pełnej historii).

Przed pobraniem weryfikowany jest schemat tabeli (``DESCRIBE``) względem kolumn wymaganych
w ``config/options.yaml``, a po pobraniu – aktualność danych (data ostatniej obserwacji).
Gdy źródło jest niedostępne, ma inny schemat lub jest nieaktualne, fakt ten zapisywany jest
w manifeście, a kalibracja powierzchni korzysta wyłącznie ze zrzutów yfinance.

Uwaga: tabela ``option_chain`` nie zawiera ceny instrumentu bazowego – jest ona dołączana
z ``artifacts/market/underlyings.parquet`` (Close z dnia notowania).

Wyjście: ``artifacts/market/dolthub_options/<SYMBOL>.parquet`` (dopisywanie przyrostowe).

Uruchomienie:
    python -m pf.data.dolthub_options [--dolt-dir ŚCIEŻKA]
"""

from __future__ import annotations

import argparse
import io
import logging
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

from pf import ARTIFACTS_DIR, config
from pf.artifacts import update_manifest, write_parquet
from pf.data.http import RateLimiter, get_json

log = logging.getLogger(__name__)

OUT_DIR = ARTIFACTS_DIR / "market" / "dolthub_options"
PAGE_SIZE = 1000
STALE_AFTER_DAYS = 30
NUMERIC = ["strike", "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho"]


class SchemaError(RuntimeError):
    pass


class DoltHubClient:
    """Wykonuje zapytania SQL przez API DoltHub lub lokalny klon ``dolt``."""

    def __init__(self, cfg: dict, dolt_dir: Path | None = None, min_interval: float = 0.5):
        self.url = cfg["api_url"].format(owner=cfg["owner"], repo=cfg["repo"], branch=cfg["branch"])
        self.dolt_dir = dolt_dir
        self.limiter = RateLimiter(min_interval)

    def query(self, sql: str) -> pd.DataFrame:
        if self.dolt_dir is not None:
            out = subprocess.run(["dolt", "sql", "-r", "csv", "-q", sql], cwd=self.dolt_dir,
                                 check=True, capture_output=True, text=True).stdout
            return pd.read_csv(io.StringIO(out)) if out.strip() else pd.DataFrame()
        self.limiter.wait()
        return parse_api_response(get_json(self.url, params={"q": sql}))


def parse_api_response(payload: dict) -> pd.DataFrame:
    status = payload.get("query_execution_status")
    if status not in ("Success", "RowLimit"):
        raise RuntimeError(f"DoltHub: {status} – {payload.get('query_execution_message')}")
    cols = [c["columnName"] for c in payload.get("schema", [])]
    return pd.DataFrame(payload.get("rows", []), columns=cols or None)


def check_schema(client: DoltHubClient, table: str, required: list[str]) -> list[str]:
    desc = client.query(f"DESCRIBE `{table}`")
    field_col = "Field" if "Field" in desc.columns else desc.columns[0]
    columns = desc[field_col].astype(str).tolist()
    missing = sorted(set(required) - set(columns))
    if missing:
        raise SchemaError(f"Tabela {table}: brak kolumn {missing}; dostępne: {columns}")
    return columns


def list_dates(client: DoltHubClient, table: str, symbol: str, start: str) -> list[str]:
    dates, offset = [], 0
    while True:
        df = client.query(f"SELECT DISTINCT `date` FROM `{table}` WHERE act_symbol = '{symbol}' "
                          f"AND `date` >= '{start}' ORDER BY `date` LIMIT {PAGE_SIZE} OFFSET {offset}")
        dates += df["date"].astype(str).str[:10].tolist() if not df.empty else []
        if len(df) < PAGE_SIZE:
            return dates
        offset += PAGE_SIZE


def fetch_day(client: DoltHubClient, table: str, symbol: str, day: str) -> pd.DataFrame:
    frames, offset = [], 0
    while True:
        df = client.query(f"SELECT * FROM `{table}` WHERE act_symbol = '{symbol}' AND `date` = '{day}' "
                          f"ORDER BY expiration, strike, call_put LIMIT {PAGE_SIZE} OFFSET {offset}")
        frames.append(df)
        if len(df) < PAGE_SIZE:
            return pd.concat(frames, ignore_index=True)
        offset += PAGE_SIZE


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    # call_put bywa zapisane jako "Call"/"Put" – sprowadzamy do C/P jak w zrzutach yfinance.
    df["call_put"] = df["call_put"].astype(str).str.strip().str[0].str.upper()
    return df


def sync_symbol(client: DoltHubClient, cfg: dict, symbol: str, out_dir: Path = OUT_DIR) -> dict:
    path = out_dir / f"{symbol}.parquet"
    existing = pd.read_parquet(path) if path.exists() else None
    start = cfg["start"]
    if existing is not None and not existing.empty:
        start = (existing["date"].max() + pd.Timedelta(days=1)).date().isoformat()
    days = list_dates(client, cfg["table"], symbol, start)
    log.info("%s: %d nowych dni od %s", symbol, len(days), start)
    frames = [normalize(fetch_day(client, cfg["table"], symbol, d)) for d in days]
    new = pd.concat(frames, ignore_index=True) if frames else None
    data = pd.concat([x for x in (existing, new) if x is not None], ignore_index=True) \
        if (existing is not None or new is not None) else pd.DataFrame()
    if data.empty:
        return {"status": "empty", "n_rows": 0}
    write_parquet(data, path)
    last = data["date"].max().date()
    stale = (date.today() - last).days > STALE_AFTER_DAYS
    return {"status": "stale" if stale else "ok", "path": path, "n_rows": len(data),
            "first_date": data["date"].min().date().isoformat(), "last_date": last.isoformat()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dolt-dir", type=Path, default=None, help="lokalny klon repozytorium Dolt")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config.load("options")["dolthub"]
    client = DoltHubClient(cfg, args.dolt_dir)
    source = f"dolt:{args.dolt_dir}" if args.dolt_dir else client.url
    try:
        columns = check_schema(client, cfg["table"], cfg["required_columns"])
    except Exception as exc:  # noqa: BLE001
        log.error("DoltHub niedostępny lub zmieniony schemat (%s) – kalibracja tylko na zrzutach yfinance", exc)
        update_manifest("market/dolthub_options", params={"status": "unavailable", "source": source,
                                                         "error": str(exc)})
        return
    for symbol in cfg["symbols"]:
        res = sync_symbol(client, cfg, symbol)
        path = res.pop("path", None)
        if res["status"] == "stale":
            log.warning("%s: dane DoltHub nieaktualne (ostatnia data %s)", symbol, res["last_date"])
        update_manifest(f"market/dolthub_options/{symbol}", path, n_rows=res.pop("n_rows"),
                        params={"source": source, "columns": columns, **res})


if __name__ == "__main__":
    main()
