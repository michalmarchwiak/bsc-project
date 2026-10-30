"""Notowania instrumentów bazowych modułu opcji: SPY, S&P 500, WIG20, GLD, USO oraz stopa ^IRX.

Do wyceny opcji służy cena nieskorygowana (Close) – ceny wykonania są nominalne; Adj Close
zapisywany jest do liczenia stóp zwrotu. ^IRX (rentowność 13-tygodniowych bonów skarbowych
USA, % p.a., stopa dyskontowa) przybliża stopę wolną od ryzyka r dla opcji USD.

Źródła: Yahoo Finance (podstawowe), Stooq (zapasowe).
Wyjście: ``artifacts/market/underlyings.parquet`` (kolumny wielopoziomowe: nazwa × {Close, Adj Close}).

Uruchomienie:
    python -m pf.data.underlyings
"""

from __future__ import annotations

import io
import logging

import pandas as pd

from pf import ARTIFACTS_DIR, config
from pf.artifacts import update_manifest, write_parquet
from pf.data.http import get_text, retry
from pf.data.vol_indices import validate_series

log = logging.getLogger(__name__)

OUT_PATH = ARTIFACTS_DIR / "market" / "underlyings.parquet"


def fetch_yahoo(symbol: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    def call() -> pd.DataFrame:
        df = yf.download(symbol, start=start, auto_adjust=False, actions=False, progress=False)
        if df is None or df.empty:
            raise ValueError(f"Yahoo zwróciło pusty wynik dla {symbol}")
        return df

    df = retry(call, what=f"yfinance {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df[["Close", "Adj Close"]]


def fetch_stooq(symbol: str, start: str) -> pd.DataFrame:
    text = get_text("https://stooq.pl/q/d/l/", params={"s": symbol, "d1": start.replace("-", ""), "i": "d"})
    df = pd.read_csv(io.StringIO(text))
    df = df.rename(columns={"Data": "Date", "Zamkniecie": "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df["Adj Close"] = df["Close"]  # Stooq podaje ceny skorygowane, brak osobnej kolumny
    return df[["Close", "Adj Close"]]


def download(cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or config.load("options")["underlyings"]
    frames, reports = {}, {}
    for name, sym in cfg["symbols"].items():
        source = "yahoo"
        try:
            df = fetch_yahoo(sym["yahoo"], cfg["start"])
        except Exception as exc:  # noqa: BLE001
            if not sym.get("stooq"):
                log.error("%s: Yahoo niedostępne (%s), brak źródła zapasowego", name, exc)
                reports[name] = {"source": "MISSING", "error": str(exc)}
                continue
            log.warning("%s: Yahoo niedostępne (%s) – używam Stooq", name, exc)
            source = "stooq"
            df = fetch_stooq(sym["stooq"], cfg["start"])
        close, rep = validate_series(df["Close"])
        frames[name] = df.loc[close.index]
        reports[name] = {"source": source, **rep}
    wide = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    return wide, reports


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    wide, reports = download()
    write_parquet(wide, OUT_PATH)
    update_manifest("market/underlyings", OUT_PATH, params={"validation": reports}, n_rows=len(wide))
    log.info("Zapisano %s", OUT_PATH)


if __name__ == "__main__":
    main()
