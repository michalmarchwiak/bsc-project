"""Dzienne zamknięcia indeksów zmienności Cboe: VIX9D, VIX, VIX3M, VIX6M, VIX1Y, VVIX, SKEW, OVX, GVZ.

Źródła:
    * Cboe (podstawowe) – pliki ``cdn.cboe.com/api/global/us_indices/daily_prices/<SYMBOL>_History.csv``;
      indeksy z pełnym OHLC mają kolumny DATE, OPEN, HIGH, LOW, CLOSE, pozostałe DATE, <SYMBOL>;
    * Yahoo Finance (zapasowe, ``yfinance``) – symbole ``^VIX`` itd.

Walidacja każdego szeregu: duplikaty dat (zostaje ostatnia obserwacja), wartości ≤ 0 i braki
(usuwane), najdłuższa przerwa w dniach roboczych. Raport walidacji trafia do manifestu.

Wyjście: ``artifacts/market/vol_indices.parquet`` (Date × symbol, poziomy indeksów w punktach
procentowych zmienności rocznej).

Uruchomienie:
    python -m pf.data.vol_indices
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd

from pf import ARTIFACTS_DIR, config
from pf.artifacts import update_manifest, write_parquet
from pf.data.http import get_text, retry

log = logging.getLogger(__name__)

OUT_PATH = ARTIFACTS_DIR / "market" / "vol_indices.parquet"


def parse_cboe_csv(text: str, symbol: str) -> pd.Series:
    """Zamienia plik historyczny Cboe na szereg zamknięć indeksowany datą."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    if "DATE" not in df.columns:
        raise ValueError(f"{symbol}: brak kolumny DATE w pliku Cboe ({list(df.columns)})")
    if "CLOSE" in df.columns:
        col = "CLOSE"
    elif symbol.upper() in df.columns:
        col = symbol.upper()
    else:
        col = [c for c in df.columns if c != "DATE"][-1]
    dates = pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="coerce")
    if dates.isna().all():
        dates = pd.to_datetime(df["DATE"], errors="coerce")
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(), index=dates, name=symbol)
    s.index.name = "Date"
    return s[s.index.notna()]


def validate_series(s: pd.Series) -> tuple[pd.Series, dict]:
    """Usuwa duplikaty, braki i wartości ≤ 0; zwraca szereg oczyszczony i raport."""
    s = s.sort_index()
    dup = s.index.duplicated(keep="last")
    s = s[~dup]
    nonpos = s <= 0
    n_missing = int(s.isna().sum())
    clean = s[s.notna() & ~nonpos]
    report = {
        "n_obs": int(clean.size),
        "n_duplicates": int(dup.sum()),
        "n_nonpositive": int(nonpos.sum()),
        "n_missing": n_missing,
        "first_date": clean.index.min().date().isoformat() if clean.size else None,
        "last_date": clean.index.max().date().isoformat() if clean.size else None,
        "max_gap_bdays": int(np.busday_count(clean.index[:-1].values.astype("datetime64[D]"),
                                             clean.index[1:].values.astype("datetime64[D]")).max())
        if clean.size > 1 else None,
    }
    return clean, report


def fetch_cboe(symbol: str, url_template: str) -> pd.Series:
    return parse_cboe_csv(get_text(url_template.format(symbol=symbol)), symbol)


def fetch_yahoo_close(yahoo_symbol: str, start: str, name: str) -> pd.Series:
    import yfinance as yf

    def call() -> pd.DataFrame:
        df = yf.download(yahoo_symbol, start=start, auto_adjust=False, actions=False, progress=False)
        if df is None or df.empty:
            raise ValueError(f"Yahoo zwróciło pusty wynik dla {yahoo_symbol}")
        return df

    df = retry(call, what=f"yfinance {yahoo_symbol}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "Date"
    return close.rename(name)


def download(cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or config.load("options")["vol_indices"]
    series, reports = [], {}
    for symbol, yahoo_symbol in cfg["symbols"].items():
        source = "cboe"
        try:
            s = fetch_cboe(symbol, cfg["cboe_url"])
        except Exception as exc:  # noqa: BLE001
            log.warning("Cboe %s niedostępne (%s) – używam Yahoo %s", symbol, exc, yahoo_symbol)
            source = "yahoo"
            try:
                s = fetch_yahoo_close(yahoo_symbol, cfg["start"], symbol)
            except Exception as exc2:  # noqa: BLE001
                log.error("Nie udało się pobrać %s: %s", symbol, exc2)
                reports[symbol] = {"source": "MISSING", "error": str(exc2)}
                continue
        s = s[s.index >= pd.Timestamp(cfg["start"])]
        clean, rep = validate_series(s)
        reports[symbol] = {"source": source, **rep}
        series.append(clean)
        log.info("%s (%s): %d obserwacji, %s – %s", symbol, source, rep["n_obs"],
                 rep["first_date"], rep["last_date"])
    wide = pd.concat(series, axis=1).sort_index() if series else pd.DataFrame()
    return wide, reports


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    wide, reports = download()
    if wide.empty:
        raise SystemExit("Nie pobrano żadnego indeksu zmienności")
    write_parquet(wide, OUT_PATH)
    update_manifest("market/vol_indices", OUT_PATH, params={"validation": reports}, n_rows=len(wide))
    log.info("Zapisano %s", OUT_PATH)


if __name__ == "__main__":
    main()
