"""Dzienny zrzut łańcuchów opcji z Yahoo Finance (``yfinance``) dla SPY (opcjonalnie ^SPX, GLD, USO).

Zapisywane są wszystkie terminy wygaśnięcia, kolumny: strike, bid, ask, lastPrice, volume,
openInterest, impliedVolatility, lastTradeDate, cena instrumentu bazowego i znacznik czasu
pobrania (UTC). Dane surowe dopisywane są do partycji ``date=<RRRR-MM-DD>`` (data sesji
w Nowym Jorku), bez jakiejkolwiek obróbki.

Zasady:
    * moduł uruchamiany wyłącznie offline (cron w GitHub Actions) – aplikacja nigdy nie
      odpytuje Yahoo;
    * ograniczenie częstotliwości zapytań i ponawianie z wykładniczym backoffem;
    * kolumna ``impliedVolatility`` z Yahoo jest zachowana tylko informacyjnie – zmienność
      implikowaną liczymy sami (``pf.options.iv``) ze środka bid/ask.
Dane Yahoo są nieoficjalne i przeznaczone do użytku osobistego/badawczego.

Wyjście: ``artifacts/raw/options_snapshots/date=<RRRR-MM-DD>/<SYMBOL>.parquet``

Uruchomienie:
    python -m pf.data.options_snapshot [--symbols SPY GLD]
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

from pf import ARTIFACTS_DIR, config
from pf.artifacts import update_manifest, write_parquet
from pf.data.http import RateLimiter, retry

log = logging.getLogger(__name__)

OUT_DIR = ARTIFACTS_DIR / "raw" / "options_snapshots"

CHAIN_COLUMNS = ["contractSymbol", "strike", "bid", "ask", "lastPrice", "volume", "openInterest",
                 "impliedVolatility", "lastTradeDate"]
OUTPUT_COLUMNS = ["symbol", "expiration", "call_put", *CHAIN_COLUMNS, "underlying_price", "fetched_at_utc"]


def session_date(ts_utc: pd.Timestamp) -> str:
    """Data sesji NYSE odpowiadająca chwili pobrania (strefa America/New_York)."""
    return ts_utc.tz_convert("America/New_York").date().isoformat()


def normalize_chain(calls: pd.DataFrame, puts: pd.DataFrame, symbol: str, expiration: str,
                    underlying_price: float, fetched_at: pd.Timestamp) -> pd.DataFrame:
    """Łączy tabele call/put z yfinance w jeden długi format o stałym schemacie."""
    parts = []
    for cp, df in (("C", calls), ("P", puts)):
        if df is None or df.empty:
            continue
        d = df.reindex(columns=CHAIN_COLUMNS).copy()
        d.insert(0, "call_put", cp)
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "expiration", pd.Timestamp(expiration))
    out.insert(0, "symbol", symbol)
    out["lastTradeDate"] = pd.to_datetime(out["lastTradeDate"], utc=True)
    for c in ("strike", "bid", "ask", "lastPrice", "impliedVolatility"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    for c in ("volume", "openInterest"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    out["underlying_price"] = float(underlying_price)
    out["fetched_at_utc"] = fetched_at
    return out[OUTPUT_COLUMNS]


def underlying_price(ticker) -> float:
    try:
        px = ticker.fast_info["lastPrice"]
        if px and px > 0:
            return float(px)
    except Exception:  # noqa: BLE001
        pass
    hist = ticker.history(period="5d", auto_adjust=False)
    return float(hist["Close"].dropna().iloc[-1])


def snapshot_symbol(symbol: str, limiter: RateLimiter, retries: int, backoff: float) -> pd.DataFrame:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    limiter.wait()
    expirations = retry(lambda: list(ticker.options), retries, backoff, what=f"{symbol} terminy")
    limiter.wait()
    spot = retry(lambda: underlying_price(ticker), retries, backoff, what=f"{symbol} cena")
    fetched_at = pd.Timestamp(datetime.now(timezone.utc))
    frames = []
    for exp in expirations:
        limiter.wait()
        chain = retry(lambda: ticker.option_chain(exp), retries, backoff, what=f"{symbol} {exp}")
        frames.append(normalize_chain(chain.calls, chain.puts, symbol, exp, spot, fetched_at))
    log.info("%s: %d terminów, %d serii, S = %.2f", symbol, len(expirations),
             sum(len(f) for f in frames), spot)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)


def append_partition(df: pd.DataFrame, symbol: str, out_dir=OUT_DIR):
    """Dopisuje zrzut do partycji dnia sesji; powtórny zrzut tego samego dnia jest dołączany."""
    path = out_dir / f"date={session_date(df['fetched_at_utc'].iloc[0])}" / f"{symbol.replace('^', '')}.parquet"
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df = df.drop_duplicates(subset=["contractSymbol", "fetched_at_utc"], keep="last")
    write_parquet(df, path)
    return path, len(df)


def main() -> None:
    cfg = config.load("options")["options_snapshot"]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbols", nargs="+", default=cfg["symbols"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    limiter = RateLimiter(cfg["min_interval_s"])
    failed = []
    for symbol in args.symbols:
        try:
            df = snapshot_symbol(symbol, limiter, cfg["retries"], cfg["backoff_s"])
        except Exception as exc:  # noqa: BLE001
            log.error("%s: zrzut nieudany (%s)", symbol, exc)
            failed.append(symbol)
            continue
        if df.empty:
            log.warning("%s: pusty łańcuch opcji", symbol)
            continue
        path, n = append_partition(df, symbol)
        update_manifest(f"raw/options_snapshots/{symbol}/latest", path, n_rows=n,
                        params={"source": "yfinance", "note": "dane nieoficjalne, użytek osobisty"})
    if failed and len(failed) == len(args.symbols):
        raise SystemExit(f"Nie udało się pobrać żadnego łańcucha: {failed}")


if __name__ == "__main__":
    main()
