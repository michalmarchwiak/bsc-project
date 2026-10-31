"""Pobieranie notowań dziennych (2005-01-01 – 2026-10-29) oraz kursów walut USD/PLN, EUR/PLN.

Źródła:
    * Yahoo Finance (biblioteka ``yfinance``) – źródło podstawowe dla wszystkich tickerów
      (OHLC, Close oraz Adj Close skorygowany o dywidendy i splity).
    * Stooq (stooq.pl / stooq.com, CSV) – źródło zapasowe, gdy Yahoo nie zwróci danych także
      przy ponownym zapytaniu o pojedynczy ticker (spółki GPW: ``pko``, akcje/ETF USA: ``aapl.us``;
      wyjątki w ``STOOQ_SYMBOLS``, np. AVB → ``avb.us``). Odpowiedź z komunikatem o limicie
      zapytań lub bez nagłówka CSV jest odrzucana; opcjonalny klucz API: zmienna ``STOOQ_APIKEY``.
    * NBP API (api.nbp.pl, tabela A, kurs średni) – oficjalne kursy walut względem PLN.

Waluty notowań: tickery z sufiksem ``.WA`` – PLN, pozostałe (NYSE/NASDAQ/NYSE Arca) – USD.
Żaden z tickerów nie jest notowany w EUR, więc EUR/PLN jest pobierany tylko gdy
``ALWAYS_FETCH_EUR = True`` lub gdy na liście pojawi się instrument w EUR.

Uruchomienie:
    python data_import.py

Wyniki (katalog ``data/raw``):
    ohlcv/<TICKER>.csv   – pełne notowania dla każdego instrumentu
    adj_close.csv        – ceny zamknięcia skorygowane (szeroki format, waluta notowania)
    close.csv            – ceny zamknięcia nieskorygowane
    fx_nbp.csv           – kursy średnie NBP (PLN za 1 jednostkę waluty)
    adj_close_pln.csv    – ceny skorygowane przeliczone na PLN (kurs NBP, forward-fill)
    coverage.csv         – raport pokrycia danych (źródło, pierwsza/ostatnia data, liczba obserwacji)
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

START = "2005-01-01"
END = "2026-10-29"

OUT_DIR = Path(__file__).resolve().parent / "data" / "raw"

ALWAYS_FETCH_EUR = False

TICKERS_PL = [
    "PKO.WA", "PZU.WA", "KGH.WA", "PKN.WA", "PEO.WA",
    "LPP.WA", "CDR.WA", "DNP.WA", "PGE.WA", "CPS.WA",
]
TICKERS_US_EQUITY = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "JNJ", "XOM", "PG", "WMT"]
TICKERS_US_TREASURY = ["SHY", "IEI", "IEF", "TLT", "GOVT", "VGSH", "VGIT", "VGLT", "SCHO", "SCHR"]
TICKERS_CORP_IG = ["LQD", "VCIT", "VCSH", "IGSB", "IGIB", "USIG"]
TICKERS_CORP_HY = ["HYG", "JNK", "ANGL", "FALN"]
TICKERS_DEVELOPED = ["EFA", "EWJ", "EWU", "EWG", "EWQ", "EWL", "EWC", "EWA", "EWN", "EWS"]
TICKERS_EMERGING = ["EEM", "MCHI", "INDA", "EWZ", "EWW", "EWT", "THD", "IDX", "TUR", "EZA"]
TICKERS_GOLD = ["GLD", "IAU", "SGOL", "GDX", "GDXJ", "NEM", "AEM", "KGC", "WPM", "FNV"]
TICKERS_REIT = ["PLD", "AMT", "EQIX", "O", "SPG", "PSA", "WELL", "DLR", "AVB", "EXR"]

TICKERS = (
    TICKERS_PL + TICKERS_US_EQUITY + TICKERS_US_TREASURY + TICKERS_CORP_IG + TICKERS_CORP_HY
    + TICKERS_DEVELOPED + TICKERS_EMERGING + TICKERS_GOLD + TICKERS_REIT
)

# Symbole Stooq odbiegające od reguły (zapas, gdy Yahoo nie zwróci danych). AVB brakowało w poprzednim
# pobraniu (coverage.csv: MISSING) – symbol podany jawnie.
STOOQ_SYMBOLS: dict[str, str] = {"AVB": "avb.us"}
STOOQ_URLS = ["https://stooq.pl/q/d/l/", "https://stooq.com/q/d/l/"]
YAHOO_RETRIES = 3

# Waluta notowań; wyjątki od reguły ".WA" -> PLN, reszta -> USD można dopisać tutaj.
CURRENCY_OVERRIDES: dict[str, str] = {}

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

log = logging.getLogger("data_import")


def quote_currency(ticker: str) -> str:
    if ticker in CURRENCY_OVERRIDES:
        return CURRENCY_OVERRIDES[ticker]
    return "PLN" if ticker.endswith(".WA") else "USD"


# --------------------------------------------------------------------------- Yahoo Finance

def fetch_yahoo(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    # yfinance traktuje `end` jako wyłączny, stąd +1 dzień.
    end_excl = (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers,
        start=start,
        end=end_excl,
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    result: dict[str, pd.DataFrame] = {}
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t].copy()
        else:
            df = raw.copy()
        df = df.reindex(columns=OHLCV_COLUMNS).dropna(how="all", subset=["Close"])
        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"
            result[t] = df
    return result


# --------------------------------------------------------------------------- Stooq

def stooq_symbol(ticker: str) -> str:
    if ticker in STOOQ_SYMBOLS:
        return STOOQ_SYMBOLS[ticker]
    if ticker.endswith(".WA"):
        return ticker[:-3].lower()
    return f"{ticker.lower()}.us"


STOOQ_RENAME = {"Data": "Date", "Otwarcie": "Open", "Najwyzszy": "High", "Najnizszy": "Low",
                "Zamkniecie": "Close", "Wolumen": "Volume"}


def parse_stooq_prices(text: str) -> pd.DataFrame | None:
    """CSV Stooq (nagłówki PL lub EN) → OHLCV; None przy komunikacie zamiast danych (limit, brak danych)."""
    text = text.strip()
    if not text:
        return None
    header = text.splitlines()[0]
    if "Date" not in header and "Data" not in header:
        return None  # np. „Przekroczony dzienny limit wywolan”, „Brak danych”
    df = pd.read_csv(io.StringIO(text)).rename(columns=STOOQ_RENAME)
    if "Close" not in df.columns or df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.index.name = "Date"
    # Stooq podaje ceny skorygowane o splity i dywidendy – brak osobnej kolumny Adj Close.
    df["Adj Close"] = df["Close"]
    return df.reindex(columns=OHLCV_COLUMNS)


def fetch_stooq(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    params = {
        "s": stooq_symbol(ticker),
        "d1": start.replace("-", ""),
        "d2": end.replace("-", ""),
        "i": "d",
    }
    if os.environ.get("STOOQ_APIKEY"):
        params["apikey"] = os.environ["STOOQ_APIKEY"]
    for url in STOOQ_URLS:
        try:
            resp = requests.get(url, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("Stooq %s (%s): %s", ticker, url, exc)
            continue
        df = parse_stooq_prices(resp.text)
        if df is not None and not df.empty:
            return df
        log.warning("Stooq %s (%s): odpowiedź bez danych: %r", ticker, url, resp.text[:120])
    return None


def fetch_yahoo_single(ticker: str, start: str, end: str, retries: int = YAHOO_RETRIES) -> pd.DataFrame | None:
    """Ponowne zapytanie Yahoo o pojedynczy ticker (zbiorcze pobranie gubi tickery przy limicie zapytań)."""
    for attempt in range(retries):
        try:
            df = fetch_yahoo([ticker], start, end).get(ticker)
            if df is not None and not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001
            log.warning("Yahoo %s (próba %d): %s", ticker, attempt + 1, exc)
        time.sleep(2 ** (attempt + 1))
    return None


# --------------------------------------------------------------------------- NBP

def fetch_nbp_fx(currency: str, start: str, end: str) -> pd.Series:
    """Kursy średnie NBP (tabela A); API ogranicza zapytanie do 93 dni."""
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    frames = []
    while d0 <= d1:
        chunk_end = min(d0 + timedelta(days=92), d1)
        url = (f"https://api.nbp.pl/api/exchangerates/rates/a/{currency.lower()}/"
               f"{d0.isoformat()}/{chunk_end.isoformat()}/")
        for attempt in range(4):
            try:
                resp = requests.get(url, params={"format": "json"}, timeout=30)
                if resp.status_code == 404:  # brak notowań w danym okresie
                    break
                resp.raise_for_status()
                rates = resp.json()["rates"]
                frames.append(pd.DataFrame(rates)[["effectiveDate", "mid"]])
                break
            except requests.RequestException as exc:
                if attempt == 3:
                    raise
                log.warning("NBP %s %s: %s – ponawiam", currency, d0, exc)
                time.sleep(2 ** (attempt + 1))
        d0 = chunk_end + timedelta(days=1)
    df = pd.concat(frames, ignore_index=True)
    s = df.set_index(pd.to_datetime(df["effectiveDate"]))["mid"].sort_index()
    s.index.name = "Date"
    s.name = f"{currency.upper()}PLN"
    return s


def fetch_yahoo_fx(currency: str, start: str, end: str) -> pd.Series:
    """Zapasowe kursy z Yahoo (USD/PLN: 'PLN=X', EUR/PLN: 'EURPLN=X')."""
    symbol = "PLN=X" if currency.upper() == "USD" else f"{currency.upper()}PLN=X"
    data = fetch_yahoo([symbol], start, end)
    s = data[symbol]["Close"]
    s.name = f"{currency.upper()}PLN"
    return s


# --------------------------------------------------------------------------- main

def download_prices() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    log.info("Yahoo Finance: %d tickerów", len(TICKERS))
    prices = fetch_yahoo(TICKERS, START, END)
    sources = {t: "yahoo" for t in prices}

    for t in [t for t in TICKERS if t not in prices]:
        log.info("Brak danych z Yahoo dla %s – ponawiam zapytanie dla pojedynczego tickera", t)
        df = fetch_yahoo_single(t, START, END)
        if df is not None:
            prices[t], sources[t] = df, "yahoo"
            continue
        log.info("Brak danych z Yahoo dla %s – próbuję Stooq (%s)", t, stooq_symbol(t))
        df = fetch_stooq(t, START, END)
        if df is not None and not df.empty:
            prices[t] = df
            sources[t] = "stooq"
        else:
            log.error("Nie udało się pobrać %s", t)
    return prices, sources


def download_fx(currencies: set[str]) -> pd.DataFrame:
    series = []
    for cur in sorted(currencies):
        try:
            s = fetch_nbp_fx(cur, START, END)
            log.info("NBP %s/PLN: %d obserwacji", cur, len(s))
        except Exception as exc:  # noqa: BLE001
            log.warning("NBP %s niedostępne (%s) – używam Yahoo", cur, exc)
            s = fetch_yahoo_fx(cur, START, END)
        series.append(s)
    return pd.concat(series, axis=1).sort_index() if series else pd.DataFrame()


def to_pln(adj_close: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    # Kurs NBP z dnia t (lub ostatni dostępny) – forward-fill na dni sesyjne giełd.
    fx_aligned = fx.reindex(adj_close.index.union(fx.index)).ffill().reindex(adj_close.index)
    out = adj_close.copy()
    for t in out.columns:
        cur = quote_currency(t)
        if cur != "PLN":
            out[t] = out[t] * fx_aligned[f"{cur}PLN"]
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    (OUT_DIR / "ohlcv").mkdir(parents=True, exist_ok=True)

    prices, sources = download_prices()
    for t, df in prices.items():
        df.to_csv(OUT_DIR / "ohlcv" / f"{t}.csv")

    ordered = [t for t in TICKERS if t in prices]
    adj_close = pd.DataFrame({t: prices[t]["Adj Close"] for t in ordered}).sort_index()
    close = pd.DataFrame({t: prices[t]["Close"] for t in ordered}).sort_index()
    adj_close.to_csv(OUT_DIR / "adj_close.csv")
    close.to_csv(OUT_DIR / "close.csv")

    currencies = {quote_currency(t) for t in TICKERS} - {"PLN"}
    if ALWAYS_FETCH_EUR:
        currencies.add("EUR")
    fx = download_fx(currencies)
    fx.to_csv(OUT_DIR / "fx_nbp.csv")
    to_pln(adj_close, fx).to_csv(OUT_DIR / "adj_close_pln.csv")

    coverage = pd.DataFrame([
        {
            "ticker": t,
            "currency": quote_currency(t),
            "source": sources.get(t, "MISSING"),
            "first_date": prices[t].index.min().date() if t in prices else None,
            "last_date": prices[t].index.max().date() if t in prices else None,
            "n_obs": len(prices[t]) if t in prices else 0,
        }
        for t in TICKERS
    ])
    coverage.to_csv(OUT_DIR / "coverage.csv", index=False)

    late = coverage[pd.to_datetime(coverage["first_date"]) > pd.Timestamp(START) + timedelta(days=10)]
    if not late.empty:
        log.warning("Instrumenty z historią krótszą niż okres badania:\n%s",
                    late[["ticker", "first_date"]].to_string(index=False))
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from pf.data.quality import coverage_report

    report = coverage_report(coverage, START, END)
    if report["missing"]:
        log.error("Brak danych dla: %s", ", ".join(report["missing"]))
    if report["stale"]:
        log.error("Notowania kończą się przed %s: %s", END, ", ".join(report["stale"]))
    log.info("Zapisano dane w %s (źródła: %s)", OUT_DIR, report["sources"])


if __name__ == "__main__":
    main()