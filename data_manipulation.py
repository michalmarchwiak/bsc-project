"""Dzienne logarytmiczne stopy zwrotu (2005-01-01 – 2026-10-29) oraz wykresy dla każdego tickera.

Definicja: r_t = ln(P_t / P_{t-1}), gdzie P_t to cena zamknięcia skorygowana (Adj Close).

Stopy liczone są osobno dla każdego instrumentu na jego własnych dniach sesyjnych
(szeroka tabela cen zawiera sumę kalendarzy GPW i giełd USA, więc różnicowanie
bezpośrednio na niej gubiłoby stopy po dniach wolnych na jednej z giełd).
Pomijane są dni z zerowym wolumenem (np. święta na GPW, które Yahoo wypełnia ceną
z poprzedniej sesji), gdyż generowałyby sztuczne zerowe stopy zwrotu.

Warianty:
    local – w walucie notowania (PLN dla GPW, USD dla USA),
    pln   – w PLN (ceny USA przeliczone kursem średnim NBP).

Wejście (``data/raw``): adj_close.csv, adj_close_pln.csv
Wyjście:
    data/processed/log_returns_<wariant>.csv        – tabela stóp (Date × ticker)
    data/processed/log_returns_<wariant>_stats.csv  – statystyki opisowe
    figures/log_returns/<wariant>/<TICKER>.png      – wykresy

Uruchomienie:
    python data_manipulation.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

START = "2005-01-01"
END = "2026-10-29"

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures" / "log_returns"

VARIANTS = {
    "local": RAW_DIR / "adj_close.csv",
    "pln": RAW_DIR / "adj_close_pln.csv",
}

TRADING_DAYS = 252

log = logging.getLogger("data_manipulation")


def load_ohlcv(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "ohlcv" / f"{ticker}.csv", index_col="Date", parse_dates=True)
    return df.sort_index().loc[START:END]


def non_trading_days(ticker: str) -> pd.DatetimeIndex:
    """Dni z zerowym wolumenem – Yahoo uzupełnia nimi m.in. święta na GPW (cena nieaktualna)."""
    path = RAW_DIR / "ohlcv" / f"{ticker}.csv"
    if not path.exists():
        return pd.DatetimeIndex([])
    df = load_ohlcv(ticker)
    return df.index[df["Volume"] == 0]


def load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
    prices = prices.loc[START:END].dropna(axis=1, how="all")
    for ticker in prices.columns:
        prices.loc[prices.index.intersection(non_trading_days(ticker)), ticker] = np.nan
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = {}
    for ticker in prices.columns:
        p = prices[ticker].dropna()
        p = p[p > 0]
        returns[ticker] = np.log(p).diff().dropna()
    return pd.DataFrame(returns).reindex(columns=prices.columns)


def describe(returns: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame({
        "n_obs": returns.count(),
        "first_date": returns.apply(pd.Series.first_valid_index).dt.date,
        "last_date": returns.apply(pd.Series.last_valid_index).dt.date,
        "mean": returns.mean(),
        "std": returns.std(),
        "mean_ann": returns.mean() * TRADING_DAYS,
        "std_ann": returns.std() * np.sqrt(TRADING_DAYS),
        "min": returns.min(),
        "max": returns.max(),
        "skewness": returns.skew(),
        "excess_kurtosis": returns.kurt(),
        "cum_log_return": returns.sum(),
    })
    stats.index.name = "ticker"
    return stats


def plot_ticker(r: pd.Series, ticker: str, currency_label: str, path: Path) -> None:
    r = r.dropna()
    mu, sigma = r.mean(), r.std()

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    ax_ts = fig.add_subplot(gs[0, :])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_cum = fig.add_subplot(gs[1, 1])

    ax_ts.plot(r.index, r.values, lw=0.6, color="#1f4e79")
    ax_ts.axhline(0, color="black", lw=0.6)
    ax_ts.set_title("Dzienne logarytmiczne stopy zwrotu")
    ax_ts.set_ylabel(r"$r_t$")
    ax_ts.grid(alpha=0.3)

    ax_hist.hist(r.values, bins=100, density=True, color="#1f4e79", alpha=0.7, label="empiryczny")
    x = np.linspace(r.min(), r.max(), 400)
    ax_hist.plot(x, np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)),
                 color="#c0392b", lw=1.2, label=rf"$N(\hat\mu, \hat\sigma^2)$")
    ax_hist.set_title("Rozkład stóp zwrotu")
    ax_hist.set_xlabel(r"$r_t$")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.3)

    ax_cum.plot(r.index, r.cumsum().values, lw=1.0, color="#1f4e79")
    ax_cum.axhline(0, color="black", lw=0.6)
    ax_cum.set_title(r"Skumulowana stopa logarytmiczna $\sum r_t$")
    ax_cum.grid(alpha=0.3)

    fig.suptitle(
        f"{ticker} ({currency_label}), {r.index.min():%Y-%m-%d} – {r.index.max():%Y-%m-%d}\n"
        rf"$\hat\mu$ = {mu:.5f}, $\hat\sigma$ = {sigma:.5f}, "
        f"skośność = {r.skew():.2f}, kurtoza nadwyżkowa = {r.kurt():.2f}, n = {len(r)}",
        fontsize=11,
    )
    fig.savefig(path, dpi=120)
    plt.close(fig)


def currency_label(ticker: str, variant: str) -> str:
    if variant == "pln" or ticker.endswith(".WA"):
        return "PLN"
    return "USD"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for variant, price_path in VARIANTS.items():
        if not price_path.exists():
            log.warning("Brak pliku %s – pomijam wariant %s", price_path, variant)
            continue

        prices = load_prices(price_path)
        returns = log_returns(prices)
        returns.to_csv(PROCESSED_DIR / f"log_returns_{variant}.csv")
        describe(returns).to_csv(PROCESSED_DIR / f"log_returns_{variant}_stats.csv")
        log.info("%s: %d tickerów, %d dni", variant, returns.shape[1], returns.shape[0])

        fig_dir = FIG_DIR / variant
        fig_dir.mkdir(parents=True, exist_ok=True)
        for ticker in returns.columns:
            plot_ticker(returns[ticker], ticker, currency_label(ticker, variant),
                        fig_dir / f"{ticker}.png")
        log.info("Zapisano wykresy w %s", fig_dir)


if __name__ == "__main__":
    main()
