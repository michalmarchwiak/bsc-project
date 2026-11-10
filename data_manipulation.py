"""Dzienne logarytmiczne stopy zwrotu (2005-01-01 – 2026-10-29) oraz wykresy dla każdego tickera.

Definicja: r_t = ln(P_t / P_{t-1}), gdzie P_t to cena zamknięcia skorygowana (Adj Close).

Stopy liczone są osobno dla każdego instrumentu na jego własnych dniach sesyjnych
(szeroka tabela cen zawiera sumę kalendarzy GPW i giełd USA, więc różnicowanie
bezpośrednio na niej gubiłoby stopy po dniach wolnych na jednej z giełd).
Pomijane są dni z zerowym wolumenem (np. święta na GPW, które Yahoo wypełnia ceną
z poprzedniej sesji), gdyż generowałyby sztuczne zerowe stopy zwrotu.

Obligacje oszczędnościowe Skarbu Państwa (SF10, tickery ``*.SP``) nie są notowane: ich stopy liczone są
z indeksów wartości (``data/raw/savings_bonds/index.csv``, ``pf.data.savings_bonds``) na wszystkich sesjach
tabeli stóp (r_t = ln V(t) − ln V(t′), t′ – poprzednia sesja), w obu wariantach w PLN.

Warianty:
    local – w walucie notowania (PLN dla GPW, USD dla USA),
    pln   – w PLN (ceny USA przeliczone kursem średnim NBP).

Wejście (``data/raw``): adj_close.csv, adj_close_pln.csv
Wyjście:
    data/processed/log_returns_<wariant>.csv        – tabela stóp (Date × ticker)
    data/processed/log_returns_<wariant>_stats.csv  – statystyki opisowe
    data/processed/log_returns_<wariant>_check.json – kontrola kompletności (pełny zakres, brak pustych
                                                       kolumn, wszystkie tickery z ``data_import.TICKERS``
                                                       i obligacji oszczędnościowych)
    data/processed/savings_bonds_redemption_cost.csv – opłata za wcześniejszy wykup obligacji (ułamek
                                                       wartości) na sesjach tabeli stóp
    data/processed/turnover_monthly.csv             – mediana dziennej wartości obrotu (Close·Volume, waluta
                                                       notowania) w miesiącu kalendarzowym; wybór składu
                                                       sub-funduszy punktowo w czasie (``pf.universe``)
    figures/log_returns/<wariant>/<TICKER>.png      – wykresy

Uruchomienie:
    python data_manipulation.py              # tabele i wykresy wszystkich tickerów
    python data_manipulation.py --no-plots   # tylko tabele (przy szerokich pulach spółek wykresów jest kilkaset)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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

sys.path.insert(0, str(ROOT / "src"))
from data_import import TICKERS  # noqa: E402
from pf.data import savings_bonds  # noqa: E402
from pf.data.quality import returns_report  # noqa: E402

log = logging.getLogger("data_manipulation")


def load_ohlcv(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "ohlcv" / f"{ticker}.csv", index_col="Date", parse_dates=True)
    return df.sort_index().loc[START:END]


def non_trading_days(ticker: str) -> pd.DatetimeIndex:
    """Dni z zerowym wolumenem – Yahoo uzupełnia nimi m.in. święta na GPW (cena nieaktualna).

    Szereg bez dodatniego wolumenu w całej historii (indeks giełdowy) nie jest filtrowany."""
    path = RAW_DIR / "ohlcv" / f"{ticker}.csv"
    if not path.exists():
        return pd.DatetimeIndex([])
    df = load_ohlcv(ticker)
    if not (df["Volume"] > 0).any():
        return pd.DatetimeIndex([])
    return df.index[df["Volume"] == 0]


def monthly_turnover(tickers: list[str]) -> pd.DataFrame:
    """Mediana dziennej wartości obrotu Close·Volume (waluta notowania) w miesiącu kalendarzowym.

    Dni bez obrotu są pomijane; miesiąc bez notowań – NaN. Indeks: pierwszy dzień miesiąca."""
    out = {}
    for t in tickers:
        if not (RAW_DIR / "ohlcv" / f"{t}.csv").exists():
            continue
        df = load_ohlcv(t)
        value = (df["Close"] * df["Volume"]).where(df["Volume"] > 0).dropna()
        if len(value):
            out[t] = value.groupby(value.index.to_period("M")).median()
    table = pd.DataFrame(out).sort_index()
    table.index = table.index.to_timestamp()
    table.index.name = "Date"
    return table


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


def add_savings_bonds(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Dołącza log-stopy obligacji oszczędnościowych na sesjach tabeli; zwraca też opłaty za wykup."""
    cfg = savings_bonds.load_config()
    if not (ROOT / cfg["index_path"]).exists():
        log.warning("Brak %s – pomijam obligacje oszczędnościowe (python -m pf.data.savings_bonds)",
                    cfg["index_path"])
        return returns, None
    index, cost = savings_bonds.load_index(cfg)
    bonds = savings_bonds.log_returns_on(index, returns.index)
    return pd.concat([returns, bonds], axis=1), savings_bonds.cost_on(cost, returns.index)


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
    if variant == "pln" or ticker.endswith((".WA", ".SP")):
        return "PLN"
    return "USD"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dzienne log-stopy zwrotu, statystyki, obroty i wykresy")
    parser.add_argument("--no-plots", action="store_true", help="bez wykresów figures/log_returns/")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    monthly_turnover(TICKERS).to_csv(PROCESSED_DIR / "turnover_monthly.csv", float_format="%.6g")
    log.info("Zapisano miesięczne mediany obrotów (%s)", PROCESSED_DIR / "turnover_monthly.csv")

    for variant, price_path in VARIANTS.items():
        if not price_path.exists():
            log.warning("Brak pliku %s – pomijam wariant %s", price_path, variant)
            continue

        prices = load_prices(price_path)
        returns, bond_cost = add_savings_bonds(log_returns(prices))
        returns.to_csv(PROCESSED_DIR / f"log_returns_{variant}.csv")
        if bond_cost is not None:
            bond_cost.to_csv(PROCESSED_DIR / "savings_bonds_redemption_cost.csv", float_format="%.8g")
        describe(returns).to_csv(PROCESSED_DIR / f"log_returns_{variant}_stats.csv")
        log.info("%s: %d tickerów, %d dni", variant, returns.shape[1], returns.shape[0])
        check = returns_report(returns, TICKERS + savings_bonds.tickers(), START, END)
        (PROCESSED_DIR / f"log_returns_{variant}_check.json").write_text(
            json.dumps(check, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not check["ok"]:
            log.error("%s: dane niekompletne: %s", variant, check)

        if args.no_plots:
            continue
        fig_dir = FIG_DIR / variant
        fig_dir.mkdir(parents=True, exist_ok=True)
        for ticker in returns.columns:
            plot_ticker(returns[ticker], ticker, currency_label(ticker, variant),
                        fig_dir / f"{ticker}.png")
        log.info("Zapisano wykresy w %s", fig_dir)


if __name__ == "__main__":
    main()
