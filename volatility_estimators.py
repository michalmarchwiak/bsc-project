"""Zakresowe estymatory zmienności Garmana–Klassa i Yanga–Zhanga (2005-01-01 – 2026-10-29).

Oznaczenia (ceny skorygowane, logarytmy naturalne):
    o_t = ln(O_t / C_{t-1})   – stopa zwrotu z przerwy między sesjami (overnight),
    c_t = ln(C_t / O_t)       – stopa zwrotu w trakcie sesji,
    u_t = ln(H_t / O_t),  d_t = ln(L_t / O_t).

Garman, Klass (1980), wariancja dzienna:
    σ²_GK,t = ½ [ln(H_t / L_t)]² − (2 ln 2 − 1) c_t²,
    σ²_GK   = (1/n) Σ σ²_GK,t.

Yang, Zhang (2000), dla okna n sesji:
    σ²_YZ = σ²_o + k σ²_c + (1 − k) σ²_RS,   k = 0,34 / (1,34 + (n + 1)/(n − 1)),
    σ²_o, σ²_c – wariancje próbkowe (dzielnik n − 1) o_t oraz c_t,
    σ²_RS = (1/n) Σ [u_t (u_t − c_t) + d_t (d_t − c_t)]   (Rogers, Satchell 1991).

Punkt odniesienia: klasyczny estymator σ²_CC – wariancja próbkowa dziennych
logarytmicznych stóp zwrotu z cen zamknięcia. Zmienność annualizowana: σ·√252.

Przygotowanie danych:
    * ceny O, H, L, C mnożone przez czynnik a_t = AdjClose_t / Close_t – Yahoo koryguje
      O, H, L jedynie o splity, więc bez korekty o_t zawierałby spadki cen w dniu
      odcięcia dywidendy (składniki śróddzienne są niezmiennicze względem a_t);
    * pomijane są dni z zerowym wolumenem (m.in. święta na GPW uzupełnione przez Yahoo);
    * pomijane są wiersze niespójne: H < max(O, C) lub L > min(O, C).
Estymatory zakresowe mierzą zmienność w czasie sesji przy ciągłym obrocie; dyskretność
notowań (tick) zaniża rozpiętość H − L, co dotyczy zwłaszcza krótkoterminowych ETF-ów
obligacyjnych. Wariant w PLN nie jest liczony – brak dziennych H, L kursu USD/PLN.

Wejście:  data/raw/ohlcv/<TICKER>.csv
Wyjście:
    data/processed/volatility_full_sample.csv            – σ annualizowane dla całej próby
    data/processed/volatility_rolling<w>_<est>.csv       – σ annualizowane w oknach kroczących
    figures/volatility/<TICKER>.png                      – wykresy zmienności kroczącej

Uruchomienie:
    python volatility_estimators.py
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_manipulation import PROCESSED_DIR, RAW_DIR, ROOT, TRADING_DAYS, load_ohlcv

FIG_DIR = ROOT / "figures" / "volatility"
WINDOWS = [21, 63]
ESTIMATORS = {"cc": "close-to-close", "gk": "Garman–Klass", "yz": "Yang–Zhang"}

log = logging.getLogger("volatility_estimators")


def prepare_ohlc(ticker: str) -> tuple[pd.DataFrame, dict[str, int]]:
    df = load_ohlcv(ticker).dropna(subset=["Open", "High", "Low", "Close", "Adj Close"])
    n_raw = len(df)

    df = df[df["Volume"] > 0]
    n_traded = len(df)

    consistent = ((df["High"] >= df[["Open", "Close"]].max(axis=1))
                  & (df["Low"] <= df[["Open", "Close"]].min(axis=1))
                  & (df["Low"] > 0))
    df = df[consistent]

    factor = df["Adj Close"] / df["Close"]
    adj = pd.DataFrame({col: df[col] * factor for col in ["Open", "High", "Low", "Close"]})
    info = {"n_raw": n_raw, "n_zero_volume": n_raw - n_traded,
            "n_inconsistent": n_traded - len(df), "n_flat_range": int((df["High"] == df["Low"]).sum())}
    return adj, info


def daily_components(adj: pd.DataFrame) -> pd.DataFrame:
    log_o, log_h, log_l, log_c = (np.log(adj[c]) for c in ["Open", "High", "Low", "Close"])
    o = log_o - log_c.shift(1)
    c = log_c - log_o
    u = log_h - log_o
    d = log_l - log_o
    return pd.DataFrame({
        "r": log_c.diff(),
        "o": o,
        "c": c,
        "gk": 0.5 * (log_h - log_l) ** 2 - (2 * np.log(2) - 1) * c ** 2,
        "rs": u * (u - c) + d * (d - c),
    }).iloc[1:]  # pierwszy dzień nie ma C_{t-1}


def yz_k(n: int | float) -> float:
    return 0.34 / (1.34 + (n + 1) / (n - 1))


def variances_full(x: pd.DataFrame) -> dict[str, float]:
    n = len(x)
    k = yz_k(n)
    return {
        "cc": x["r"].var(),
        "gk": x["gk"].mean(),
        "yz": x["o"].var() + k * x["c"].var() + (1 - k) * x["rs"].mean(),
    }


def variances_rolling(x: pd.DataFrame, window: int) -> dict[str, pd.Series]:
    roll = x.rolling(window, min_periods=window)
    k = yz_k(window)
    return {
        "cc": roll["r"].var(),
        "gk": roll["gk"].mean(),
        "yz": roll["o"].var() + k * roll["c"].var() + (1 - k) * roll["rs"].mean(),
    }


def annualize(var):
    return np.sqrt(TRADING_DAYS * var)


def plot_ticker(ticker: str, rolling: dict[int, dict[str, pd.Series]], full: dict[str, float],
                path) -> None:
    colors = {"cc": "#7f7f7f", "gk": "#1f4e79", "yz": "#c0392b"}
    fig, axes = plt.subplots(len(WINDOWS), 1, figsize=(12, 3.6 * len(WINDOWS)), sharex=True,
                             constrained_layout=True)
    for ax, window in zip(np.atleast_1d(axes), WINDOWS):
        for est, label in ESTIMATORS.items():
            s = rolling[window][est]
            ax.plot(s.index, s.values, lw=0.9 if est != "cc" else 0.7, color=colors[est],
                    label=rf"$\hat\sigma_{{{est.upper()}}}$ ({label})")
        ax.set_title(f"Zmienność annualizowana, okno kroczące {window} sesji")
        ax.set_ylabel(r"$\hat\sigma$ (p.a.)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"{ticker}: cała próba – " + ", ".join(
            rf"$\hat\sigma_{{{e.upper()}}}$ = {full[e]:.3f}" for e in ESTIMATORS),
        fontsize=11,
    )
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    tickers = sorted(p.stem for p in (RAW_DIR / "ohlcv").glob("*.csv"))
    summary_rows = []
    rolling_tables = {(w, e): {} for w in WINDOWS for e in ESTIMATORS}

    for ticker in tickers:
        adj, info = prepare_ohlc(ticker)
        x = daily_components(adj).dropna()
        if len(x) < max(WINDOWS):
            log.warning("%s: za mało obserwacji (%d) – pomijam", ticker, len(x))
            continue

        full = {e: annualize(v) for e, v in variances_full(x).items()}
        rolling = {w: {e: annualize(v) for e, v in variances_rolling(x, w).items()} for w in WINDOWS}
        for w in WINDOWS:
            for e in ESTIMATORS:
                rolling_tables[(w, e)][ticker] = rolling[w][e]

        summary_rows.append({
            "ticker": ticker, **info, "n_obs": len(x),
            "first_date": x.index.min().date(), "last_date": x.index.max().date(),
            "sigma_cc": full["cc"], "sigma_gk": full["gk"], "sigma_yz": full["yz"],
            "gk_to_cc": full["gk"] / full["cc"], "yz_to_cc": full["yz"] / full["cc"],
            # udział wariancji przerwy międzysesyjnej w σ²_YZ
            "overnight_share_yz": (TRADING_DAYS * x["o"].var()) / full["yz"] ** 2,
        })
        plot_ticker(ticker, rolling, full, FIG_DIR / f"{ticker}.png")

    summary = pd.DataFrame(summary_rows).set_index("ticker")
    summary.to_csv(PROCESSED_DIR / "volatility_full_sample.csv")
    for (w, e), table in rolling_tables.items():
        pd.DataFrame(table).sort_index().to_csv(PROCESSED_DIR / f"volatility_rolling{w}_{e}.csv")

    log.info("Zmienność annualizowana (cała próba):\n%s",
             summary[["sigma_cc", "sigma_gk", "sigma_yz", "gk_to_cc", "yz_to_cc"]]
             .describe().round(3).to_string())
    log.info("Zapisano wyniki w %s oraz wykresy w %s", PROCESSED_DIR, FIG_DIR)


if __name__ == "__main__":
    main()
