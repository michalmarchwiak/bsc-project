"""Dane rynkowe nakładki opcyjnej: ceny instrumentów bazowych, kursy, stopy i indeksy zmienności w datach t_j.

Ceny instrumentów bazowych odtwarzane są z dziennych log-stóp zwrotu w walucie notowania
(``data/processed/log_returns_local.csv``, ceny skorygowane o dywidendy i splity):

    S_i(d) = A_i · exp(Σ_{s ≤ d} r_{i,s}),

z kontynuacją ostatniej ceny w dniach bez notowania i stałą A_i dobraną tak, aby ostatni poziom równał się
ostatniej cenie skorygowanej z ``data/raw/adj_close.csv`` (opcjonalnie; w przeciwnym razie A_i = 1).
Ceny są skorygowane (szereg total return), więc stopa dywidendy q = 0. Wycena BSM jest jednorodna
stopnia 1 względem (S, K), więc premie względne P/S nie zależą od A_i.

Waluta tickera: PLN dla sufiksów ``.WA`` i ``.SP``, w pozostałych przypadkach USD. Dla tickera w USD
stopa w PLN spełnia r^PLN = r^USD + Δln(USD/PLN) na sesjach tickera (kurs średni NBP z ostatniego fixingu),
więc brutto-stopa okresu w PLN G = g · f, gdzie g – brutto-stopa w USD, f – zmiana kursu między
ostatnimi sesjami tickera ≤ t_open i ≤ t_close. Kurs tickera w dniu d: fx_i(d) = USD/PLN w ostatniej
sesji tickera ≤ d (dla PLN: 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pf.backtest.data import load_returns

PLN_SUFFIXES = (".WA", ".SP")


def currency(ticker: str) -> str:
    return "PLN" if str(ticker).endswith(PLN_SUFFIXES) else "USD"


def price_levels(log_returns: pd.DataFrame, anchor_last: pd.Series | None = None) -> pd.DataFrame:
    """S_i(d) = A_i·exp(Σ_{s ≤ d} r_is) z kontynuacją w dniach bez notowania; NaN przed pierwszą stopą."""
    cum = log_returns.fillna(0.0).cumsum()
    started = log_returns.notna().cummax()
    levels = np.exp(cum).where(started)
    if anchor_last is not None:
        last = levels.ffill().iloc[-1]
        scale = (anchor_last.reindex(levels.columns) / last).where(lambda s: np.isfinite(s) & (s > 0), 1.0)
        levels = levels * scale
    return levels


def last_session(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Data ostatniej sesji z notowaniem ≤ d dla każdego (d, ticker)."""
    idx = pd.Series(log_returns.index, index=log_returns.index)
    out = pd.DataFrame({c: idx.where(log_returns[c].notna()).ffill() for c in log_returns.columns})
    return out.reindex(columns=log_returns.columns)


def asof(series: pd.Series, d, max_stale_days: int | None = None) -> float:
    """Ostatnia wartość szeregu z datą ≤ d (NaN, gdy brak lub starsza niż ``max_stale_days`` dni)."""
    s = series.dropna()
    s = s.loc[s.index <= pd.Timestamp(d)]
    if s.empty:
        return float("nan")
    if max_stale_days is not None and (pd.Timestamp(d) - s.index[-1]).days > max_stale_days:
        return float("nan")
    return float(s.iloc[-1])


@dataclass
class MarketData:
    """Dane wejściowe nakładki; wszystkie szeregi indeksowane datą."""
    r_local: pd.DataFrame                     # log-stopy w walucie notowania (sesje × tickery)
    r_pln: pd.DataFrame                       # log-stopy w PLN
    fx: pd.Series                             # USD/PLN (kurs średni NBP)
    vol: pd.DataFrame                         # indeksy zmienności Cboe [p.p.]
    rate_pln: pd.Series                       # WIBOR 1M (ułamek, ACT/365)
    rate_usd: pd.Series                       # bony 13-tyg. (ułamek, ACT/360)
    spx: pd.Series | None = None              # S&P 500 (zamknięcia) – m_t
    anchor: pd.Series | None = None           # ostatnie ceny skorygowane (poziomy S w rejestrze)
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.levels = price_levels(self.r_local, self.anchor).ffill()
        self.last = last_session(self.r_local)
        fx_ff = self.fx.sort_index()
        self._fx_ff = fx_ff[~fx_ff.index.duplicated(keep="last")]

    @staticmethod
    def _row(df: pd.DataFrame, d, tickers) -> pd.Series:
        pos = df.index.searchsorted(pd.Timestamp(d), side="right") - 1
        if pos < 0 or len(df.index) == 0 or pd.Timestamp(d) > df.index[-1]:   # dane nie sięgają d
            return pd.Series(np.nan, index=list(tickers))
        return df.iloc[pos].reindex(list(tickers))

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.r_local.index

    def spot(self, d, tickers) -> pd.Series:
        """S_i(d) w walucie notowania (ostatnia cena ≤ d)."""
        return self._row(self.levels, d, tickers).astype(float)

    def fx_of(self, d, tickers) -> pd.Series:
        """fx_i(d): USD/PLN z ostatniej sesji tickera ≤ d (PLN: 1)."""
        out = {}
        row = self._row(self.last, d, tickers)
        for t in tickers:
            if currency(t) == "PLN":
                out[t] = 1.0
            else:
                ls = row.get(t)
                out[t] = asof(self._fx_ff, ls) if pd.notna(ls) else float("nan")
        return pd.Series(out, dtype=float)

    def gross(self, sessions: pd.DatetimeIndex, tickers, pln: bool) -> pd.Series:
        """Brutto-stopa okresu exp(Σ r) (brak notowania = 0) w walucie notowania lub w PLN."""
        r = self.r_pln if pln else self.r_local
        return np.exp(r.loc[sessions, list(tickers)].fillna(0.0).sum())

    def vol_level(self, name: str, d, max_stale_days: int = 7) -> float:
        if name not in self.vol.columns:
            return float("nan")
        return asof(self.vol[name], d, max_stale_days)

    def rate(self, ccy: str, d) -> float:
        return asof(self.rate_pln if ccy == "PLN" else self.rate_usd, d)

    def fx_log_returns(self) -> pd.Series:
        """Log-stopy USD/PLN na sesjach kalendarza (zmiana od poprzedniej sesji; NaN w dniach bez fixingu)."""
        return fx_log_returns(self._fx_ff, self.index)


def fx_log_returns(fx: pd.Series, index: pd.DatetimeIndex, name: str = "USDPLN") -> pd.Series:
    """Log-stopy kursu na sesjach ``index``: ln fx(d) − ln fx(d′), d′ – poprzednia sesja (kurs z ostatniego fixingu);
    NaN w sesjach bez fixingu (zmiana kursu przechodzi na następną sesję z fixingiem)."""
    fx = fx.sort_index()
    fx = fx[~fx.index.duplicated(keep="last")]
    lv = fx.reindex(fx.index.union(index)).ffill().reindex(index)
    r = np.log(lv).diff()
    return r.where(index.isin(fx.index)).rename(name)


def load_market(ocfg: dict, bcfg: dict, root: Path) -> MarketData:
    """Wczytanie danych z plików (``config/options.yaml: monthly_overlay.data``)."""
    from pf.data import riskfree
    from pf.data import usd_rate

    d = ocfg["data"]
    info: dict = {}
    r_local = load_returns(root / d["returns_local"])
    r_pln = load_returns(root / d["returns_pln"]).reindex(index=r_local.index, columns=r_local.columns)
    fx = pd.read_csv(root / d["fx"], index_col=0, parse_dates=True)[d.get("fx_name", "USDPLN")].dropna()
    vol = pd.read_parquet(root / d["vol_indices"])
    rate_pln = riskfree.load(root / d.get("pln_rate", bcfg["riskfree"]["path"]))
    usd_path = root / d["usd_rate"]
    if usd_path.exists():
        rate_usd = usd_rate.load(usd_path)
        info["usd_rate_source"] = "usd_rate"
    else:
        und_path = root / d["underlyings"]
        irx = None
        if und_path.exists():
            und = pd.read_parquet(und_path)
            if ("IRX", "Close") in und.columns:
                irx = und[("IRX", "Close")].dropna() / 100.0
        if irx is not None and len(irx):
            rate_usd, info["usd_rate_source"] = irx, "underlyings_irx"
        else:
            rate_usd = pd.Series(ocfg["rates"]["default_usd"], index=r_local.index[:1])
            info["usd_rate_source"] = "default_constant"
    spx = None
    und_path = root / d["underlyings"]
    name = ocfg["implied_vol"].get("spx_name", "SPX")
    if und_path.exists():
        und = pd.read_parquet(und_path)
        if (name, "Close") in und.columns:
            spx = und[(name, "Close")].dropna()
    anchor = None
    raw = root / "data" / "raw" / "adj_close.csv"
    if raw.exists():
        anchor = pd.read_csv(raw, index_col=0, parse_dates=True).ffill().iloc[-1]
    return MarketData(r_local, r_pln, fx, vol, riskfree_rate(rate_pln), rate_usd.astype(float), spx, anchor, info)


def riskfree_rate(df_or_series) -> pd.Series:
    if isinstance(df_or_series, pd.DataFrame):
        return df_or_series["rate"].astype(float)
    return df_or_series.astype(float)
