"""Obligacje oszczędnościowe Skarbu Państwa (SF10): indeksy wartości rolowanych pozycji i opłaty za wykup.

Dane: zbiór „Obligacje oszczędnościowe Skarbu Państwa: emisje i oprocentowanie” (Błażewski 2026,
obrotomierz.pl, DOI 10.5281/zenodo.23208023, CC BY 4.0; dane MF i listy emisyjne): ``emisje.csv``
(seria, rodzina, okno sprzedaży, termin wykupu, okres odsetkowy, kapitalizacja, opłata za wcześniejszy
wykup, cena zamiany) i ``oprocentowanie.csv`` (stopa każdej serii w każdym okresie odsetkowym).

Indeks rodziny F (ticker ``<F>.SP``, PLN, 100 = wartość w dniu pierwszego zakupu) to wartość pozycji
kupionej w pierwszej serii F w sprzedaży od ``start`` i rolowanej przy wykupie:

* seria kupiona w dniu p ma okresy odsetkowe [b_{k−1}, b_k), b_k = p + k·m miesięcy (m = 1, 6, 12 albo
  cały termin – OTS), stopę r_k (% p.a.) i ułamek roku τ_k = m/12;
* wartość jednej obligacji (nominał 100 zł) w dniu d ∈ [b_{k−1}, b_k):

      u(d) = 100 · Π_{j<k} (1 + r_j τ_j) · (1 + r_k τ_k · (d − b_{k−1}) / (b_k − b_{k−1})),

  tj. odsetki narastają liniowo w okresie, a na jego końcu są kapitalizowane (EDO, DOS, TOS) albo
  wypłacane jako kupon (COI, TOZ, ROR, DOR) i – w indeksie całkowitego zwrotu – reinwestowane
  w tę samą pozycję;
* w dniu wykupu b_K wartość pozycji V jest reinwestowana w serię tej rodziny w sprzedaży w tym dniu
  po cenie zamiany c (``cena_zamiany_zl``, brak – 100 zł): V⁺ = V · 100 / c. Gdy sprzedaż kolejnej serii
  zaczyna się kilka dni później, środki do tego dnia nie są oprocentowane; brak serii kończy indeks.

Stopy okresów, które zaczęły się po dacie danych MF (brak w ``oprocentowanie.csv``), przyjmowane są
równe stopie poprzedniego okresu (pierwszego okresu: ``oprocentowanie_1_okres_pct``).

Opłata za wcześniejszy wykup (koszt sprzedaży pozycji w backteście), jako ułamek wartości obligacji:

    f(d) = min(o, a(d)) / (100 + a(d)),

o – opłata z listu emisyjnego [zł/obligację] (brak w danych → 0), a(d) – odsetki narosłe od zakupu
(kapitalizacja) lub od początku bieżącego okresu (kupon); obligacje OTS („bez odsetek”): f = a/(100 + a).
Ograniczenie opłaty do narosłych odsetek odpowiada listom emisyjnym obecnych serii i jest przyjęte dla
wszystkich serii. Podatek od dochodów kapitałowych pominięto (jak dla pozostałych aktywów).

Dzienne log-stopy na sesjach panelu danych (``log_returns_on``): r_t = ln V(t) − ln V(t′), t′ – poprzednia
sesja panelu (stopa sesji obejmuje wszystkie dni kalendarzowe od t′). Stopa w horyzoncie miesięcznym
jest znana z góry (z dokładnością do zmiany stopy na początku nowego okresu odsetkowego), stąd margines
deterministyczny w ``pf.models.margins``.

Uruchomienie (pobranie danych i zapis indeksów):
    python -m pf.data.savings_bonds [--offline]
"""

from __future__ import annotations

import argparse
import io
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pf import ROOT, config

log = logging.getLogger(__name__)

PERIOD_MONTHS = {"miesięczny": 1, "półroczny": 6, "roczny": 12}
CONSTANT_RATE = "cały okres (stała stopa)"
NO_INTEREST = "bez odsetek"
MAX_GAP_DAYS = 10          # najdłuższe oczekiwanie na początek sprzedaży serii przy rolowaniu
SESSION_FFILL_DAYS = 7     # wartość po wykupie bez rolowania przenoszona na najbliższe sesje


def load_config() -> dict:
    return config.load("savings_bonds")


def tickers(cfg: dict | None = None) -> list[str]:
    return list((cfg or load_config())["families"])


# --------------------------------------------------------------------------- dane źródłowe

def read_issues(text_or_path) -> pd.DataFrame:
    src = io.StringIO(text_or_path) if isinstance(text_or_path, str) and "\n" in text_or_path else text_or_path
    e = pd.read_csv(src, encoding="utf-8-sig", dtype={"lista_emisyjna": str})
    for c in ("sprzedaz_od", "sprzedaz_do"):
        e[c] = pd.to_datetime(e[c])
    e["kapitalizacja"] = e["kapitalizacja"].astype(str).str.lower().eq("true")
    return e.sort_values(["rodzina", "sprzedaz_od"]).reset_index(drop=True)


def read_rates(text_or_path) -> pd.DataFrame:
    src = io.StringIO(text_or_path) if isinstance(text_or_path, str) and "\n" in text_or_path else text_or_path
    return pd.read_csv(src, encoding="utf-8-sig")


def fetch_sources(cfg: dict | None = None, offline: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pobranie ``emisje.csv`` i ``oprocentowanie.csv`` (zapis kopii lokalnej); bez sieci – kopia lokalna."""
    from pf.data.http import get_text

    cfg = cfg or load_config()
    local = ROOT / cfg["source"]["local_dir"]
    local.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in cfg["source"]["urls"].items():
        path = local / f"{name}.csv"
        if not offline:
            try:
                text = get_text(url, retries=3)
                path.write_text(text.lstrip("﻿"), encoding="utf-8-sig")
            except Exception as exc:  # noqa: BLE001 – kopia lokalna jako zapas
                log.warning("Obligacje: %s niedostępne (%s) – używam kopii lokalnej", url, exc)
        if not path.exists():
            raise FileNotFoundError(f"Brak danych obligacji: {path}")
        out[name] = path
    return read_issues(out["emisje"]), read_rates(out["oprocentowanie"])


# --------------------------------------------------------------------------- seria i jej wartość

@dataclass
class Holding:
    """Seria obligacji kupiona w dniu ``purchase``: granice okresów, stopy, ułamki roku, opłata."""
    series: str
    purchase: pd.Timestamp
    bounds: list[pd.Timestamp]       # b_0 = zakup, …, b_K = wykup
    rates: np.ndarray                # r_k [ułamek p.a.]
    tau: np.ndarray                  # τ_k [lata]
    capitalized: bool
    fee: float                       # opłata o [zł/obligację]
    no_interest_on_early: bool       # OTS: wcześniejszy wykup bez odsetek
    exchange_price: float            # cena zamiany [zł] (NaN – brak)

    @property
    def maturity(self) -> pd.Timestamp:
        return self.bounds[-1]

    def unit_value(self, d: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """(u(d), a(d)): wartość obligacji i odsetki narosłe podlegające opłacie (d ∈ [zakup, wykup])."""
        b = pd.DatetimeIndex(self.bounds)
        K = len(self.rates)
        k = np.clip(b.searchsorted(d, side="right") - 1, 0, K - 1)
        span = (b[1:] - b[:-1]).days.to_numpy(dtype=float)
        frac = np.clip((d - b[k]).days.to_numpy(dtype=float) / span[k], 0.0, 1.0)
        growth = 1.0 + self.rates * self.tau
        A = np.concatenate([[1.0], np.cumprod(growth)])
        coupon = self.rates[k] * self.tau[k] * frac
        u = 100.0 * A[k] * (1.0 + coupon)
        acc = u - 100.0 if (self.capitalized or self.no_interest_on_early) else 100.0 * coupon
        return u, acc

    def redemption_cost(self, d: pd.DatetimeIndex) -> np.ndarray:
        """f(d) = min(o, a) / (100 + a); OTS: a / (100 + a)."""
        _, acc = self.unit_value(d)
        fee = acc if self.no_interest_on_early else np.minimum(self.fee, acc)
        return np.maximum(fee, 0.0) / (100.0 + acc)


def series_rates(rates: pd.DataFrame, row: pd.Series, n_periods: int) -> np.ndarray:
    """Stopy r_1, …, r_K serii [ułamek]; brakujące okresy – stopa poprzedniego okresu."""
    g = rates[rates["seria"] == row["seria"]]
    if len(g) and (g["jednostka_okresu"] == CONSTANT_RATE).all():
        return np.full(n_periods, float(g["oprocentowanie_pct"].iloc[0]) / 100.0)
    by_k = dict(zip(g["okres"].astype(int), g["oprocentowanie_pct"].astype(float)))
    out, last = np.empty(n_periods), float(row["oprocentowanie_1_okres_pct"])
    for k in range(1, n_periods + 1):
        last = by_k.get(k, last)
        out[k - 1] = last / 100.0
    return out


def make_holding(row: pd.Series, rates: pd.DataFrame, purchase: pd.Timestamp) -> Holding:
    term = int(row["miesiace_do_wykupu"])
    m = PERIOD_MONTHS.get(row["okres_odsetkowy"], term)
    K = max(term // m, 1)
    bounds = [pd.Timestamp(purchase) + pd.DateOffset(months=k * m) for k in range(K + 1)]
    fee = row.get("oplata_wykup_zl")
    return Holding(series=row["seria"], purchase=pd.Timestamp(purchase), bounds=bounds,
                   rates=series_rates(rates, row, K), tau=np.full(K, m / 12.0),
                   capitalized=bool(row["kapitalizacja"]), fee=float(fee) if pd.notna(fee) else 0.0,
                   no_interest_on_early=row.get("oplata_wykup_rodzaj") == NO_INTEREST,
                   exchange_price=float(row["cena_zamiany_zl"]) if pd.notna(row.get("cena_zamiany_zl")) else np.nan)


def series_on_sale(issues: pd.DataFrame, family: str, d: pd.Timestamp,
                   max_gap: int = MAX_GAP_DAYS) -> tuple[pd.Series, pd.Timestamp] | None:
    """Seria rodziny w sprzedaży w dniu d (lub pierwsza, której sprzedaż zaczyna się ≤ ``max_gap`` dni
    później) i dzień zakupu; None – brak serii."""
    f = issues[issues["rodzina"] == family]
    now = f[(f["sprzedaz_od"] <= d) & (f["sprzedaz_do"] >= d)]
    if len(now):
        return now.iloc[0], pd.Timestamp(d)
    later = f[(f["sprzedaz_od"] > d) & (f["sprzedaz_od"] <= d + pd.Timedelta(days=max_gap))]
    if len(later):
        row = later.iloc[0]
        return row, pd.Timestamp(row["sprzedaz_od"])
    return None


def chain(issues: pd.DataFrame, rates: pd.DataFrame, family: str, start, end,
          use_exchange_price: bool = True) -> tuple[pd.Series, pd.Series, list[Holding]]:
    """Indeks wartości V(d) i opłata f(d) (kalendarz dzienny) rolowanej pozycji w rodzinie ``family``."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = pd.date_range(start, end, freq="D")
    V = pd.Series(np.nan, index=days, name=family)
    F = pd.Series(np.nan, index=days, name=family)
    f = issues[issues["rodzina"] == family]
    first = f[f["sprzedaz_do"] >= start]
    if first.empty:
        return V, F, []
    t = max(start, pd.Timestamp(first["sprzedaz_od"].iloc[0]))
    level, holdings, initial = 100.0, [], True
    while t <= end:
        found = series_on_sale(issues, family, t)
        if found is None:
            break
        row, p = found
        if p > end:
            break
        if p > t:                                  # oczekiwanie na sprzedaż: środki bez oprocentowania
            gap = days[(days >= t) & (days < p)]
            V[gap], F[gap] = level, 0.0
        h = make_holding(row, rates, p)
        price = h.exchange_price if (use_exchange_price and not initial and np.isfinite(h.exchange_price)) else 100.0
        units = level / price
        seg = days[(days >= p) & (days <= min(h.maturity, end))]
        u, _ = h.unit_value(seg)
        V[seg] = units * u
        F[seg] = h.redemption_cost(seg)
        holdings.append(h)
        level = units * h.unit_value(pd.DatetimeIndex([h.maturity]))[0][0]
        if h.maturity in V.index:
            F[h.maturity] = 0.0                    # wykup w terminie – bez opłaty
        t, initial = h.maturity, False
    return V, F, holdings


def build(issues: pd.DataFrame, rates: pd.DataFrame, cfg: dict | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(indeks V, opłata f, przegląd posiadanych serii) dla wszystkich tickerów z konfiguracji."""
    cfg = cfg or load_config()
    V, F, rows = {}, {}, []
    for ticker, fam in cfg["families"].items():
        v, f, hs = chain(issues, rates, fam, cfg["start"], cfg["end"], cfg.get("use_exchange_price", True))
        V[ticker], F[ticker] = v, f
        rows += [{"ticker": ticker, "series": h.series, "purchase": h.purchase.date(), "maturity": h.maturity.date(),
                  "rates_pct": " ".join(f"{100 * r:g}" for r in h.rates), "fee_zl": h.fee} for h in hs]
    idx = pd.DataFrame(V)
    idx.index.name = "Date"
    cost = pd.DataFrame(F)
    cost.index.name = "Date"
    return idx, cost, pd.DataFrame(rows)


# --------------------------------------------------------------------------- stopy na sesjach

def log_returns_on(index: pd.DataFrame, sessions: pd.DatetimeIndex,
                   ffill_days: int = SESSION_FFILL_DAYS) -> pd.DataFrame:
    """Log-stopy indeksów na sesjach panelu: r_t = ln V(t) − ln V(t′) (t′ – poprzednia sesja lub dzień
    pierwszego zakupu). Po wykupie bez rolowania wartość jest przenoszona ≤ ``ffill_days`` dni."""
    sessions = pd.DatetimeIndex(sessions)
    out = {}
    for c in index.columns:
        v = index[c].ffill(limit=ffill_days).dropna()
        if v.empty:
            out[c] = pd.Series(np.nan, index=sessions)
            continue
        grid = sessions[(sessions >= v.index[0]) & (sessions <= v.index[-1])].union(v.index[:1])
        lv = np.log(v.reindex(grid))
        out[c] = lv.diff().reindex(sessions)
    return pd.DataFrame(out, index=sessions)


def cost_on(cost: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Opłata f(t) na sesjach panelu (dni bez pozycji – NaN)."""
    return cost.reindex(pd.DatetimeIndex(sessions))


def load_index(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    idx = pd.read_csv(ROOT / cfg["index_path"], index_col=0, parse_dates=True)
    cost = pd.read_csv(ROOT / cfg["redemption_cost_path"], index_col=0, parse_dates=True)
    return idx, cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--offline", action="store_true", help="tylko kopia lokalna danych źródłowych")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    issues, rates = fetch_sources(cfg, offline=args.offline)
    idx, cost, holdings = build(issues, rates, cfg)
    for path, df in ((cfg["index_path"], idx), (cfg["redemption_cost_path"], cost)):
        (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ROOT / path, float_format="%.10g")
    holdings.to_csv(ROOT / cfg["source"]["local_dir"] / "holdings.csv", index=False)
    for c in idx.columns:
        s = idx[c].dropna()
        if len(s):
            years = (s.index[-1] - s.index[0]).days / 365.25
            log.info("%s: %s – %s, %d serii, %.2f%% p.a.", c, s.index[0].date(), s.index[-1].date(),
                     int((holdings["ticker"] == c).sum()), 100 * ((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1))
    log.info("Zapisano %s i %s", cfg["index_path"], cfg["redemption_cost_path"])


if __name__ == "__main__":
    main()
