"""Skład sub-funduszy punktowo w czasie: wybór spółek z puli według wartości obrotu (dane sprzed daty decyzji).

Stała lista spółek dobrana z perspektywy końca próby (np. dzisiejsze największe spółki) wprowadza do backtestu
informację z przyszłości. Tutaj sub-fundusz z kluczem ``select`` (``config/subfunds.yaml``) jest pulą kandydatów,
a jego skład w dacie rebalansowania t wyznacza reguła korzystająca wyłącznie z danych sprzed t:

    r(t)  – data rewizji rankingu: pierwsza sesja miesiąca ``review_month`` w roku t (gdy przypada po t – w roku
            poprzednim; gdy przed nią jest mniej niż ``min_obs`` sesji danych – sama data t),
    s_i   – średnia z ostatnich ``lookback_months`` miesięcznych median dziennej wartości obrotu
            Close·Volume (waluta notowania; miesiące zakończone przed r),
    A_t   – spółki puli dopuszczone w oknie estymacji przed t (``pf.backtest.data.eligible_assets``),
    skład – tickery ``core`` ∩ A_t oraz ``top`` spółek z A_t \\ core o największym s_i.

Ranking obrotów zmienia się raz w roku (rewizja roczna, jak w indeksach giełdowych), więc skład jest stały między
rewizjami, o ile nie zmienia się zbiór spółek dopuszczonych (spółka po debiucie wchodzi do rankingu, gdy ma
``min_obs`` obserwacji w oknie). Zmiana składu generuje obrót i koszty transakcyjne w backteście.

Wartość obrotu jest przybliżeniem wielkości i płynności spółki dostępnym w danych notowań; historycznych
kapitalizacji ani składów indeksów w danych nie ma. Pula zawiera tylko spółki notowane do dziś – spółek wycofanych
z obrotu reguła nie odtworzy (pozostałe obciążenie przeżywalnością).

Miesięczne mediany obrotu: ``data/processed/turnover_monthly.csv`` (``data_manipulation.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pf import ROOT, config
from pf.backtest.data import eligible_assets, estimation_window
from pf.portfolio import deterministic_tickers

log = logging.getLogger(__name__)


def selection_config() -> dict:
    return dict(config.load("subfunds").get("selection") or {})


def uses_selection(subfunds: dict) -> bool:
    return any(spec.get("select") for spec in subfunds.values())


def pool_tickers(subfunds: dict) -> list[str]:
    """Wszystkie tickery uniwersum (pełne pule), bez powtórzeń, w kolejności konfiguracji."""
    return list(dict.fromkeys(t for spec in subfunds.values() for t in spec["tickers"]))


def load_turnover(path: Path | str) -> pd.DataFrame:
    """Miesięczne mediany dziennej wartości obrotu (indeks: pierwszy dzień miesiąca, kolumny: tickery)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df.index = pd.DatetimeIndex(df.index).to_period("M")
    return df


def review_date(index: pd.DatetimeIndex, t: pd.Timestamp, review_month: int = 1) -> pd.Timestamp:
    """Ostatnia data rewizji ≤ t: pierwsza sesja miesiąca ``review_month`` (brak wcześniejszej – pierwsza sesja)."""
    t = pd.Timestamp(t)
    past = index[index <= t]
    hits = past[past.month == int(review_month)]
    if len(hits) == 0:
        return index[0]
    return hits[hits.year == hits[-1].year][0]


def turnover_score(turnover: pd.DataFrame, r: pd.Timestamp, lookback_months: int = 12) -> pd.Series:
    """s_i: średnia miesięcznych median obrotu z ostatnich ``lookback_months`` miesięcy zakończonych przed r."""
    hist = turnover.loc[turnover.index < pd.Timestamp(r).to_period("M")].iloc[-int(lookback_months):]
    return hist.mean()


def select_members(spec: dict, eligible: set[str], score: pd.Series) -> list[str]:
    """Skład SF: ``core`` oraz ``top`` dopuszczonych spółek puli o największym s_i (kolejność puli)."""
    sel = spec["select"]
    pool = [t for t in spec["tickers"] if t in eligible]
    core = [t for t in pool if t in set(sel.get("core", []))]
    rest = [t for t in pool if t not in set(core)]
    s = score.reindex(rest).dropna()
    s = s[s > 0]
    chosen = set(s.sort_values(ascending=False, kind="stable").index[: int(sel["top"])])
    return [t for t in pool if t in chosen or t in set(core)]


def plain(subfunds: dict) -> dict:
    """Specyfikacja SF bez klucza ``select`` (skład już wybrany)."""
    return {sf: {k: v for k, v in spec.items() if k != "select"} for sf, spec in subfunds.items()}


class Universe:
    """Skład sub-funduszy w datach rebalansowania (``at``) i zawężenie tabeli stóp do aktywów daty (``restrict``)."""

    def __init__(self, subfunds: dict, returns: pd.DataFrame, rcfg: dict, turnover: pd.DataFrame | None = None,
                 scfg: dict | None = None):
        self.subfunds = subfunds
        self.returns = returns
        self.rcfg = rcfg
        self.scfg = dict(scfg or {})
        self.turnover = turnover
        self.active = uses_selection(subfunds)
        self.pool = set(pool_tickers(subfunds))
        self._cache: dict[pd.Timestamp, tuple[dict, pd.DataFrame]] = {}
        if self.active and turnover is None:
            raise FileNotFoundError("Uniwersum z wyborem składu (select) wymaga tabeli obrotów – uruchom "
                                    "data_import.py i data_manipulation.py (selection.turnover_path)")

    def review(self, t: pd.Timestamp) -> pd.Timestamp:
        """Data rewizji obowiązująca w t; gdy przed nią jest mniej niż ``min_obs`` sesji (początek danych) – samo t."""
        r = review_date(self.returns.index, t, self.scfg.get("review_month", 1))
        return r if self.returns.index.searchsorted(r) >= int(self.rcfg["min_obs"]) else pd.Timestamp(t)

    def _select(self, t: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
        t = pd.Timestamp(t)
        if t in self._cache:
            return self._cache[t]
        r = self.review(t)
        window = estimation_window(self.returns, t, self.rcfg["window_days"])
        eligible = set(eligible_assets(window, self.rcfg["min_obs"], deterministic_tickers(self.subfunds),
                                       self.rcfg.get("min_obs_deterministic")))
        score = turnover_score(self.turnover, r, self.scfg.get("lookback_months", 12))
        out, rows = {}, []
        for sf, spec in self.subfunds.items():
            if not spec.get("select"):
                out[sf] = spec
                continue
            members = select_members(spec, eligible, score)
            n_pool = sum(x in eligible for x in spec["tickers"])
            if n_pool and len(members) < min(n_pool, int(spec["select"]["top"])):
                log.warning("%s, %s (rewizja %s): wybrano %d z %d dopuszczonych spółek (brak danych o obrotach?)",
                            sf, t.date(), r.date(), len(members), n_pool)
            out[sf] = {k: v for k, v in spec.items() if k != "select"} | {"tickers": members}
            core = set(spec["select"].get("core", []))
            ranked = score.reindex([x for x in spec["tickers"] if x in eligible and x not in core])
            rank = ranked.rank(ascending=False, method="first")
            rows += [{"date": t, "review": r, "subfund": sf, "ticker": x, "eligible": x in eligible, "core": x in core,
                      "score": float(score.get(x, float("nan"))),
                      "rank": float(rank.get(x, float("nan"))), "selected": x in set(members)}
                     for x in spec["tickers"]]
        self._cache[t] = (out, pd.DataFrame(rows))
        return self._cache[t]

    def at(self, t: pd.Timestamp) -> dict:
        """Sub-fundusze ze składem obowiązującym w dacie t (bez klucza ``select``)."""
        if not self.active:
            return self.subfunds
        return self._select(t)[0]

    def restrict(self, subfunds_t: dict, returns: pd.DataFrame | None = None) -> pd.DataFrame:
        """Tabela stóp z kolumnami składu daty oraz aktywami spoza pul (np. kurs USD/PLN); kolejność bez zmian."""
        returns = self.returns if returns is None else returns
        if not self.active:
            return returns
        keep = set(pool_tickers(subfunds_t))
        cols = [c for c in returns.columns if c in keep or c not in self.pool]
        return returns[cols]

    def view(self, t: pd.Timestamp, returns: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
        """(tabela stóp zawężona do składu daty t, sub-fundusze ze składem daty t)."""
        sub = self.at(t)
        return self.restrict(sub, returns), sub

    def schedule(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Tabela składu (date, review, subfund, ticker, eligible, core, score, rank, selected) dla dat ``dates``."""
        if not self.active:
            return pd.DataFrame()
        return pd.concat([self._select(t)[1] for t in dates], ignore_index=True)


def build(subfunds: dict, returns: pd.DataFrame, rcfg: dict, root: Path = ROOT, scfg: dict | None = None) -> Universe:
    """``Universe`` z konfiguracji (``selection``) i pliku obrotów; uniwersum bez ``select`` nie wymaga pliku."""
    scfg = selection_config() if scfg is None else scfg
    turnover = None
    if uses_selection(subfunds):
        path = Path(root) / scfg.get("turnover_path", "data/processed/turnover_monthly.csv")
        if path.is_file():
            turnover = load_turnover(path)
    return Universe(subfunds, returns, rcfg, turnover, scfg)


def restrict_to_pool(returns: pd.DataFrame, subfunds: dict, extra=()) -> pd.DataFrame:
    """Kolumny tabeli stóp należące do uniwersum (pełne pule) i ``extra``; kolejność pliku bez zmian."""
    keep = set(pool_tickers(subfunds)) | set(extra)
    return returns[[c for c in returns.columns if c in keep]]
