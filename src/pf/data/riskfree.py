"""Stopa wolna od ryzyka w PLN: WIBOR 1M (Stooq), zapasowo stopa referencyjna NBP.

Źródła (``config/backtest.yaml``, sekcja ``riskfree``):
    * Stooq – WIBOR 1M, symbol ``plopln1m`` (% p.a., fixing dzienny), plik CSV
      ``stooq.pl/q/d/l/?s=plopln1m&i=d`` (nagłówki polskie) lub ``stooq.com`` (angielskie);
    * NBP – archiwum stóp procentowych (XML): stopa referencyjna obowiązująca od daty
      wskazanej w atrybucie ``obowiazuje_od`` (funkcja schodkowa, prawostronnie ciągła).

Szereg dzienny r(t) na dniach roboczych [start, end]:
    r(t) = WIBOR1M(τ(t)),  τ(t) = ostatni fixing ≤ t, o ile t − τ(t) ≤ ``max_ffill_days`` dni;
    w pozostałych dniach (WIBOR kończy się przed końcem próby, zaczyna po jej początku lub ma
    dłuższą lukę) r(t) = stopa referencyjna NBP obowiązująca w dniu t.
Kolumna ``source`` wskazuje pochodzenie wartości, a manifest – liczbę dni uzupełnionych stopą NBP.

Stopa okresu (konwencja ACT/365, oprocentowanie proste):
    r_f(t0, t1) = WIBOR1M(t0) · (t1 − t0)[dni] / 365.

Wyjście: ``artifacts/market/riskfree_pln.parquet`` (Date × {rate, wibor1m, nbp_ref, source}; stopy jako
ułamki, np. 0.0575).

Uruchomienie:
    python -m pf.data.riskfree
"""

from __future__ import annotations

import io
import logging
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from pf import ARTIFACTS_DIR, config
from pf.artifacts import update_manifest, write_parquet
from pf.data.http import get_text

log = logging.getLogger(__name__)

OUT_PATH = ARTIFACTS_DIR / "market" / "riskfree_pln.parquet"

STOOQ_COLUMNS = {"Data": "Date", "Otwarcie": "Open", "Najwyzszy": "High", "Najnizszy": "Low",
                 "Zamkniecie": "Close", "Wolumen": "Volume"}


# --------------------------------------------------------------------------- parsery

def parse_stooq_csv(text: str, symbol: str = "") -> pd.Series:
    """Plik CSV Stooq (nagłówki PL lub EN) → szereg zamknięć [% p.a.] indeksowany datą."""
    text = text.strip()
    first = text.splitlines()[0] if text else ""
    if "Date" not in first and "Data" not in first:
        raise ValueError(f"Stooq {symbol}: nieoczekiwana odpowiedź ({first[:80]!r})")
    df = pd.read_csv(io.StringIO(text)).rename(columns=STOOQ_COLUMNS)
    s = pd.Series(pd.to_numeric(df["Close"], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df["Date"], errors="coerce"), name=symbol or "Close")
    s = s[s.index.notna()].dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.index.name = "Date"
    if s.empty:
        raise ValueError(f"Stooq {symbol}: brak obserwacji")
    return s


def _pl_float(x: str) -> float:
    return float(str(x).replace(" ", "").replace(" ", "").replace(",", "."))


def parse_nbp_rates_xml(text: str, rate_id: str = "ref") -> pd.Series:
    """Archiwum stóp NBP (XML) → szereg schodkowy stopy ``rate_id`` [% p.a.] (data obowiązywania → stopa).

    Obsługiwane są oba układy plików NBP: atrybut ``obowiazuje_od`` w elemencie nadrzędnym
    (``<pozycje obowiazuje_od=…><pozycja id="ref" oprocentowanie="5,75"/></pozycje>``) albo
    w samym elemencie ``pozycja``.
    """
    root = ET.fromstring(text)
    parent = {child: node for node in root.iter() for child in node}
    rows = {}
    for el in root.iter():
        if el.get("id") != rate_id or el.get("oprocentowanie") is None:
            continue
        node, date = el, el.get("obowiazuje_od")
        while date is None and node in parent:
            node = parent[node]
            date = node.get("obowiazuje_od")
        if date is None:
            continue
        rows[pd.Timestamp(date)] = _pl_float(el.get("oprocentowanie"))
    if not rows:
        raise ValueError(f"Brak stopy '{rate_id}' w pliku NBP")
    s = pd.Series(rows, name=f"nbp_{rate_id}").sort_index()
    s.index.name = "Date"
    return s


# --------------------------------------------------------------------------- pobieranie

def fetch_wibor(cfg: dict) -> tuple[pd.Series, str]:
    """WIBOR 1M ze Stooq (kolejne adresy z konfiguracji); zwraca szereg [% p.a.] i użyty adres."""
    w = cfg["wibor"]
    params = {"s": w["symbol"], "d1": cfg["start"].replace("-", ""), "d2": cfg["end"].replace("-", ""), "i": "d"}
    if os.environ.get("STOOQ_APIKEY"):
        params["apikey"] = os.environ["STOOQ_APIKEY"]
    errors = []
    for url in w["urls"]:
        try:
            return parse_stooq_csv(get_text(url, params=params, retries=cfg.get("retries", 3)), w["symbol"]), url
        except Exception as exc:  # noqa: BLE001
            log.warning("WIBOR %s (%s): %s", w["symbol"], url, exc)
            errors.append(f"{url}: {exc}")
    raise RuntimeError("WIBOR niedostępny: " + "; ".join(errors))


def fetch_nbp_reference(cfg: dict) -> tuple[pd.Series, str]:
    n = cfg["nbp"]
    errors = []
    for url in n["urls"]:
        try:
            return parse_nbp_rates_xml(get_text(url, retries=cfg.get("retries", 3)), n["rate_id"]), url
        except Exception as exc:  # noqa: BLE001
            log.warning("NBP (%s): %s", url, exc)
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Stopy NBP niedostępne: " + "; ".join(errors))


# --------------------------------------------------------------------------- szereg dzienny

def combine(wibor: pd.Series | None, nbp: pd.Series | None, start: str, end: str,
            max_ffill_days: int) -> tuple[pd.DataFrame, dict]:
    """Dzienny szereg stopy [ułamek] na dniach roboczych: WIBOR (ffill ≤ max_ffill_days), braki → NBP."""
    days = pd.bdate_range(start, end, name="Date")
    out = pd.DataFrame(index=days)
    out["wibor1m"] = np.nan
    out["nbp_ref"] = np.nan
    if wibor is not None and not wibor.empty:
        w = wibor.sort_index() / 100.0
        last_fix = pd.Series(w.index, index=w.index).reindex(days.union(w.index)).ffill().reindex(days)
        val = w.reindex(days.union(w.index)).ffill().reindex(days)
        fresh = (days.to_series() - last_fix).dt.days <= max_ffill_days
        out["wibor1m"] = val.where(fresh.to_numpy())
    if nbp is not None and not nbp.empty:
        n = nbp.sort_index() / 100.0
        out["nbp_ref"] = n.reindex(days.union(n.index)).ffill().reindex(days)
    use_w = out["wibor1m"].notna()
    out["rate"] = out["wibor1m"].where(use_w, out["nbp_ref"])
    out["source"] = np.where(use_w, "wibor1m", np.where(out["nbp_ref"].notna(), "nbp_ref", "missing"))
    filled = out.index[out["source"] == "nbp_ref"]
    report = {
        "start": str(days.min().date()), "end": str(days.max().date()), "n_days": int(len(days)),
        "wibor_first": str(wibor.index.min().date()) if wibor is not None and len(wibor) else None,
        "wibor_last": str(wibor.index.max().date()) if wibor is not None and len(wibor) else None,
        "n_filled_nbp": int(len(filled)),
        "filled_ranges": _ranges(filled),
        "n_missing": int((out["source"] == "missing").sum()),
    }
    return out, report


def _ranges(idx: pd.DatetimeIndex) -> list[list[str]]:
    """Zwięzły opis zbioru dni roboczych jako listy przedziałów [od, do]."""
    if len(idx) == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for d in idx[1:]:
        if len(pd.bdate_range(prev, d)) > 2:
            out.append([str(start.date()), str(prev.date())])
            start = d
        prev = d
    out.append([str(start.date()), str(prev.date())])
    return out


# --------------------------------------------------------------------------- użycie

def load(path=OUT_PATH) -> pd.Series:
    """Dzienna stopa roczna [ułamek] z artefaktu ``riskfree_pln.parquet``."""
    return pd.read_parquet(path)["rate"]


def rate_at(rate: pd.Series, t0: pd.Timestamp) -> float:
    """Ostatnia znana stopa roczna w dniu t0 (ostatnia obserwacja ≤ t0)."""
    pos = rate.index.searchsorted(pd.Timestamp(t0), side="right") - 1
    if pos < 0:
        raise KeyError(f"Brak stopy wolnej od ryzyka przed {t0}")
    return float(rate.iloc[pos])


def period_rate(rate: pd.Series | float, t0: pd.Timestamp, t1: pd.Timestamp, day_count: int = 365) -> float:
    """r_f(t0, t1) = r(t0) · (t1 − t0)[dni] / ``day_count`` (ACT/365, oprocentowanie proste)."""
    r0 = float(rate) if np.isscalar(rate) else rate_at(rate, t0)
    return r0 * (pd.Timestamp(t1) - pd.Timestamp(t0)).days / day_count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config.load("backtest")["riskfree"]
    sources = {}
    try:
        wibor, sources["wibor_url"] = fetch_wibor(cfg)
    except RuntimeError as exc:
        log.error("%s", exc)
        wibor = None
    nbp = None
    end = pd.Timestamp(cfg["end"])
    needs_nbp = wibor is None or wibor.index.max() < end - pd.Timedelta(days=cfg["max_ffill_days"]) \
        or wibor.index.min() > pd.Timestamp(cfg["start"])
    if needs_nbp or cfg["nbp"].get("always_fetch", True):
        try:
            nbp, sources["nbp_url"] = fetch_nbp_reference(cfg)
        except RuntimeError as exc:
            if needs_nbp:
                raise
            log.warning("%s", exc)
    table, report = combine(wibor, nbp, cfg["start"], cfg["end"], cfg["max_ffill_days"])
    if report["n_missing"]:
        raise RuntimeError(f"Szereg stopy wolnej od ryzyka ma {report['n_missing']} dni bez wartości")
    if report["n_filled_nbp"]:
        log.warning("Dni uzupełnione stopą referencyjną NBP: %d (%s)", report["n_filled_nbp"],
                    report["filled_ranges"])
    write_parquet(table, OUT_PATH)
    update_manifest("market/riskfree_pln", OUT_PATH, n_rows=len(table), params={
        "symbol": cfg["wibor"]["symbol"], "day_count": cfg["day_count"], **sources, **report})
    log.info("Zapisano %s (%d dni, WIBOR do %s)", OUT_PATH, len(table), report["wibor_last"])


if __name__ == "__main__":
    main()
