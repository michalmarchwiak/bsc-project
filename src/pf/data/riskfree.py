"""Stopa wolna od ryzyka w PLN: WIBOR 1M (Stooq), zapasowo średnia miesięczna WIBOR 1M (Eurostat)
i stopa referencyjna NBP.

Źródła (``config/backtest.yaml``, sekcja ``riskfree``), w kolejności pierwszeństwa:
    * Stooq – WIBOR 1M, symbol ``plopln1m`` (% p.a., fixing dzienny), plik CSV
      ``stooq.pl/q/d/l/?s=plopln1m&i=d`` (nagłówki polskie) lub ``stooq.com`` (angielskie);
      opcjonalnie lokalna kopia pliku (``wibor.local_csv``);
    * Eurostat – ``irt_st_m`` (geo = PL, int_rt = IRT_M1): średnia miesięczna fixingów WIBOR 1M;
      wartość miesiąca m przypisana jest wszystkim dniom m;
    * NBP – archiwum stóp procentowych (XML): stopa referencyjna obowiązująca od daty
      wskazanej w atrybucie ``obowiazuje_od`` (funkcja schodkowa, prawostronnie ciągła).

Szereg dzienny r(t) na dniach roboczych [start, end]:
    r(t) = WIBOR1M(τ(t)),  τ(t) = ostatni fixing ≤ t, o ile t − τ(t) ≤ ``max_ffill_days`` dni;
    w pozostałych dniach r(t) = średnia miesięczna WIBOR 1M miesiąca dnia t (Eurostat), a gdy jej
    brak – stopa referencyjna NBP obowiązująca w dniu t.
Kolumna ``source`` wskazuje pochodzenie wartości, a manifest – liczbę i zakresy dni z każdego źródła.

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
from pf.data.http import get_bytes, get_text

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


def parse_eurostat_json(payload: dict) -> pd.Series:
    """Odpowiedź JSON-stat API Eurostatu (jeden szereg) → szereg miesięczny [% p.a.] indeksowany 1. dniem miesiąca."""
    time = payload["dimension"]["time"]["category"]["index"]
    inv = {v: k for k, v in time.items()}
    rows = {pd.Period(inv[int(k)], "M").to_timestamp(): float(v) for k, v in payload["value"].items()}
    s = pd.Series(rows, name="eurostat_m1").sort_index()
    s.index.name = "Date"
    return s


def parse_nbp_rates_xml(text: str | bytes, rate_id: str = "ref") -> pd.Series:
    """Archiwum stóp NBP (XML) → szereg schodkowy stopy ``rate_id`` [% p.a.] (data obowiązywania → stopa).

    Obsługiwane są oba układy plików NBP: atrybut ``obowiazuje_od`` w elemencie nadrzędnym
    (``<pozycje obowiazuje_od=…><pozycja id="ref" oprocentowanie="5,75"/></pozycje>``) albo
    w samym elemencie ``pozycja``.
    """
    if isinstance(text, bytes):
        root = ET.fromstring(text.lstrip(b"\xef\xbb\xbf").strip())   # bajty: kodowanie z deklaracji XML
    else:
        root = ET.fromstring(text.lstrip("\ufeff").strip())
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
    local = w.get("local_csv")
    if local:
        from pf import ROOT

        path = ROOT / local
        if path.exists():
            return parse_stooq_csv(path.read_text(encoding="utf-8"), w["symbol"]), str(path.relative_to(ROOT))
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


def fetch_eurostat(cfg: dict) -> tuple[pd.Series, str]:
    from pf.data.http import get_json

    e = cfg["eurostat"]
    url = e["url"]
    return parse_eurostat_json(get_json(url, params=e["params"], retries=cfg.get("retries", 3))), url


def fetch_nbp_reference(cfg: dict) -> tuple[pd.Series, str]:
    n = cfg["nbp"]
    errors = []
    for url in n["urls"]:
        try:
            return parse_nbp_rates_xml(get_bytes(url, retries=cfg.get("retries", 3)), n["rate_id"]), url
        except Exception as exc:  # noqa: BLE001
            log.warning("NBP (%s): %s", url, exc)
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Stopy NBP niedostępne: " + "; ".join(errors))


# --------------------------------------------------------------------------- szereg dzienny

def combine(wibor: pd.Series | None, nbp: pd.Series | None, start: str, end: str, max_ffill_days: int,
            monthly: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
    """Dzienny szereg stopy [ułamek] na dniach roboczych: WIBOR (ffill ≤ max_ffill_days), braki → średnia
    miesięczna WIBOR 1M (``monthly``, Eurostat), dalej → stopa referencyjna NBP."""
    days = pd.bdate_range(start, end, name="Date")
    out = pd.DataFrame(index=days)
    out["wibor1m"] = np.nan
    out["wibor1m_monthly"] = np.nan
    out["nbp_ref"] = np.nan
    if wibor is not None and not wibor.empty:
        w = wibor.sort_index() / 100.0
        last_fix = pd.Series(w.index, index=w.index).reindex(days.union(w.index)).ffill().reindex(days)
        val = w.reindex(days.union(w.index)).ffill().reindex(days)
        fresh = (days.to_series() - last_fix).dt.days <= max_ffill_days
        out["wibor1m"] = val.where(fresh.to_numpy())
    if monthly is not None and not monthly.empty:
        m = monthly / 100.0
        m.index = m.index.to_period("M")
        out["wibor1m_monthly"] = days.to_period("M").map(lambda p: m.get(p, np.nan)).to_numpy(dtype=float)
    if nbp is not None and not nbp.empty:
        n = nbp.sort_index() / 100.0
        out["nbp_ref"] = n.reindex(days.union(n.index)).ffill().reindex(days)
    use_w = out["wibor1m"].notna()
    use_m = ~use_w & out["wibor1m_monthly"].notna()
    out["rate"] = out["wibor1m"].where(use_w, out["wibor1m_monthly"].where(use_m, out["nbp_ref"]))
    out["source"] = np.where(use_w, "wibor1m", np.where(use_m, "wibor1m_monthly",
                                                        np.where(out["nbp_ref"].notna(), "nbp_ref", "missing")))
    report = {
        "start": str(days.min().date()), "end": str(days.max().date()), "n_days": int(len(days)),
        "wibor_first": str(wibor.index.min().date()) if wibor is not None and len(wibor) else None,
        "wibor_last": str(wibor.index.max().date()) if wibor is not None and len(wibor) else None,
        "n_monthly": int(use_m.sum()),
        "monthly_ranges": _ranges(out.index[out["source"] == "wibor1m_monthly"]),
        "n_filled_nbp": int((out["source"] == "nbp_ref").sum()),
        "filled_ranges": _ranges(out.index[out["source"] == "nbp_ref"]),
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
    sources, errors = {}, {}
    wibor = monthly = nbp = None
    try:
        wibor, sources["wibor_url"] = fetch_wibor(cfg)
    except RuntimeError as exc:
        log.error("%s", exc)
        errors["wibor"] = str(exc)
    if cfg.get("eurostat", {}).get("enabled", False):
        try:
            monthly, sources["eurostat_url"] = fetch_eurostat(cfg)
        except Exception as exc:  # noqa: BLE001
            log.error("Eurostat: %s", exc)
            errors["eurostat"] = str(exc)
    try:
        nbp, sources["nbp_url"] = fetch_nbp_reference(cfg)
    except RuntimeError as exc:
        log.error("%s", exc)
        errors["nbp"] = str(exc)
    table, report = combine(wibor, nbp, cfg["start"], cfg["end"], cfg["max_ffill_days"], monthly)
    if report["n_missing"]:
        raise RuntimeError(f"Szereg stopy wolnej od ryzyka ma {report['n_missing']} dni bez wartości")
    if report["n_monthly"] or report["n_filled_nbp"]:
        log.warning("Dni bez dziennego WIBOR: średnia miesięczna Eurostat %d, stopa referencyjna NBP %d (%s)",
                    report["n_monthly"], report["n_filled_nbp"], report["filled_ranges"])
    write_parquet(table, OUT_PATH)
    update_manifest("market/riskfree_pln", OUT_PATH, n_rows=len(table), params={
        "symbol": cfg["wibor"]["symbol"], "day_count": cfg["day_count"], **sources, **report,
        "errors": errors, "source_counts": table["source"].value_counts().to_dict()})
    log.info("Zapisano %s (%d dni; źródła: %s)", OUT_PATH, len(table), table["source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
