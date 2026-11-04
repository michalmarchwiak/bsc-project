"""Stopa wolna od ryzyka w USD: rentowność 13-tygodniowych bonów skarbowych USA.

Źródła, w kolejności pierwszeństwa:
    * ^IRX (Yahoo Finance) z artefaktu ``artifacts/market/underlyings.parquet`` (``pf.data.underlyings``);
    * FRED DTB3 (3-Month Treasury Bill Secondary Market Rate, Discount Basis) – plik CSV
      ``fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3`` – uzupełnia braki ^IRX.

Obie wielkości to stopa dyskontowa w % p.a. (konwencja ACT/360). Stopa ciągła na okres opcji
(``continuous_rate``): r = ln(1 + y·d/basis) / τ, gdzie d – liczba dni kalendarzowych okresu,
τ – czas do wygaśnięcia w latach (u nas: liczba sesji / 252). Czynnik dyskontowy e^{−rτ} równa się
wtedy dokładnie 1 / (1 + y·d/basis), tj. dyskontowaniu stopą prostą rynku pieniężnego.

Wyjście: ``artifacts/market/usd_rate.parquet`` (Date × {rate, source}; rate jako ułamek, np. 0.0525).

Uruchomienie:
    python -m pf.data.usd_rate
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd

from pf import ARTIFACTS_DIR
from pf.artifacts import update_manifest, write_parquet
from pf.data.http import get_text

log = logging.getLogger(__name__)

OUT_PATH = ARTIFACTS_DIR / "market" / "usd_rate.parquet"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def parse_fred_csv(text: str, series: str = "DTB3") -> pd.Series:
    """Plik CSV FRED (observation_date/DATE, <series>) → szereg [% p.a.]; braki ('.', puste) usuwane."""
    df = pd.read_csv(io.StringIO(text))
    date_col = [c for c in df.columns if c.lower() in ("observation_date", "date")][0]
    s = pd.Series(pd.to_numeric(df[series], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df[date_col]), name=series).dropna()
    s.index.name = "Date"
    return s.sort_index()


def combine(irx_pct: pd.Series | None, fred_pct: pd.Series | None) -> pd.DataFrame:
    """Szereg dzienny (ułamek): ^IRX tam, gdzie dostępny, w pozostałych dniach DTB3."""
    parts = []
    if irx_pct is not None and len(irx_pct):
        parts.append(pd.DataFrame({"rate": irx_pct.dropna() / 100.0, "source": "irx"}))
    if fred_pct is not None and len(fred_pct):
        f = fred_pct.dropna() / 100.0
        if parts:
            f = f[~f.index.isin(parts[0].index)]
        parts.append(pd.DataFrame({"rate": f, "source": "fred_dtb3"}))
    if not parts:
        return pd.DataFrame(columns=["rate", "source"])
    out = pd.concat(parts).sort_index()
    out.index.name = "Date"
    return out[~out.index.duplicated(keep="first")]


def continuous_rate(y, days, tau, basis: float = 360.0) -> np.ndarray:
    """r = ln(1 + y·d/basis)/τ – stopa ciągła o czynniku dyskontowym 1/(1 + y·d/basis) na okresie τ."""
    y, days, tau = (np.asarray(x, dtype=float) for x in (y, days, tau))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tau > 0, np.log1p(y * days / basis) / tau, y)


def load(path=OUT_PATH) -> pd.Series:
    """Szereg stopy USD (ułamek) z artefaktu."""
    return pd.read_parquet(path)["rate"].astype(float)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    irx = fred = None
    report = {}
    und_path = ARTIFACTS_DIR / "market" / "underlyings.parquet"
    if und_path.exists():
        und = pd.read_parquet(und_path)
        if ("IRX", "Close") in und.columns:
            irx = und[("IRX", "Close")].dropna()
            report["irx"] = {"n_obs": int(len(irx)), "first": str(irx.index.min().date())}
    try:
        fred = parse_fred_csv(get_text(FRED_URL, params={"id": "DTB3"}))
        report["fred_dtb3"] = {"n_obs": int(len(fred)), "first": str(fred.index.min().date())}
    except Exception as exc:  # noqa: BLE001
        log.warning("FRED DTB3 niedostępne: %s", exc)
        report["fred_dtb3"] = {"error": str(exc)}
    out = combine(irx, fred)
    if out.empty:
        raise SystemExit("Brak stopy USD: ani ^IRX (pf.data.underlyings), ani FRED DTB3")
    out = out.loc["2000-01-01":]
    write_parquet(out, OUT_PATH)
    update_manifest("market/usd_rate", OUT_PATH, n_rows=len(out),
                    params={"sources": report, "convention": "stopa dyskontowa ACT/360, ułamek",
                            "counts": out["source"].value_counts().to_dict()})
    log.info("Zapisano %s (%d dni)", OUT_PATH, len(out))


if __name__ == "__main__":
    main()
