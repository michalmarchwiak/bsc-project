"""Indeksy strategii opcyjnych Cboe do walidacji wyceny: PPUT (opcjonalnie PUT, BXM).

PPUT (Cboe S&P 500 5% Put Protection Index) – portfel S&P 500 z długą pozycją w miesięcznym
pucie 5% OTM rolowanym w trzeci piątek miesiąca. Szereg służy do replikacji strategii
wyceną syntetyczną (BS + powierzchnia) i pomiaru tracking error.

Źródła: plik historyczny Cboe (jak dla indeksów zmienności), zapasowo Yahoo. Jeżeli żadne
źródło nie zwróci danych, walidacja jest oznaczana w manifeście jako pominięta
(``status: skipped``).

Wyjście: ``artifacts/market/cboe_strategy_indices.parquet`` (Date × symbol).

Uruchomienie:
    python -m pf.data.cboe_strategy_indices
"""

from __future__ import annotations

import logging

import pandas as pd

from pf import ARTIFACTS_DIR, config
from pf.artifacts import update_manifest, write_parquet
from pf.data.vol_indices import fetch_cboe, fetch_yahoo_close, validate_series

log = logging.getLogger(__name__)

OUT_PATH = ARTIFACTS_DIR / "market" / "cboe_strategy_indices.parquet"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config.load("options")["cboe_strategy"]
    series, reports = [], {}
    for symbol in cfg["symbols"]:
        try:
            s, source = fetch_cboe(symbol, cfg["cboe_url"]), "cboe"
        except Exception as exc:  # noqa: BLE001
            log.warning("Cboe %s niedostępne (%s) – próbuję Yahoo", symbol, exc)
            try:
                s, source = fetch_yahoo_close(cfg["yahoo"][symbol], "1986-01-01", symbol), "yahoo"
            except Exception as exc2:  # noqa: BLE001
                reports[symbol] = {"source": "MISSING", "error": str(exc2)}
                continue
        clean, rep = validate_series(s)
        series.append(clean)
        reports[symbol] = {"source": source, **rep}

    if "PPUT" not in {s.name for s in series}:
        log.warning("Brak danych PPUT – walidacja replikacji PPUT zostanie pominięta")
        update_manifest("market/cboe_strategy_indices",
                        params={"status": "skipped", "reason": "PPUT niedostępny", "validation": reports})
        return
    wide = pd.concat(series, axis=1).sort_index()
    write_parquet(wide, OUT_PATH)
    update_manifest("market/cboe_strategy_indices", OUT_PATH, n_rows=len(wide),
                    params={"status": "ok", "validation": reports})


if __name__ == "__main__":
    main()
