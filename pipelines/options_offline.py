"""Pełny pipeline offline modułu opcji (Etapy 1–2) jedną komendą.

    python -m pipelines.options_offline [--skip-dolthub] [--skip-snapshot]

Kolejność: indeksy zmienności → instrumenty bazowe → zrzut łańcuchów opcji → DoltHub →
indeksy strategii Cboe → kalibracja powierzchni + raport OOS → replikacja PPUT.
"""

from __future__ import annotations

import argparse
import sys

from pf.data import cboe_strategy_indices, dolthub_options, options_snapshot, underlyings, vol_indices
from pipelines import calibrate_surface, validate_pput


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-dolthub", action="store_true")
    parser.add_argument("--skip-snapshot", action="store_true")
    args = parser.parse_args()
    sys.argv = sys.argv[:1]  # moduły składowe parsują własne argumenty

    steps = [vol_indices.main, underlyings.main]
    if not args.skip_snapshot:
        steps.append(options_snapshot.main)
    if not args.skip_dolthub:
        steps.append(dolthub_options.main)
    steps += [cboe_strategy_indices.main, calibrate_surface.main, validate_pput.main]
    for step in steps:
        step()


if __name__ == "__main__":
    main()
