"""Skrypty uruchomieniowe (``python -m pipelines.<nazwa>``).

Pakiet ``pf`` leży w ``src/``. Jeśli nie został zainstalowany (``pip install -e .``), katalog ``src/`` jest
dopisywany do ``sys.path``, więc pipeline'y działają także bezpośrednio z katalogu repozytorium.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
try:
    import pf  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(_SRC))
