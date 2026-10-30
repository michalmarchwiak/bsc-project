"""Wczytywanie plików konfiguracyjnych YAML z katalogu ``config/``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from pf import CONFIG_DIR


@lru_cache(maxsize=None)
def load(name: str, config_dir: Path = CONFIG_DIR) -> dict:
    with open(config_dir / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
