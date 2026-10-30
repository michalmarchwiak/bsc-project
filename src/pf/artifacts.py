"""Zapis artefaktów (Parquet) oraz rejestr ``artifacts/manifest.json``.

Każdy wpis w manifeście zawiera: ścieżkę względną, hash SHA-256 pliku, liczbę wierszy,
znacznik czasu, wersje pakietów oraz dowolne parametry (np. ziarno RNG, parametry
powierzchni zmienności, status walidacji). Pozwala to odtworzyć i zweryfikować artefakty.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from pf import ARTIFACTS_DIR

PACKAGES = ["numpy", "pandas", "scipy", "pyarrow", "yfinance", "requests"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")
    return path


def update_manifest(key: str, path: Path | None = None, params: dict | None = None,
                    n_rows: int | None = None, root: Path | None = None) -> dict:
    """Dopisuje (nadpisuje) wpis ``key`` w ``<root>/manifest.json`` (domyślnie ``artifacts/``) i zwraca go."""
    root = root or ARTIFACTS_DIR
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    entry: dict = {
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": package_versions(),
    }
    if path is not None:
        entry["path"] = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        entry["sha256"] = sha256(path) if path.is_file() else None
    if n_rows is not None:
        entry["n_rows"] = int(n_rows)
    if params:
        entry["params"] = params
    manifest[key] = entry
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
                             encoding="utf-8")
    return entry
