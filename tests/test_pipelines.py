"""Test dymny pipeline'ów kalibracji i walidacji PPUT na syntetycznych artefaktach (bez sieci)."""

import json

import numpy as np
import pandas as pd

from pf import artifacts
from pf.options import pput
from pipelines import calibrate_surface, validate_pput
from tests.test_surface import TRUE, make_chain, vol_frame


def build_artifacts(root):
    dates = pd.bdate_range("2022-11-01", "2023-02-28")
    vol = vol_frame(dates)
    chain = make_chain(TRUE, dates[::3], vol)
    spot = chain.groupby("quote_date")["underlying_price"].first().reindex(dates).interpolate().bfill()
    und = pd.concat({("SPY", "Close"): spot, ("SPX", "Close"): spot * 10, ("SPXTR", "Close"): spot * 20,
                     ("IRX", "Close"): pd.Series(100 * np.expm1(0.045), index=dates)}, axis=1)
    market = root / "market"
    (market / "dolthub_options").mkdir(parents=True)
    vol.to_parquet(market / "vol_indices.parquet")
    und.to_parquet(market / "underlyings.parquet")
    chain.rename(columns={"quote_date": "date"}).drop(columns="underlying_price") \
        .to_parquet(market / "dolthub_options" / "SPY.parquet")
    zero = pd.Series(0.0, index=dates)
    official = pput.replicate(spot * 10, spot * 20, vol, TRUE, zero + 0.044, zero)
    official.rename("PPUT").to_frame().to_parquet(market / "cboe_strategy_indices.parquet")


def test_calibrate_and_validate_pipelines(tmp_path, monkeypatch):
    build_artifacts(tmp_path)
    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path)
    for mod in (calibrate_surface, validate_pput):
        monkeypatch.setattr(mod, "MARKET", tmp_path / "market")
        monkeypatch.setattr(mod, "REPORT_DIR", tmp_path / "reports")
        monkeypatch.setattr(mod, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(calibrate_surface, "PARAMS_PATH", tmp_path / "surface" / "skew_params.json")
    monkeypatch.setattr(calibrate_surface, "SNAPSHOT_DIR", tmp_path / "raw" / "options_snapshots")

    calibrate_surface.main()
    params = json.loads((tmp_path / "surface" / "skew_params.json").read_text())
    np.testing.assert_allclose([params[k] for k in "abcd"], [TRUE.a, TRUE.b, TRUE.c, TRUE.d], atol=1e-4)
    report = pd.read_csv(tmp_path / "reports" / "surface_oos.csv")
    assert set(report["bucket"]) == {"ATM", "OTM5", "OTM10"}
    assert (tmp_path / "reports" / "surface_oos.md").exists()

    validate_pput.main()
    te = pd.read_csv(tmp_path / "reports" / "pput_tracking.csv")
    assert te.loc[te["period"] == "all", "te_ann_pct"].iloc[0] < 1.0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["reports/pput_tracking"]["params"]["surface_version"] == params["version"]
