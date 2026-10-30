"""Testy warstwy danych bez dostępu do sieci (parsery, walidacja, zapis, stronicowanie)."""

import json

import numpy as np
import pandas as pd
import pytest

from pf import artifacts
from pf.data import dolthub_options, options_snapshot, vol_indices
from pf.data.http import RateLimiter, retry

CBOE_OHLC = """DATE,OPEN,HIGH,LOW,CLOSE
01/02/2024,13.21,14.23,13.10,13.20
01/03/2024,13.38,14.22,13.36,14.04
01/03/2024,13.38,14.22,13.36,14.05
01/04/2024,14.00,14.20,13.60,-1
01/05/2024,13.80,14.00,13.00,
01/09/2024,12.90,13.10,12.70,12.76
"""
CBOE_SINGLE = """DATE,VVIX
01/02/2024,85.1
01/03/2024,90.2
"""


def test_parse_cboe_ohlc_and_validate():
    s = vol_indices.parse_cboe_csv(CBOE_OHLC, "VIX")
    clean, rep = vol_indices.validate_series(s)
    assert rep["n_duplicates"] == 1 and rep["n_nonpositive"] == 1 and rep["n_missing"] == 1
    assert clean.loc["2024-01-03"] == 14.05  # zostaje ostatnia obserwacja
    assert list(clean.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-09"]
    assert rep["max_gap_bdays"] == 4
    assert (clean > 0).all()


def test_parse_cboe_single_column():
    s = vol_indices.parse_cboe_csv(CBOE_SINGLE, "VVIX")
    assert s.name == "VVIX" and s.iloc[-1] == 90.2


def test_parse_cboe_rejects_unknown_format():
    with pytest.raises(ValueError):
        vol_indices.parse_cboe_csv("foo,bar\n1,2\n", "VIX")


def yf_chain(n=3):
    return pd.DataFrame({
        "contractSymbol": [f"SPY240419P00{480 + i}000" for i in range(n)],
        "lastTradeDate": pd.to_datetime(["2024-03-01 20:59:00"] * n, utc=True),
        "strike": [480.0 + i for i in range(n)], "lastPrice": 5.0, "bid": 4.9, "ask": 5.1,
        "change": 0.0, "volume": [10, None, 3][:n], "openInterest": [100, 50, 0][:n],
        "impliedVolatility": 0.2, "inTheMoney": False,
    })


def test_normalize_chain_schema():
    ts = pd.Timestamp("2024-03-01 21:30", tz="UTC")
    out = options_snapshot.normalize_chain(yf_chain(), yf_chain(2), "SPY", "2024-04-19", 512.3, ts)
    assert list(out.columns) == options_snapshot.OUTPUT_COLUMNS
    assert (out["call_put"] == "C").sum() == 3 and (out["call_put"] == "P").sum() == 2
    assert out["volume"].isna().sum() == 2  # po jednym braku w tabeli call i put
    assert (out["underlying_price"] == 512.3).all()
    assert options_snapshot.normalize_chain(None, pd.DataFrame(), "SPY", "2024-04-19", 1, ts).empty


def test_session_date_new_york():
    # 02:00 UTC 2 marca = 21:00 1 marca w Nowym Jorku
    assert options_snapshot.session_date(pd.Timestamp("2024-03-02 02:00", tz="UTC")) == "2024-03-01"


def test_append_partition(tmp_path):
    ts1, ts2 = pd.Timestamp("2024-03-01 21:30", tz="UTC"), pd.Timestamp("2024-03-01 22:30", tz="UTC")
    a = options_snapshot.normalize_chain(yf_chain(), None, "SPY", "2024-04-19", 512.3, ts1)
    b = options_snapshot.normalize_chain(yf_chain(), None, "SPY", "2024-04-19", 512.0, ts2)
    path, n = options_snapshot.append_partition(a, "SPY", tmp_path)
    assert path == tmp_path / "date=2024-03-01" / "SPY.parquet" and n == 3
    _, n = options_snapshot.append_partition(a, "SPY", tmp_path)  # powtórzenie – bez duplikatów
    assert n == 3
    _, n = options_snapshot.append_partition(b, "SPY", tmp_path)
    assert n == 6


class FakeDolt:
    """Zastępuje DoltHubClient: wykonuje „SQL” na ramce w pamięci, z limitem wierszy strony."""

    def __init__(self, table: pd.DataFrame, page: int):
        self.table, self.page, self.calls = table, page, 0

    def query(self, sql: str) -> pd.DataFrame:
        self.calls += 1
        if sql.startswith("DESCRIBE"):
            return pd.DataFrame({"Field": self.table.columns})
        offset = int(sql.rsplit("OFFSET", 1)[1])
        limit = int(sql.split("LIMIT", 1)[1].split()[0])
        t = self.table
        start = sql.split("`date` >= '")[1][:10] if "`date` >= '" in sql else None
        if "DISTINCT" in sql:
            days = sorted(t.loc[t["date"] >= start, "date"].unique())
            return pd.DataFrame({"date": days[offset:offset + limit]})
        day = sql.split("`date` = '")[1][:10]
        rows = t[t["date"] == day].sort_values(["expiration", "strike", "call_put"])
        return rows.iloc[offset:offset + limit].reset_index(drop=True)


def dolt_table(days):
    rows = []
    for d in days:
        for k in range(5):
            for cp in ("Call", "Put"):
                rows.append({"date": d, "act_symbol": "SPY", "expiration": "2024-04-19", "strike": str(480 + k),
                             "call_put": cp, "bid": "1.0", "ask": "1.2", "vol": "0.2"})
    return pd.DataFrame(rows)


def test_dolthub_sync_paginates_and_is_incremental(tmp_path, monkeypatch):
    monkeypatch.setattr(dolthub_options, "PAGE_SIZE", 3)
    cfg = {"table": "option_chain", "start": "2024-01-01"}
    fake = FakeDolt(dolt_table(["2024-03-01", "2024-03-04", "2024-03-05", "2024-03-06"]), page=3)
    assert "strike" in dolthub_options.check_schema(fake, "option_chain", ["date", "strike"])
    res = dolthub_options.sync_symbol(fake, cfg, "SPY", tmp_path)
    assert res["n_rows"] == 40 and res["last_date"] == "2024-03-06" and res["status"] == "stale"
    df = pd.read_parquet(tmp_path / "SPY.parquet")
    assert set(df["call_put"]) == {"C", "P"} and df["strike"].dtype == float
    fake.table = dolt_table(["2024-03-01", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"])
    res = dolthub_options.sync_symbol(fake, cfg, "SPY", tmp_path)
    assert res["n_rows"] == 50


def test_dolthub_schema_error():
    fake = FakeDolt(pd.DataFrame(columns=["date", "strike"]), page=10)
    with pytest.raises(dolthub_options.SchemaError):
        dolthub_options.check_schema(fake, "option_chain", ["date", "strike", "bid"])


def test_dolthub_api_response():
    payload = {"query_execution_status": "Success", "schema": [{"columnName": "date"}, {"columnName": "bid"}],
               "rows": [{"date": "2024-03-01", "bid": "1.0"}]}
    assert dolthub_options.parse_api_response(payload).shape == (1, 2)
    with pytest.raises(RuntimeError):
        dolthub_options.parse_api_response({"query_execution_status": "Error", "query_execution_message": "x"})


def test_rate_limiter_and_retry():
    t, slept = [0.0], []
    clock = lambda: t[0]
    sleep = lambda s: slept.append(s)
    lim = RateLimiter(1.5, clock=clock, sleep=sleep)
    lim.wait(); t[0] = 0.5; lim.wait(); t[0] = 5.0; lim.wait()
    assert slept == [1.0]

    calls, delays = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("x")
        return "ok"

    assert retry(flaky, retries=4, backoff=2.0, sleep=delays.append) == "ok"
    assert delays == [2.0, 4.0]
    with pytest.raises(ConnectionError):
        retry(lambda: (_ for _ in ()).throw(ConnectionError("y")), retries=2, sleep=lambda s: None)


def test_manifest_entry(tmp_path):
    df = pd.DataFrame({"x": np.arange(3)})
    path = artifacts.write_parquet(df, tmp_path / "market" / "x.parquet")
    entry = artifacts.update_manifest("market/x", path, params={"seed": 1}, n_rows=3, root=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["market/x"]["path"] == "market/x.parquet"
    assert manifest["market/x"]["sha256"] == artifacts.sha256(path) == entry["sha256"]
    assert manifest["market/x"]["params"]["seed"] == 1 and "numpy" in manifest["market/x"]["versions"]
