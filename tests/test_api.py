"""API tests against a tiny synthetic serving dataset (no model or real data needed).

Series (28 forecast days from 2016-04-25, 28 history days before):
    FOODS_1_001 @ CA_1     forecast 2.0/day, history 1/day, price 2.50 -> 100% shortfall with 28 days of cover
    FOODS_1_001 @ TX_1     forecast 1.0/day, history 1/day             -> exactly covered, not at risk
    HOBBIES_1_001 @ CA_1   forecast 0.5/day, history 0/day             -> no recent sales, not evaluated
"""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app

START = pd.Timestamp("2016-04-25")
SERIES = [  # item, store, dept, cat, state, forecast/day, history/day, price
    ("FOODS_1_001", "CA_1", "FOODS_1", "FOODS", "CA", 2.0, 1, 2.5),
    ("FOODS_1_001", "TX_1", "FOODS_1", "FOODS", "TX", 1.0, 1, 2.5),
    ("HOBBIES_1_001", "CA_1", "HOBBIES_1", "HOBBIES", "CA", 0.5, 0, 9.0),
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("serving")
    fdates = pd.date_range(START, periods=28)
    hdates = pd.date_range(START - pd.Timedelta(days=28), periods=28)
    fc, act = [], []
    for item, store, dept, cat, state, f, h, price in SERIES:
        sid = f"{item}_{store}_validation"
        fc.append(pd.DataFrame({"id": sid, "item_id": item, "dept_id": dept, "cat_id": cat, "store_id": store,
                                "state_id": state, "date": fdates, "forecast": np.float32(f),
                                "sell_price": np.float32(price)}))
        act.append(pd.DataFrame({"id": sid, "date": hdates, "sales": np.int16(h)}))
    pd.concat(fc).astype({c: "category" for c in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]}) \
        .to_parquet(d / "forecasts.parquet", index=False)
    pd.concat(act).astype({"id": "category"}).to_parquet(d / "actuals_last28.parquet", index=False)
    (d / "metadata.json").write_text(json.dumps({
        "model": "lgbm_test", "best_iteration": 10, "val_wrmsse": 0.5, "forecast_start": "2016-04-25",
        "forecast_end": "2016-05-22", "history_start": "2016-03-28", "history_end": "2016-04-24",
        "n_series": 3, "generated_at": "2026-01-01T00:00:00+00:00"}))
    with TestClient(create_app(d)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["n_series"] == 3


def test_catalog_and_filters(client):
    body = client.get("/catalog").json()
    assert body["stores"] == ["CA_1", "TX_1"] and len(body["items"]) == 2
    assert [i["item_id"] for i in client.get("/catalog?store_id=TX_1").json()["items"]] == ["FOODS_1_001"]
    assert client.get("/catalog?store_id=XX_9").status_code == 404


def test_forecast_full_window(client):
    body = client.get("/forecast?item_id=FOODS_1_001&store_id=CA_1").json()
    assert len(body["forecast"]) == 28 and len(body["history"]) == 28
    assert body["forecast_total"] == pytest.approx(56.0)
    assert body["forecast"][0]["date"] == "2016-04-25" and body["history"][-1]["date"] == "2016-04-24"
    assert body["sell_price"] == 2.5 and body["state_id"] == "CA"


def test_forecast_date_range(client):
    body = client.get("/forecast?item_id=FOODS_1_001&store_id=CA_1&start=2016-04-26&end=2016-04-28").json()
    assert [p["date"] for p in body["forecast"]] == ["2016-04-26", "2016-04-27", "2016-04-28"]
    assert body["forecast_total"] == pytest.approx(6.0)


@pytest.mark.parametrize("query", [
    "item_id=NOPE&store_id=CA_1",               # unknown item
    "item_id=FOODS_1_001&store_id=XX_9",        # unknown store
    "item_id=HOBBIES_1_001&store_id=TX_1",      # item not sold in that store
])
def test_forecast_404(client, query):
    assert client.get(f"/forecast?{query}").status_code == 404


@pytest.mark.parametrize("dates", [
    "start=2016-13-01",                         # malformed
    "start=not-a-date",                         # malformed
    "start=2016-05-01&end=2016-04-30",          # start after end
    "start=2016-04-01",                         # before the forecast window
    "end=2016-06-30",                           # after the forecast window
])
def test_forecast_422(client, dates):
    assert client.get(f"/forecast?item_id=FOODS_1_001&store_id=CA_1&{dates}").status_code == 422


def test_aggregate(client):
    body = client.get("/aggregate?level=store").json()
    totals = {g["key"]: g for g in body["groups"]}
    assert totals["CA_1"]["forecast_total"] == pytest.approx(70.0)  # (2.0 + 0.5) x 28
    assert totals["CA_1"]["history_total"] == pytest.approx(28.0)
    assert body["forecast"][0]["units"] == pytest.approx(3.5)        # all groups summed
    one = client.get("/aggregate?level=cat&key=HOBBIES").json()
    assert one["key"] == "HOBBIES" and one["forecast"][0]["units"] == pytest.approx(0.5)
    assert client.get("/aggregate?level=store&key=XX_9").status_code == 404
    assert client.get("/aggregate?level=region").status_code == 422


def test_alerts_simulated_math(client):
    body = client.get("/alerts?days_of_cover=28").json()
    assert body["simulated"] is True and "simulated stock" in body["note"] and body["min_shortfall_pct"] == 20
    assert body["n_evaluated"] == 2 and body["n_at_risk"] == 1     # HOBBIES has no recent sales
    a = body["alerts"][0]
    assert (a["item_id"], a["store_id"]) == ("FOODS_1_001", "CA_1")
    assert a["simulated_stock"] == pytest.approx(28.0)              # 28 days x 1/day
    assert a["shortfall_units"] == pytest.approx(28.0)              # 56 - 28
    assert a["shortfall_pct"] == pytest.approx(100.0)
    assert a["est_lost_revenue"] == pytest.approx(70.0)             # 28 x 2.50
    assert a["stockout_date"] == "2016-05-09"                       # day 15: 30 > 28


def test_alerts_min_shortfall_pct(client):
    assert client.get("/alerts?days_of_cover=28&min_shortfall_pct=99").json()["n_at_risk"] == 1
    assert client.get("/alerts?days_of_cover=28&min_shortfall_pct=100").json()["n_at_risk"] == 0  # strictly more
    # 20 days of cover: CA_1 stock 20 vs 56 (180%), TX_1 stock 20 vs 28 (40%) -> both above 20%, CA_1 first
    body = client.get("/alerts?days_of_cover=20").json()
    assert [a["store_id"] for a in body["alerts"]] == ["CA_1", "TX_1"]
    assert client.get("/alerts?days_of_cover=20&min_shortfall_pct=50").json()["n_at_risk"] == 1
    assert client.get("/alerts?days_of_cover=28&min_shortfall_pct=-1").status_code == 422


def test_alerts_filters_and_validation(client):
    assert client.get("/alerts?days_of_cover=28&cat_id=HOBBIES").json()["n_evaluated"] == 0
    assert client.get("/alerts?days_of_cover=28&store_id=TX_1").json()["n_at_risk"] == 0
    assert client.get("/alerts?days_of_cover=28&store_id=XX_9").status_code == 404
    for bad in ["days_of_cover=0", "days_of_cover=366", "days_of_cover=abc", ""]:
        assert client.get(f"/alerts?{bad}").status_code == 422


def test_drift_report_missing_then_served(client):
    report_path = client.app.state.serving_dir / "drift_report.json"
    assert client.get("/drift").status_code == 404
    report = {
        "generated_at": "2026-01-01T00:00:00+00:00", "model": "lgbm_test",
        "reference": {"start": "2015-01-01", "end": "2016-02-28", "sample_fraction": 0.1, "seed": 42,
                      "rows_sampled": 100},
        "windows": {"recent_actuals": {"start": "2016-03-28", "end": "2016-04-24", "last_year_start": "2015-03-28",
                                       "last_year_end": "2015-04-24", "rows": 84}},
        "thresholds": {"psi_flag": 0.2, "psi_watch": 0.1, "prediction_gap_pct": 15.0},
        "features": [{"feature": "rmean_28", "window": "recent_actuals", "psi": 0.31, "psi_vs_last_year": 0.05,
                      "ks_stat": 0.2, "ks_pvalue": 0.0, "ref_mean": 1.0, "cur_mean": 1.4, "status": "flagged"}],
        "target": [{"category": "Total", "ref_mean_daily": 1.0, "cur_mean_daily": 1.1, "change_pct": 10.0,
                    "ref_zero_share": 0.6, "cur_zero_share": 0.55, "zero_share_change_pp": -5.0}],
        "prediction": [{"category": "Total", "forecast_mean_daily": 1.05, "recent_actual_mean_daily": 1.1,
                        "gap_pct": -4.5, "status": "ok"}],
        "flagged": ["rmean_28@recent_actuals"],
    }
    report_path.write_text(json.dumps(report))
    try:
        body = client.get("/drift").json()
        assert body["flagged"] == ["rmean_28@recent_actuals"] and body["reference"]["seed"] == 42
        assert body["features"][0]["status"] == "flagged" and body["features"][0]["psi_vs_last_year"] == 0.05
    finally:
        report_path.unlink()
