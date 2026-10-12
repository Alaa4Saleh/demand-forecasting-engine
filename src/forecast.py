"""Produce the serving forecast: the next 28 days after the last sales date, for every series.

Uses the final LightGBM model (artifacts/lgbm_final.txt) as is. Future feature rows are built by
src.features.build_store_features(future_days=28), the same code path as training; every
sales feature is lagged >= 28 days, so the whole horizon is computable from real history.

Writes data/serving/:
    forecasts.parquet       id, hierarchy, date, forecast, sell_price  (30,490 x 28 rows)
    actuals_last28.parquet  id, date, sales                            (last 28 days of actuals)
    metadata.json           model, windows, scores, generation time
Run:  python -m src.forecast
"""
import argparse
import json
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data import HORIZON, PROJECT_ROOT, load_sell_prices
from src.features import build_store_features, category_levels, load_calendar
from src.models.lgbm import ARTIFACTS_DIR, EXPERIMENT_LOG, to_matrix

SERVING_DIR = PROJECT_ROOT / "data" / "serving"
HIERARCHY = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _experiment_record(name: str) -> dict:
    """Last logged run with this name (scores and config for the metadata), if any."""
    if not EXPERIMENT_LOG.exists():
        return {}
    records = [json.loads(line) for line in EXPERIMENT_LOG.read_text(encoding="utf-8").splitlines() if line]
    return next((r for r in reversed(records) if r["name"] == name), {})


def make_forecasts(model_name: str = "final", horizon: int = HORIZON):
    booster = lgb.Booster(model_file=str(ARTIFACTS_DIR / f"lgbm_{model_name}.txt"))
    features = booster.feature_name()
    levels, prices, calendar = category_levels(), load_sell_prices(), load_calendar()

    forecasts, actuals = [], []
    for store in levels["store_id"]:
        t = time.time()
        df = build_store_features(store, prices, levels, calendar, future_days=horizon)
        last_actual = df.loc[df["sold"].notna(), "date"].max()
        fut = df[df["date"] > last_actual]
        out = fut[["id", *HIERARCHY, "date", "sell_price"]].copy()
        out["forecast"] = booster.predict(to_matrix(fut, features)).astype(np.float32)
        forecasts.append(out)
        hist = df[(df["date"] > last_actual - pd.Timedelta(days=horizon)) & (df["date"] <= last_actual)]
        actuals.append(pd.DataFrame({"id": hist["id"], "date": hist["date"],
                                     "sales": hist["sold"].astype(np.int16)}))
        print(f"{store}: {len(fut):,} forecast rows in {time.time() - t:.0f}s", flush=True)

    fc = pd.concat(forecasts, ignore_index=True)[["id", *HIERARCHY, "date", "forecast", "sell_price"]]
    act = pd.concat(actuals, ignore_index=True)
    fc["sell_price"] = fc["sell_price"].astype(np.float32)

    # Sanity check: complete, finite, non-negative forecasts; compare volume with the last 28 actual days.
    assert fc["forecast"].notna().all() and (fc["forecast"] >= 0).all(), "forecasts must be non-negative, no NaN"
    assert len(fc) == fc["id"].nunique() * horizon == act["id"].nunique() * horizon, "series/horizon mismatch"
    print(f"Total forecast units {fc['forecast'].sum():,.0f} ({fc['date'].min().date()}..{fc['date'].max().date()})"
          f" vs last 28 actual days {act['sales'].sum():,} ({act['date'].min().date()}..{act['date'].max().date()})")
    return fc, act, booster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="final", help="artifacts/lgbm_<model>.txt")
    args = ap.parse_args()

    fc, act, booster = make_forecasts(args.model)
    SERVING_DIR.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(SERVING_DIR / "forecasts.parquet", index=False)
    act.to_parquet(SERVING_DIR / "actuals_last28.parquet", index=False)

    record = _experiment_record(args.model)
    meta = {
        "model": f"lgbm_{args.model}",
        "best_iteration": booster.current_iteration(),
        "train_start": record.get("train_start"),
        "inner_wrmsse": record.get("inner_wrmsse"),
        "val_wrmsse": record.get("val_wrmsse"),
        "forecast_start": str(fc["date"].min().date()),
        "forecast_end": str(fc["date"].max().date()),
        "history_start": str(act["date"].min().date()),
        "history_end": str(act["date"].max().date()),
        "n_series": int(fc["id"].nunique()),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (SERVING_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    sizes = {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in sorted(SERVING_DIR.glob("*.parquet"))}
    print(f"wrote {SERVING_DIR}: {sizes}")


if __name__ == "__main__":
    main()
