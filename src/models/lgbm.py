"""Global LightGBM (one model for all 30,490 series) on the features from src/features.py.

Time-based setup, all relative to the end of the data:
    fit:         train_start .. (val_start - 29d)
    early stop:  the last 28 days of train (inner window, never the scored window)
    scored val:  the last 28 days of the data -> WRMSSE
Run:  python -m src.models.lgbm --train-start 2014-01-01
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data import EVAL_COLUMNS, HORIZON, PROJECT_ROOT, load_sales_long, load_sell_prices, time_split, to_wide
from src.features import CATEGORICAL_FEATURES, FEATURES, FEATURES_DIR
from src.metrics import WRMSSEEvaluator
from src.models.baselines import naive, seasonal_naive

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "max_bin": 255,
    "seed": 42,
    "verbose": -1,
}


def load_matrix(start, end, stores=None):
    """Features for start <= date <= end as one float32 matrix, filled store by store.

    Categoricals are stored as their (globally consistent) codes; -1 -> NaN (missing).
    Returns X, y, keys (id, date) for mapping predictions back.
    """
    paths = [FEATURES_DIR / f"{s}.parquet" for s in stores] if stores \
        else sorted(FEATURES_DIR.glob("*.parquet"))
    filters = [("date", ">=", pd.Timestamp(start)), ("date", "<=", pd.Timestamp(end))]
    n = sum(pq.read_table(p, columns=["date"], filters=filters).num_rows for p in paths)

    X = np.empty((n, len(FEATURES)), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    keys, row = [], 0
    for p in paths:
        chunk = pd.read_parquet(p, columns=["id", "date", "sold", *FEATURES], filters=filters)
        m = len(chunk)
        for j, col in enumerate(FEATURES):
            if col in CATEGORICAL_FEATURES:
                codes = chunk[col].cat.codes.to_numpy(np.float32)
                codes[codes < 0] = np.nan
                X[row:row + m, j] = codes
            else:
                X[row:row + m, j] = chunk[col].to_numpy(np.float32)
        y[row:row + m] = chunk["sold"].to_numpy(np.float32)
        keys.append(chunk[["id", "date"]])
        row += m
    return X, y, pd.concat(keys, ignore_index=True)


def train(train_start, fit_end, stop_start, stop_end, params=PARAMS,
          num_boost_round=3000, early_stopping_rounds=100, stores=None) -> lgb.Booster:
    X, y, _ = load_matrix(train_start, fit_end, stores)
    print(f"fit rows: {len(y):,}  ({X.nbytes / 1e9:.1f} GB)", flush=True)
    dtrain = lgb.Dataset(X, y, feature_name=FEATURES, categorical_feature=CATEGORICAL_FEATURES,
                         free_raw_data=True)
    dtrain.construct()
    del X, y  # LightGBM keeps only the binned data

    Xv, yv, _ = load_matrix(stop_start, stop_end, stores)
    dvalid = lgb.Dataset(Xv, yv, reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round, valid_sets=[dvalid], valid_names=["inner_val"],
                     callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(100)])


def predict_wide(booster, start, end, stores=None) -> pd.DataFrame:
    X, _, keys = load_matrix(start, end, stores)
    keys["pred"] = np.clip(booster.predict(X, num_iteration=booster.best_iteration), 0, None)
    return to_wide(keys, "pred", dates=pd.date_range(start, end))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-start", default="2014-01-01")
    ap.add_argument("--stores", nargs="*", help="subset of stores (smoke tests); WRMSSE needs all")
    ap.add_argument("--rounds", type=int, default=3000)
    args = ap.parse_args()
    t = time.time()

    data_end = pq.read_table(next(FEATURES_DIR.glob("*.parquet")), columns=["date"])["date"].to_pandas().max()
    val_start = data_end - pd.Timedelta(days=HORIZON - 1)
    stop_start = val_start - pd.Timedelta(days=HORIZON)
    fit_end = stop_start - pd.Timedelta(days=1)
    print(f"fit {args.train_start}..{fit_end.date()} | early stop {stop_start.date()}.."
          f"{(val_start - pd.Timedelta(days=1)).date()} | val {val_start.date()}..{data_end.date()}")

    booster = train(args.train_start, fit_end, stop_start, val_start - pd.Timedelta(days=1),
                    num_boost_round=args.rounds, stores=args.stores)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    booster.save_model(ARTIFACTS_DIR / "lgbm_tweedie.txt", num_iteration=booster.best_iteration)
    pred = predict_wide(booster, val_start, data_end, args.stores)
    print(f"trained + predicted in {time.time() - t:.0f}s, best_iteration={booster.best_iteration}")

    importance = pd.Series(booster.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print((importance / importance.sum()).round(3).head(15).to_string())

    if args.stores:
        return  # WRMSSE weights/scales are defined over all 30,490 series
    train_long, val_long = time_split(load_sales_long(EVAL_COLUMNS))
    evaluator = WRMSSEEvaluator.from_long(train_long, load_sell_prices())
    train_wide = to_wide(train_long)
    actual = to_wide(val_long, dates=pd.date_range(val_start, data_end))
    del train_long, val_long
    scores = pd.DataFrame({
        "Naive": evaluator.level_scores(actual, naive(train_wide)),
        "Seasonal Naive": evaluator.level_scores(actual, seasonal_naive(train_wide)),
        "LightGBM": evaluator.level_scores(actual, pred),
    })
    print(scores.round(4).to_string())
    print("WRMSSE:\n" + scores.mean().round(4).to_string())


if __name__ == "__main__":
    main()
