"""Global LightGBM (one model for all 30,490 series) on the features from src/features.py.

Time windows, all relative to the end of the data:
    fit:      train_start .. 2016-02-28
    inner:    2016-02-29 .. 2016-03-27  early stopping (RMSE) and config selection (inner WRMSSE)
    val:      2016-03-28 .. 2016-04-24  scored only with --score-val, for the final chosen config

Inner WRMSSE uses an evaluator built from data up to 2016-02-28 only, so model selection
never sees the scored window. Each run appends a record to artifacts/experiments.jsonl.

Run:  python -m src.models.lgbm --name e1 --lr 0.02 --lambda-l2 1.0 --min-data-in-leaf 500 --patience 200
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data import EVAL_COLUMNS, HORIZON, PROJECT_ROOT, load_sales_long, load_sell_prices, to_wide
from src.features import CATEGORICAL_FEATURES, FEATURE_SETS, FEATURES_DIR
from src.metrics import WRMSSEEvaluator, dollar_sales
from src.models.baselines import naive, seasonal_naive

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EXPERIMENT_LOG = ARTIFACTS_DIR / "experiments.jsonl"

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


def _paths(stores=None):
    return [FEATURES_DIR / f"{s}.parquet" for s in stores] if stores \
        else sorted(FEATURES_DIR.glob("*.parquet"))


def _filters(start, end):
    return [("date", ">=", pd.Timestamp(start)), ("date", "<=", pd.Timestamp(end))]


def _read_chunk(path, start, end, features):
    """One store's rows in [start, end] as a float32 matrix (categoricals as codes, -1 -> NaN)."""
    chunk = pd.read_parquet(path, columns=["id", "date", "sold", *features], filters=_filters(start, end))
    return to_matrix(chunk, features), chunk["sold"].to_numpy(np.float32), chunk[["id", "date"]]


def to_matrix(chunk: pd.DataFrame, features) -> np.ndarray:
    """Feature frame -> float32 model input (categoricals as their global codes, -1 -> NaN)."""
    X = np.empty((len(chunk), len(features)), dtype=np.float32)
    for j, col in enumerate(features):
        if col in CATEGORICAL_FEATURES:
            codes = chunk[col].cat.codes.to_numpy(np.float32)
            codes[codes < 0] = np.nan
            X[:, j] = codes
        else:
            X[:, j] = chunk[col].to_numpy(np.float32)
    return X


def load_matrix(start, end, features, stores=None):
    """Small windows (inner / val) fully in memory: X, y, keys (id, date)."""
    parts = [_read_chunk(p, start, end, features) for p in _paths(stores)]
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            pd.concat([p[2] for p in parts], ignore_index=True))


class _StoreSequence(lgb.Sequence):
    """Lazily loads one store's feature rows; only one store is held in memory at a time.

    LightGBM reads each Sequence front to back (bin sampling, then pushing rows), so the
    single-entry cache means each store file is read about twice per Dataset build.
    """
    _cache = {"key": None, "X": None}

    def __init__(self, path, start, end, features, n_rows):
        self.key = (str(path), str(start), str(end), tuple(features))
        self.n_rows = n_rows
        self.batch_size = 1 << 16

    def __len__(self):
        return self.n_rows

    def __getitem__(self, idx):
        cache = _StoreSequence._cache
        if cache["key"] != self.key:
            cache.update(key=None, X=None)  # drop the previous store before loading the next
            path, start, end, features = self.key
            cache.update(key=self.key, X=_read_chunk(path, start, end, list(features))[0])
        return cache["X"][idx].astype(np.float64)  # Sequence input must be double; cache stays float32

    @classmethod
    def clear(cls):
        cls._cache.update(key=None, X=None)


def train_dataset(start, end, features, params, stores=None) -> lgb.Dataset:
    """Streams the fit window store by store: peak memory ~ binned data + one store."""
    seqs, labels = [], []
    for p in _paths(stores):
        y = pd.read_parquet(p, columns=["sold"], filters=_filters(start, end))["sold"].to_numpy(np.float32)
        seqs.append(_StoreSequence(p, start, end, features, len(y)))
        labels.append(y)
    ds = lgb.Dataset(seqs, label=np.concatenate(labels), feature_name=features,
                     categorical_feature=[c for c in features if c in CATEGORICAL_FEATURES],
                     params=params, free_raw_data=True)
    ds.construct()
    _StoreSequence.clear()
    return ds


def predict_wide(booster, start, end, features, stores=None) -> pd.DataFrame:
    X, _, keys = load_matrix(start, end, features, stores)
    keys["pred"] = np.clip(booster.predict(X, num_iteration=booster.best_iteration), 0, None)
    return to_wide(keys, "pred", dates=pd.date_range(start, end))


def build_evaluation(windows: dict, horizon: int = HORIZON) -> dict:
    """For each name -> (train_end, window_start, window_end): an evaluator from data <= train_end,
    the actual sales in the window, and baseline level scores. Actuals are loaded only for the
    windows asked for, which keeps the scored window locked unless requested."""
    sales = load_sales_long(EVAL_COLUMNS)
    prices = load_sell_prices()
    wide = to_wide(sales)
    out = {}
    for name, (train_end, start, end) in windows.items():
        recent = sales[(sales["date"] > train_end - pd.Timedelta(days=horizon)) & (sales["date"] <= train_end)]
        train_wide = wide.loc[:, :train_end]
        ev = WRMSSEEvaluator(train_wide, dollar_sales(recent, prices), horizon)
        actual = wide.loc[:, start:end]
        out[name] = {
            "evaluator": ev,
            "actual": actual,
            "baselines": {"Naive": ev.level_scores(actual, naive(train_wide, horizon)),
                          "Seasonal Naive": ev.level_scores(actual, seasonal_naive(train_wide, horizon))},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="experiment id, used for the log and artifact names")
    ap.add_argument("--train-start", default="2015-01-01")
    ap.add_argument("--features", default="base", choices=sorted(FEATURE_SETS))
    ap.add_argument("--lr", type=float, default=PARAMS["learning_rate"])
    ap.add_argument("--lambda-l2", type=float, default=PARAMS["lambda_l2"])
    ap.add_argument("--min-data-in-leaf", type=int, default=PARAMS["min_data_in_leaf"])
    ap.add_argument("--tweedie-power", type=float, default=PARAMS["tweedie_variance_power"])
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=5000)
    ap.add_argument("--score-val", action="store_true",
                    help="also score the held-out val window (final config only)")
    ap.add_argument("--stores", nargs="*", help="subset of stores (smoke tests only)")
    args = ap.parse_args()
    t0 = time.time()

    params = {**PARAMS, "learning_rate": args.lr, "lambda_l2": args.lambda_l2,
              "min_data_in_leaf": args.min_data_in_leaf, "tweedie_variance_power": args.tweedie_power}
    features = FEATURE_SETS[args.features]

    data_end = pq.read_table(_paths()[0], columns=["date"])["date"].to_pandas().max()
    val_start = data_end - pd.Timedelta(days=HORIZON - 1)
    inner_end = val_start - pd.Timedelta(days=1)
    inner_start = inner_end - pd.Timedelta(days=HORIZON - 1)
    fit_end = inner_start - pd.Timedelta(days=1)
    print(f"[{args.name}] fit {args.train_start}..{fit_end.date()} | inner {inner_start.date()}.."
          f"{inner_end.date()} | val {'SCORED' if args.score_val else 'locked'}", flush=True)

    windows = {"inner": (fit_end, inner_start, inner_end)}
    if args.score_val:
        windows["val"] = (inner_end, val_start, data_end)
    evaluation = {} if args.stores else build_evaluation(windows)

    dtrain = train_dataset(args.train_start, fit_end, features, params, args.stores)
    n_fit = dtrain.num_data()
    print(f"fit rows: {n_fit:,}  ({time.time() - t0:.0f}s)", flush=True)
    Xi, yi, _ = load_matrix(inner_start, inner_end, features, args.stores)
    dinner = lgb.Dataset(Xi, yi, reference=dtrain)
    booster = lgb.train(params, dtrain, args.rounds, valid_sets=[dinner], valid_names=["inner"],
                        callbacks=[lgb.early_stopping(args.patience), lgb.log_evaluation(100)])
    del dtrain, dinner, Xi, yi
    train_secs = time.time() - t0

    importance = pd.Series(booster.feature_importance("gain"), index=features)
    importance = (importance / importance.sum()).sort_values(ascending=False)
    print(f"best_iteration={booster.best_iteration}  ({train_secs:.0f}s)")
    print(importance.round(3).head(15).to_string())

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    booster.save_model(ARTIFACTS_DIR / f"lgbm_{args.name}.txt", num_iteration=booster.best_iteration)
    record = {"name": args.name, "train_start": args.train_start, "features": args.features,
              "params": params, "patience": args.patience, "fit_rows": n_fit,
              "best_iteration": booster.best_iteration,
              "inner_rmse": booster.best_score["inner"]["rmse"],
              "train_seconds": round(train_secs),
              "importance_top15": importance.head(15).round(4).to_dict()}
    if args.stores:
        print(json.dumps(record, indent=1))
        return  # WRMSSE weights/scales are defined over all 30,490 series

    for name in windows:
        e = evaluation[name]
        start, end = windows[name][1:]
        pred = predict_wide(booster, start, end, features)
        scores = pd.DataFrame({**e["baselines"], "LightGBM": e["evaluator"].level_scores(e["actual"], pred)})
        print(f"\n{name} WRMSSE per level:\n{scores.round(4).to_string()}")
        print(f"{name} WRMSSE: " + "  ".join(f"{k} {v:.4f}" for k, v in scores.mean().items()))
        record[f"{name}_wrmsse"] = round(float(scores["LightGBM"].mean()), 4)
        record[f"{name}_levels"] = scores["LightGBM"].round(4).to_dict()
        if name == "val":
            long = pred.stack().rename("pred").reset_index().rename(columns={"level_1": "date"})
            long.to_parquet(ARTIFACTS_DIR / f"lgbm_{args.name}_val_pred.parquet", index=False)

    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nlogged to {EXPERIMENT_LOG.name}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
