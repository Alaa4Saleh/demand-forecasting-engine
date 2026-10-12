"""Feature engineering for the global LightGBM model.

Built per store from sales_long_clean.parquet + sell_prices.csv and written to
data/processed/features/<store_id>.parquet (one row per id x date, same rows as the source).

Leakage: every sales-derived feature only looks at `sold` at least 28 days back, so the
whole 28-day validation window is computable from train alone (no recursive forecasting).
Price and calendar features are known in advance in M5, so they are used as-of the row's date.
"""
import time

import numpy as np
import pandas as pd

from src.data import PROCESSED_DIR, load_sell_prices
from src.metrics import hierarchy_from_ids

FEATURES_DIR = PROCESSED_DIR / "features"
SALES_PATH = PROCESSED_DIR / "sales_long_clean.parquet"

LAGS = (28, 35, 42)
ROLL_SHIFT = 28
ROLL_WINDOWS = (7, 14, 28)

CATEGORICAL_FEATURES = ["item_id", "dept_id", "cat_id", "store_id", "state_id",
                        "event_name_1", "event_type_1"]
NUMERIC_FEATURES = (
    [f"lag_{k}" for k in LAGS]
    + [f"r{stat}_{w}" for w in ROLL_WINDOWS for stat in ("mean", "std")]
    + ["dow", "month", "has_event", "snap", "is_store_closed",
       "sell_price", "price_change", "price_rel_cat"]
)
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
EXTRA_FEATURES = ["mday", "week_of_year", "price_rmean_4w", "price_vs_rmean_4w", "zero_streak_28"]
FEATURE_SETS = {"base": FEATURES, "extended": FEATURES + EXTRA_FEATURES}

_SOURCE_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "date", "sold",
                   "wm_yr_wk", "month", "event_name_1", "event_type_1",
                   "snap_CA", "snap_TX", "snap_WI", "is_store_closed"]


def category_levels() -> dict:
    """Global category lists, so per-store files share identical category codes.

    (Filtered parquet reads only see the dictionary values present in that store.)
    """
    cats = pd.read_parquet(SALES_PATH, columns=["id", "dept_id", "cat_id", "state_id",
                                                "event_name_1", "event_type_1"])
    levels = {c: list(cats[c].cat.categories) for c in cats.columns}
    hier = hierarchy_from_ids(levels["id"])
    levels["item_id"] = sorted(hier["item_id"].unique())
    levels["store_id"] = sorted(hier["store_id"].unique())
    return levels


def _shift_within_series(values: np.ndarray, pos: np.ndarray, k: int) -> np.ndarray:
    """values[i - k] for rows sorted by (id, date) with contiguous days; NaN in each series' first k rows."""
    out = np.full_like(values, np.nan)
    out[k:] = values[:-k]
    out[pos < k] = np.nan
    return out


def _price_features(prices: pd.DataFrame, store: str, item_levels) -> pd.DataFrame:
    p = prices.loc[prices["store_id"] == store, ["item_id", "wm_yr_wk", "sell_price"]].copy()
    p["item_id"] = pd.Categorical(p["item_id"].astype(str), categories=item_levels)
    p = p.sort_values(["item_id", "wm_yr_wk"])
    prev = p.groupby("item_id", observed=True)["sell_price"].shift(1)
    p["price_change"] = (p["sell_price"] / prev - 1).astype(np.float32)
    p["cat"] = p["item_id"].astype(str).str.split("_").str[0]
    cat_mean = p.groupby(["cat", "wm_yr_wk"], observed=True)["sell_price"].transform("mean")
    p["price_rel_cat"] = (p["sell_price"] / cat_mean).astype(np.float32)
    rmean = p.groupby("item_id", observed=True)["sell_price"].transform(
        lambda s: s.rolling(4, min_periods=1).mean())
    p["price_rmean_4w"] = rmean.astype(np.float32)
    p["price_vs_rmean_4w"] = (p["sell_price"] / rmean).astype(np.float32)
    return p.drop(columns="cat")


def _zero_streak(sold: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Consecutive zero-sale days ending at each row (0 on a sale day), reset at each series start."""
    is_zero = (sold == 0).astype(np.int32)
    run_id = np.cumsum((is_zero == 0) | (pos == 0))
    return pd.Series(is_zero).groupby(run_id).cumsum().to_numpy(np.float32)


def build_store_features(store: str, prices: pd.DataFrame, levels: dict) -> pd.DataFrame:
    df = pd.read_parquet(SALES_PATH, columns=_SOURCE_COLUMNS, filters=[("store_id", "==", store)])
    for col in ["id", *CATEGORICAL_FEATURES]:
        df[col] = pd.Categorical(df[col].astype(object), categories=levels[col])
    df = df.sort_values(["id", "date"], kind="stable").reset_index(drop=True)

    # Lags and rolling stats, all on sold shifted >= 28 days.
    sold = df["sold"].to_numpy(np.float32)
    pos = df.groupby("id", observed=True).cumcount().to_numpy()
    for k in LAGS:
        df[f"lag_{k}"] = _shift_within_series(sold, pos, k)
    shifted = pd.Series(_shift_within_series(sold, pos, ROLL_SHIFT))
    for w in ROLL_WINDOWS:
        # min_periods=w: windows touching a series' first 28 rows (NaN) stay NaN, never cross ids.
        roll = shifted.rolling(w, min_periods=w)
        df[f"rmean_{w}"] = roll.mean().to_numpy(np.float32)
        df[f"rstd_{w}"] = roll.std().to_numpy(np.float32)
    # Zero run as of t-28 (not t): how long the item had been without sales at the forecast origin.
    df["zero_streak_28"] = _shift_within_series(_zero_streak(sold, pos), pos, ROLL_SHIFT)

    # Calendar.
    df["dow"] = df["date"].dt.dayofweek.astype(np.int8)
    df["mday"] = df["date"].dt.day.astype(np.int8)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(np.int8).to_numpy()
    df["month"] = df["month"].astype(np.int8)
    df["has_event"] = df["event_name_1"].notna().astype(np.int8)
    df["snap"] = df[f"snap_{store.split('_')[0]}"].astype(np.int8)
    df["is_store_closed"] = df["is_store_closed"].astype(np.int8)

    # Price.
    df["wm_yr_wk"] = df["wm_yr_wk"].astype(np.int32)
    df = df.merge(_price_features(prices, store, levels["item_id"]),
                  on=["item_id", "wm_yr_wk"], how="left")

    df["sold"] = df["sold"].astype(np.float32)
    return df[["id", "date", "sold", *FEATURE_SETS["extended"]]]


def build_all(stores=None) -> None:
    levels = category_levels()
    prices = load_sell_prices()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for store in stores or levels["store_id"]:
        t = time.time()
        feats = build_store_features(store, prices, levels)
        feats.to_parquet(FEATURES_DIR / f"{store}.parquet", index=False)
        print(f"{store}: {len(feats):,} rows in {time.time() - t:.0f}s", flush=True)


def load_features(start=None, end=None, columns=None, stores=None) -> pd.DataFrame:
    """Read feature files, optionally restricted to start <= date <= end."""
    filters = []
    if start is not None:
        filters.append(("date", ">=", pd.Timestamp(start)))
    if end is not None:
        filters.append(("date", "<=", pd.Timestamp(end)))
    paths = [FEATURES_DIR / f"{s}.parquet" for s in stores] if stores \
        else sorted(FEATURES_DIR.glob("*.parquet"))
    return pd.concat([pd.read_parquet(p, columns=columns, filters=filters or None) for p in paths],
                     ignore_index=True)


if __name__ == "__main__":
    build_all()
