"""Loading helpers for the processed M5 data and the time-based split."""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HORIZON = 28

# Columns needed for evaluation; item_id/store_id etc. are recovered from `id`
# (cheaper than loading the large_string columns for 46M rows).
EVAL_COLUMNS = ["id", "date", "sold", "wm_yr_wk"]


def load_sales_long(columns=None) -> pd.DataFrame:
    """Load data/processed/sales_long_clean.parquet (only `columns` if given)."""
    return pd.read_parquet(PROCESSED_DIR / "sales_long_clean.parquet", columns=columns)


def load_sell_prices() -> pd.DataFrame:
    return pd.read_csv(
        RAW_DIR / "sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category",
               "wm_yr_wk": np.int32, "sell_price": np.float32},
    )


def time_split(df: pd.DataFrame, horizon: int = HORIZON):
    """Split by date: the last `horizon` days are validation, everything before is train."""
    val_start = df["date"].max() - pd.Timedelta(days=horizon - 1)
    is_val = df["date"] >= val_start
    return df[~is_val], df[is_val]


def to_wide(df: pd.DataFrame, value_col: str = "sold", ids=None, dates=None) -> pd.DataFrame:
    """Pivot long (id, date, value) to an id x date float32 frame.

    Missing cells (days before an item's release, trimmed in 01_first_look) become 0,
    matching the raw M5 files.
    """
    ids = df["id"].cat.categories if ids is None else ids
    id_codes = pd.Categorical(df["id"], categories=ids).codes
    dates = pd.DatetimeIndex(np.sort(df["date"].unique()) if dates is None else dates)
    date_codes = dates.get_indexer(df["date"])
    if (id_codes < 0).any() or (date_codes < 0).any():
        raise ValueError("df contains ids or dates outside the requested grid")

    values = np.zeros((len(ids), len(dates)), dtype=np.float32)
    values[id_codes, date_codes] = df[value_col].to_numpy(dtype=np.float32)
    return pd.DataFrame(values, index=pd.Index(ids, name="id"), columns=pd.DatetimeIndex(dates))
