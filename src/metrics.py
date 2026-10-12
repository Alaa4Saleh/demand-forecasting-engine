"""WRMSSE, the official M5 accuracy metric.

For each of the 42,840 series in the 12-level M5 hierarchy:

    RMSSE = sqrt( mean_h (y - y_hat)^2 / mean_t (y_t - y_{t-1})^2 )

The denominator (scale) uses one-step differences over the training history, starting
from the series' first non-zero sale. Each series is weighted by its share of dollar
sales over the last 28 days of train within its level; WRMSSE is the mean of the 12
level scores (so each level contributes 1/12).
"""
import numpy as np
import pandas as pd
from scipy import sparse

from src.data import HORIZON, to_wide

LEVELS = [
    ("total", []),
    ("state", ["state_id"]),
    ("store", ["store_id"]),
    ("cat", ["cat_id"]),
    ("dept", ["dept_id"]),
    ("state_cat", ["state_id", "cat_id"]),
    ("state_dept", ["state_id", "dept_id"]),
    ("store_cat", ["store_id", "cat_id"]),
    ("store_dept", ["store_id", "dept_id"]),
    ("item", ["item_id"]),
    ("item_state", ["item_id", "state_id"]),
    ("item_store", ["item_id", "store_id"]),
]


def hierarchy_from_ids(ids) -> pd.DataFrame:
    """Recover item/dept/cat/store/state from M5 ids like 'HOBBIES_1_001_CA_1_validation'."""
    base = pd.Series(ids).str.replace(r"_(validation|evaluation)$", "", regex=True)
    parts = base.str.rsplit("_", n=2, expand=True)  # item_id | state | store number
    h = pd.DataFrame({
        "id": list(ids),
        "item_id": parts[0],
        "state_id": parts[1],
        "store_id": parts[1] + "_" + parts[2],
    })
    h["dept_id"] = h["item_id"].str.rsplit("_", n=1).str[0]
    h["cat_id"] = h["item_id"].str.split("_").str[0]
    return h


def dollar_sales(rows: pd.DataFrame, sell_prices: pd.DataFrame) -> pd.Series:
    """Per-id sum of sold * sell_price over `rows` (long: id, sold, wm_yr_wk) - the WRMSSE weights."""
    hier = hierarchy_from_ids(rows["id"].cat.categories).set_index("id")
    recent = pd.DataFrame({
        "id": rows["id"].to_numpy(),
        "sold": rows["sold"].to_numpy(),
        "wm_yr_wk": rows["wm_yr_wk"].to_numpy(),
        "item_id": hier["item_id"].reindex(rows["id"]).to_numpy(),
        "store_id": hier["store_id"].reindex(rows["id"]).to_numpy(),
    })
    prices = sell_prices[sell_prices["wm_yr_wk"].isin(recent["wm_yr_wk"].unique())]
    prices = prices.astype({"item_id": str, "store_id": str, "wm_yr_wk": recent["wm_yr_wk"].dtype})
    recent = recent.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    if recent.loc[recent["sold"] > 0, "sell_price"].isna().any():
        raise ValueError("missing sell_price for rows with sales")
    return (recent["sold"] * recent["sell_price"].fillna(0)).groupby(recent["id"]).sum()


def _aggregation_matrix(hier: pd.DataFrame, keys) -> sparse.csr_matrix:
    """(n_groups x n_bottom) 0/1 matrix summing bottom series into the level's groups."""
    n = len(hier)
    if keys:
        codes = hier.groupby(keys, sort=True).ngroup().to_numpy()
    else:
        codes = np.zeros(n, dtype=np.int64)
    return sparse.csr_matrix(
        (np.ones(n, dtype=np.float32), (codes, np.arange(n))), shape=(codes.max() + 1, n)
    )


def _scale(agg_train: np.ndarray) -> np.ndarray:
    """Mean squared one-step difference per row, counted from the first non-zero value."""
    started = np.cumsum(agg_train != 0, axis=1) > 0
    valid = started[:, :-1]  # diff y_t - y_{t-1} counts once y_{t-1} is in the series
    sq = np.diff(agg_train, axis=1).astype(np.float64) ** 2
    n = valid.sum(axis=1)
    return np.where(n > 0, (sq * valid).sum(axis=1) / np.maximum(n, 1), np.nan)


class WRMSSEEvaluator:
    """Precomputes scales and weights from train; then scores 28-day forecasts.

    train_wide: id x date sales (float), the full training history.
    dollar_sales: per-id dollar sales over the last `horizon` days of train (aligned to train_wide.index).
    """

    def __init__(self, train_wide: pd.DataFrame, dollar_sales: pd.Series, horizon: int = HORIZON):
        self.ids = train_wide.index
        self.horizon = horizon
        hier = hierarchy_from_ids(self.ids)
        train = train_wide.to_numpy(dtype=np.float32)
        dollars = dollar_sales.reindex(self.ids).fillna(0).to_numpy(dtype=np.float64)

        self.levels = {}
        self.labels = {}  # level -> one row of group keys per aggregated series (S row order)
        for name, keys in LEVELS:
            S = _aggregation_matrix(hier, keys)
            self.labels[name] = (hier.groupby(keys, sort=True).size().index.to_frame(index=False)
                                 if keys else pd.DataFrame({"total": ["Total"]}))
            scale = _scale(S @ train)
            level_dollars = S @ dollars
            weight = level_dollars / level_dollars.sum()
            # Series with no sales in train have no scale; they carry zero weight anyway.
            if np.any(np.isnan(scale) & (weight > 0)):
                raise ValueError(f"level {name}: weighted series with undefined scale")
            self.levels[name] = (S, np.nan_to_num(scale, nan=1.0), weight)

    @classmethod
    def from_long(cls, train_long: pd.DataFrame, sell_prices: pd.DataFrame, horizon: int = HORIZON):
        """Build from long train rows (id, date, sold, wm_yr_wk) and raw sell_prices."""
        last = train_long[train_long["date"] > train_long["date"].max() - pd.Timedelta(days=horizon)]
        return cls(to_wide(train_long, "sold"), dollar_sales(last, sell_prices), horizon)

    def _align(self, wide) -> np.ndarray:
        if isinstance(wide, pd.DataFrame):
            missing = self.ids.difference(wide.index)
            if len(missing):
                raise ValueError(f"{len(missing)} ids missing, e.g. {list(missing[:3])}")
            wide = wide.reindex(self.ids)
        arr = np.asarray(wide, dtype=np.float64)
        if arr.shape != (len(self.ids), self.horizon):
            raise ValueError(f"expected shape {(len(self.ids), self.horizon)}, got {arr.shape}")
        return arr

    def level_scores(self, actual_wide, pred_wide) -> pd.Series:
        """Weighted RMSSE per hierarchy level (each level's weights sum to 1)."""
        err = self._align(actual_wide) - self._align(pred_wide)
        out = {}
        for name, (S, scale, weight) in self.levels.items():
            agg_err = S @ err  # aggregation is linear, so aggregate the errors directly
            rmsse = np.sqrt((agg_err ** 2).mean(axis=1) / scale)
            out[name] = float(weight @ rmsse)
        return pd.Series(out, name="wrmsse")

    def series_scores(self, actual_wide, pred_wide, level: str = "item_store") -> pd.DataFrame:
        """Per-series weight, scale, RMSSE and contribution (weight x RMSSE) within one level.

        `share` is the series' fraction of the level's WRMSSE; contributions sum to level_scores()[level].
        """
        S, scale, weight = self.levels[level]
        agg_err = S @ (self._align(actual_wide) - self._align(pred_wide))
        out = self.labels[level].copy()
        out["weight"] = weight
        out["scale"] = scale
        out["rmsse"] = np.sqrt((agg_err ** 2).mean(axis=1) / scale)
        out["contribution"] = out["weight"] * out["rmsse"]
        out["share"] = out["contribution"] / out["contribution"].sum()
        return out

    def score(self, actual_wide, pred_wide) -> float:
        return float(self.level_scores(actual_wide, pred_wide).mean())
