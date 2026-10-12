"""In-memory serving data: loaded once from data/serving/ (written by `python -m src.forecast`)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

LEVEL_COLUMNS = {"store": "store_id", "cat": "cat_id", "dept": "dept_id", "state": "state_id"}


class ServingData:
    def __init__(self, serving_dir: Path):
        serving_dir = Path(serving_dir)
        fc = pd.read_parquet(serving_dir / "forecasts.parquet")
        act = pd.read_parquet(serving_dir / "actuals_last28.parquet")
        self.meta = json.loads((serving_dir / "metadata.json").read_text(encoding="utf-8"))

        fc["id"] = fc["id"].astype(str)
        act["id"] = act["id"].astype(str)
        self.forecast_dates = pd.DatetimeIndex(np.sort(fc["date"].unique()))
        self.history_dates = pd.DatetimeIndex(np.sort(act["date"].unique()))

        # One row per series: hierarchy + latest price; wide forecast / history matrices in the same order.
        series = (fc.sort_values("date").groupby("id", observed=True)
                  .agg(item_id=("item_id", "first"), dept_id=("dept_id", "first"), cat_id=("cat_id", "first"),
                       store_id=("store_id", "first"), state_id=("state_id", "first"),
                       sell_price=("sell_price", "last")))
        for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
            series[col] = series[col].astype(str)
        self.series = series
        self.known = {col: set(series[col]) for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]}
        self.F = (fc.pivot(index="id", columns="date", values="forecast")
                  .reindex(index=series.index, columns=self.forecast_dates).to_numpy(np.float64))
        self.H = (act.pivot(index="id", columns="date", values="sales")
                  .reindex(index=series.index, columns=self.history_dates).fillna(0).to_numpy(np.float64))
        self.row = {(r.item_id, r.store_id): i for i, r in enumerate(series.itertuples())}
        self.recent_daily = self.H.mean(axis=1)

        # Daily totals per group for each aggregation level.
        self.levels = {}
        for level, col in LEVEL_COLUMNS.items():
            keys = series[col].to_numpy()
            groups = sorted(set(keys))
            self.levels[level] = {g: (self.F[keys == g].sum(axis=0), self.H[keys == g].sum(axis=0)) for g in groups}

    @property
    def items(self) -> pd.DataFrame:
        return self.series[["item_id", "dept_id", "cat_id"]].drop_duplicates("item_id").sort_values("item_id")

    def alerts(self, days_of_cover: int, store_id=None, cat_id=None) -> pd.DataFrame:
        """SIMULATED stock-out risk. M5 has no inventory data: stock is assumed to equal
        days_of_cover x the series' mean daily sales over the last 28 actual days. A series is at
        risk when its forecast demand over the 28-day window exceeds that stock. Series with no
        sales in the last 28 days are not evaluated."""
        mask = self.recent_daily > 0
        if store_id:
            mask &= self.series["store_id"].to_numpy() == store_id
        if cat_id:
            mask &= self.series["cat_id"].to_numpy() == cat_id
        stock = days_of_cover * self.recent_daily[mask]
        cum = np.cumsum(self.F[mask], axis=1)
        at_risk = cum[:, -1] > stock
        first_short = np.argmax(cum > stock[:, None], axis=1)

        s = self.series[mask]
        out = pd.DataFrame({
            "item_id": s["item_id"].to_numpy(), "store_id": s["store_id"].to_numpy(),
            "dept_id": s["dept_id"].to_numpy(), "cat_id": s["cat_id"].to_numpy(),
            "sell_price": s["sell_price"].to_numpy(np.float64),
            "recent_daily_sales": self.recent_daily[mask],
            "simulated_stock": stock,
            "forecast_demand": cum[:, -1],
            "shortfall_units": cum[:, -1] - stock,
            "stockout_date": self.forecast_dates[first_short].date,
        })[at_risk]
        out["est_lost_revenue"] = out["shortfall_units"] * out["sell_price"]
        out.attrs["n_evaluated"] = int(mask.sum())
        return out.sort_values("est_lost_revenue", ascending=False)
