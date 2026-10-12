"""Naive baselines on wide (id x date) training sales."""
import numpy as np
import pandas as pd


def _future_dates(train_wide: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(train_wide.columns[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")


def naive(train_wide: pd.DataFrame, horizon: int = 28) -> pd.DataFrame:
    """Repeat the last observed day."""
    last = train_wide.iloc[:, -1].to_numpy()
    return pd.DataFrame(np.repeat(last[:, None], horizon, axis=1),
                        index=train_wide.index, columns=_future_dates(train_wide, horizon))


def seasonal_naive(train_wide: pd.DataFrame, horizon: int = 28, season: int = 7) -> pd.DataFrame:
    """Repeat the last `season` days (same weekday one week back)."""
    last_season = train_wide.iloc[:, -season:].to_numpy()
    reps = -(-horizon // season)
    return pd.DataFrame(np.tile(last_season, reps)[:, :horizon],
                        index=train_wide.index, columns=_future_dates(train_wide, horizon))
