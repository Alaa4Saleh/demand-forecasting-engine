"""Error analysis of a LightGBM val forecast (artifacts/lgbm_<name>_val_pred.parquet).

Run only after the final config is locked: this looks at the scored window, so nothing here
should feed back into model selection.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data import EVAL_COLUMNS, HORIZON, RAW_DIR, load_sales_long, load_sell_prices, to_wide
from src.metrics import WRMSSEEvaluator, dollar_sales, hierarchy_from_ids
from src.models.baselines import seasonal_naive
from src.models.lgbm import ARTIFACTS_DIR, load_matrix

ZERO_BUCKETS = [0, 0.2, 0.5, 0.8, 1.0]
ZERO_LABELS = ["0-20%", "20-50%", "50-80%", "80-100%"]


def load_context(name: str, horizon: int = HORIZON) -> dict:
    """Val evaluator (train up to the day before val), actuals, LightGBM and sNaive forecasts,
    per-series descriptive stats from the last 365 train days, and val-window event dates."""
    sales = load_sales_long(EVAL_COLUMNS)
    prices = load_sell_prices()
    wide = to_wide(sales)
    val_end = wide.columns[-1]
    val_start = val_end - pd.Timedelta(days=horizon - 1)
    train_end = val_start - pd.Timedelta(days=1)

    train_wide = wide.loc[:, :train_end]
    recent = sales[(sales["date"] > train_end - pd.Timedelta(days=horizon)) & (sales["date"] <= train_end)]
    ev = WRMSSEEvaluator(train_wide, dollar_sales(recent, prices), horizon)
    actual = wide.loc[:, val_start:]

    pred = (pd.read_parquet(ARTIFACTS_DIR / f"lgbm_{name}_val_pred.parquet")
            .pivot(index="id", columns="date", values="pred")
            .reindex(index=actual.index, columns=actual.columns))
    if pred.isna().any().any():
        raise ValueError("prediction file does not cover every id x val date")

    # Descriptive stats over the last 365 train days (only rows after each item's release).
    year = sales[(sales["date"] > train_end - pd.Timedelta(days=365)) & (sales["date"] <= train_end)]
    g = year.groupby("id", observed=True)["sold"]
    stats = pd.DataFrame({"zero_share": g.apply(lambda s: (s == 0).mean()), "mean_sales": g.mean()})

    # Price changes in the Walmart weeks overlapping the val window.
    val_weeks = sales.loc[sales["date"] >= val_start, "wm_yr_wk"].unique()
    p = prices.sort_values(["store_id", "item_id", "wm_yr_wk"])
    changed = p.groupby(["store_id", "item_id"], observed=True)["sell_price"].diff().fillna(0) != 0
    n_changes = changed[p["wm_yr_wk"].isin(val_weeks)].groupby(
        [p["item_id"].astype(str) + "_" + p["store_id"].astype(str)]).sum()
    hier = hierarchy_from_ids(actual.index).set_index("id")
    stats = hier.join(stats)
    stats["price_changes_val"] = (hier["item_id"] + "_" + hier["store_id"]).map(n_changes).fillna(0).astype(int).values

    cal = pd.read_csv(RAW_DIR / "calendar.csv", parse_dates=["date"])
    cal = cal[(cal["date"] >= val_start) & (cal["date"] <= val_end)]
    events = cal.loc[cal["event_name_1"].notna(), ["date", "event_name_1", "event_type_1"]]

    return {"evaluator": ev, "actual": actual, "pred": pred,
            "snaive": seasonal_naive(train_wide, horizon), "stats": stats, "events": events,
            "history": train_wide.iloc[:, -90:].copy()}


def plot_series(ctx: dict, ids, ax_size=(6, 3)):
    """Last 90 train days + val actuals vs LightGBM forecast; shading = the 28 days the
    forecast cannot see (all sales features are lagged >= 28 days)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(ids), figsize=(ax_size[0] * len(ids), ax_size[1]), squeeze=False)
    for ax, sid in zip(axes[0], ids):
        hist, act, pred = ctx["history"].loc[sid], ctx["actual"].loc[sid], ctx["pred"].loc[sid]
        ax.plot(hist.index, hist.values, color="grey", lw=1, label="train")
        ax.plot(act.index, act.values, color="black", lw=1, label="actual (val)")
        ax.plot(pred.index, pred.values, color="tab:orange", lw=2, label="LightGBM")
        ax.axvspan(hist.index[-28], hist.index[-1], color="tab:blue", alpha=0.08)
        ax.set_title(sid.replace("_validation", ""), fontsize=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def series_table(ctx: dict) -> pd.DataFrame:
    """One row per item-store: weight, RMSSE, contribution/share of the level-12 WRMSSE,
    sNaive RMSSE, forecast bias and descriptive stats."""
    ev, actual, pred = ctx["evaluator"], ctx["actual"], ctx["pred"]
    t = ev.series_scores(actual, pred, "item_store")
    t["id"] = t["item_id"] + "_" + t["store_id"] + "_validation"
    t = t.set_index("id")
    t["rmsse_snaive"] = ev.series_scores(actual, ctx["snaive"], "item_store").set_axis(t.index)["rmsse"]
    act_sum, pred_sum = actual.sum(axis=1), pred.sum(axis=1)
    t["actual_28d"] = act_sum
    t["pred_28d"] = pred_sum
    t["bias"] = (pred_sum - act_sum) / act_sum.where(act_sum > 0)
    t = t.join(ctx["stats"].drop(columns=["item_id", "store_id"]))
    t["zero_bucket"] = pd.cut(t["zero_share"], ZERO_BUCKETS, labels=ZERO_LABELS, include_lowest=True)
    return t


def error_share(table: pd.DataFrame, by: str) -> pd.DataFrame:
    """Share of the bottom-level WRMSSE vs share of the dollar weight, per group.
    over_index > 1: the group contributes more error than its weight."""
    g = table.groupby(by, observed=True)
    out = pd.DataFrame({
        "series": g.size(),
        "weight_share": g["weight"].sum(),
        "error_share": g["contribution"].sum() / table["contribution"].sum(),
        "wrmsse": g["contribution"].sum() / g["weight"].sum(),  # weighted mean RMSSE in the group
        "wrmsse_snaive": g.apply(lambda d: (d["weight"] * d["rmsse_snaive"]).sum() / d["weight"].sum()),
    })
    out["over_index"] = out["error_share"] / out["weight_share"]
    return out.sort_values("error_share", ascending=False)


def top_series(table: pd.DataFrame, ctx: dict, n: int = 20) -> pd.DataFrame:
    top = table.nlargest(n, "contribution").copy()
    e = (ctx["actual"].loc[top.index] - ctx["pred"].loc[top.index]) ** 2
    is_event = ctx["actual"].columns.isin(ctx["events"]["date"])
    top["event_day_sq_err_share"] = e.loc[:, is_event].sum(axis=1) / e.sum(axis=1)
    cols = ["dept_id", "store_id", "share", "weight", "rmsse", "rmsse_snaive", "actual_28d", "pred_28d",
            "bias", "mean_sales", "zero_share", "price_changes_val", "event_day_sq_err_share"]
    return top[cols]


def error_by_horizon(ctx: dict, level: str = "item_store") -> pd.DataFrame:
    """Per forecast day: weighted mean absolute scaled error sum_i w_i |e_ih| / sqrt(scale_i)
    and the signed version (bias; > 0 = over-forecast), for LightGBM and sNaive."""
    ev = ctx["evaluator"]
    S, scale, weight = ev.levels[level]
    actual = ev._align(ctx["actual"])
    out = pd.DataFrame(index=pd.RangeIndex(1, ev.horizon + 1, name="h"))
    out["date"] = ctx["actual"].columns
    out["dow"] = out["date"].dt.day_name().str[:3]
    out["event"] = out["date"].isin(ctx["events"]["date"])
    for name, fc in [("lgbm", ctx["pred"]), ("snaive", ctx["snaive"])]:
        scaled = (S @ (ev._align(fc) - actual)) / np.sqrt(scale)[:, None]
        out[f"{name}_abs"] = weight @ np.abs(scaled)
        out[f"{name}_bias"] = weight @ scaled
    return out


def error_by_dow(horizon_df: pd.DataFrame) -> pd.DataFrame:
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    cols = ["lgbm_abs", "snaive_abs", "lgbm_bias"]
    return horizon_df.groupby("dow")[cols].mean().reindex(order)


def zero_share_buckets(table: pd.DataFrame) -> pd.DataFrame:
    return error_share(table, "zero_bucket").reindex(ZERO_LABELS)


def shap_sample(name: str, start, end, n: int = 20_000, seed: int = 0):
    """TreeSHAP values (LightGBM pred_contrib, raw/log scale) for n sampled val rows."""
    booster = lgb.Booster(model_file=str(ARTIFACTS_DIR / f"lgbm_{name}.txt"))
    features = booster.feature_name()
    X, _, _ = load_matrix(start, end, features)
    idx = np.random.default_rng(seed).choice(len(X), size=min(n, len(X)), replace=False)
    Xs = X[idx]
    contrib = booster.predict(Xs, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=features), pd.DataFrame(Xs, columns=features)
