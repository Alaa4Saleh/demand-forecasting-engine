"""Drift monitoring: is the data the model sees now still like the data it was trained on?

Compares the training reference window (2015-01-01 -> 2016-02-28, a fixed-seed random sample of rows)
with two current windows:
    recent_actuals   the last 28 days of actuals (2016-03-28 -> 2016-04-24)
    forecast_inputs  the feature rows the served forecast is computed from (2016-04-25 -> 2016-05-22)
Each current window is also compared with the same 28 calendar days one year earlier, to separate
seasonality from genuine drift.

    feature drift     PSI (flag > 0.2, watch > 0.1) and KS statistic (context only) per numeric feature
    target drift      mean daily units and zero-sale share per category, reference vs recent actuals
    prediction drift  forecast mean vs recent actual mean per category (flag if the gap is > 15%)

Writes data/serving/drift_report.json (read by the API's /drift endpoint).
Run:  python -m src.monitoring
"""
import argparse
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data import HORIZON, load_sell_prices
from src.features import FEATURES_DIR, build_store_features, category_levels, load_calendar
from src.forecast import SERVING_DIR

REFERENCE = ("2015-01-01", "2016-02-28")
SAMPLE_FRACTION = 0.10   # reference rows kept per store (~1.3M rows in total)
SEED = 42                # fixed sampling seed: the same reference sample on every run
BINS = 10
PSI_FLAG, PSI_WATCH = 0.2, 0.1
PREDICTION_GAP_PCT = 15.0
DRIFT_FEATURES = ["lag_28", "lag_35", "lag_42",
                  "rmean_7", "rstd_7", "rmean_14", "rstd_14", "rmean_28", "rstd_28", "zero_streak_28",
                  "sell_price", "price_change", "price_rel_cat", "price_rmean_4w", "price_vs_rmean_4w"]
REPORT_PATH = SERVING_DIR / "drift_report.json"


# --- Pure functions (unit-tested) ---------------------------------------------------------------

def psi(reference, current, bins: int = BINS, eps: float = 1e-4) -> float:
    """Population Stability Index of `current` against `reference`.

    Bins are cut at the reference quantiles; repeated edges are merged (zero-inflated features
    such as lags give few distinct quantiles) and missing values get their own bin.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref_ok, cur_ok = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref_ok):
        edges = np.unique(np.quantile(ref_ok, np.linspace(0, 1, bins + 1)[1:-1]))
    else:
        edges = np.array([])

    def shares(x, x_ok):
        counts = np.bincount(np.searchsorted(edges, x_ok, side="right"), minlength=len(edges) + 1)
        counts = np.append(counts, len(x) - len(x_ok))  # NaN bin
        return np.maximum(counts / max(len(x), 1), eps)

    r, c = shares(ref, ref_ok), shares(cur, cur_ok)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_status(value: float) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return "flagged" if value > PSI_FLAG else "watch" if value > PSI_WATCH else "ok"


def feature_drift(reference: pd.DataFrame, windows: dict, features=DRIFT_FEATURES) -> pd.DataFrame:
    """One row per (feature, window). `windows` maps name -> (current frame, same-days-last-year frame or None)."""
    rows = []
    for name, (current, last_year) in windows.items():
        for f in features:
            ref, cur = reference[f].to_numpy(np.float64), current[f].to_numpy(np.float64)
            ref_ok, cur_ok = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
            ks = ks_2samp(ref_ok, cur_ok) if len(ref_ok) and len(cur_ok) else None
            p = psi(ref, cur)
            rows.append({
                "feature": f, "window": name, "psi": p,
                "psi_vs_last_year": psi(last_year[f], cur) if last_year is not None else None,
                "ks_stat": float(ks.statistic) if ks else None, "ks_pvalue": float(ks.pvalue) if ks else None,
                "ref_mean": float(np.nanmean(ref)) if len(ref_ok) else None,
                "cur_mean": float(np.nanmean(cur)) if len(cur_ok) else None,
                "status": psi_status(p),
            })
    return pd.DataFrame(rows)


def target_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Mean daily units per series and zero-sale share, per category and in total (frames: cat_id, sold)."""
    def summary(df):
        g = df.groupby(df["cat_id"].astype(str))["sold"]
        out = pd.DataFrame({"mean": g.mean(), "zero_share": g.apply(lambda s: (s == 0).mean())})
        out.loc["Total"] = [df["sold"].mean(), (df["sold"] == 0).mean()]
        return out
    r, c = summary(reference), summary(current)
    out = pd.DataFrame({
        "category": r.index,
        "ref_mean_daily": r["mean"].to_numpy(), "cur_mean_daily": c["mean"].reindex(r.index).to_numpy(),
        "ref_zero_share": r["zero_share"].to_numpy(), "cur_zero_share": c["zero_share"].reindex(r.index).to_numpy(),
    })
    out["change_pct"] = 100 * (out["cur_mean_daily"] / out["ref_mean_daily"] - 1)
    out["zero_share_change_pp"] = 100 * (out["cur_zero_share"] - out["ref_zero_share"])
    return out


def prediction_drift(forecasts: pd.DataFrame, actuals: pd.DataFrame, threshold: float = PREDICTION_GAP_PCT) -> pd.DataFrame:
    """Mean daily forecast vs mean daily recent actual per series, per category and in total.
    forecasts: cat_id, forecast; actuals: cat_id, sales."""
    def means(df, col):
        m = df.groupby(df["cat_id"].astype(str))[col].mean()
        m.loc["Total"] = df[col].mean()
        return m
    f, a = means(forecasts, "forecast"), means(actuals, "sales")
    out = pd.DataFrame({"category": f.index, "forecast_mean_daily": f.to_numpy(),
                        "recent_actual_mean_daily": a.reindex(f.index).to_numpy()})
    out["gap_pct"] = 100 * (out["forecast_mean_daily"] / out["recent_actual_mean_daily"] - 1)
    out["status"] = np.where(out["gap_pct"].abs() > threshold, "flagged", "ok")
    return out


# --- Data loading and report ---------------------------------------------------------------------

def _windows(data_end: pd.Timestamp) -> dict:
    recent = (data_end - pd.Timedelta(days=HORIZON - 1), data_end)
    future = (data_end + pd.Timedelta(days=1), data_end + pd.Timedelta(days=HORIZON))
    year = pd.DateOffset(years=1)
    return {"recent_actuals": (recent, (recent[0] - year, recent[1] - year)),
            "forecast_inputs": (future, (future[0] - year, future[1] - year))}


def collect(include_forecast_inputs: bool = True):
    """One pass per store: reference sample, current windows and their last-year windows."""
    levels = category_levels()
    rng = np.random.default_rng(SEED)
    prices, calendar = (load_sell_prices(), load_calendar()) if include_forecast_inputs else (None, None)
    cols = ["date", "sold", "cat_id", *DRIFT_FEATURES]
    ref_parts, target_ref, parts = [], [], {}
    data_end = None
    for store in levels["store_id"]:
        t = time.time()
        if include_forecast_inputs:
            df = build_store_features(store, prices, levels, calendar, future_days=HORIZON)[cols]
        else:
            df = pd.read_parquet(FEATURES_DIR / f"{store}.parquet", columns=cols)
        data_end = df.loc[df["sold"].notna(), "date"].max()
        in_ref = df["date"].between(*map(pd.Timestamp, REFERENCE))
        ref_rows = df[in_ref]
        ref_parts.append(ref_rows[rng.random(len(ref_rows)) < SAMPLE_FRACTION])
        target_ref.append(ref_rows[["cat_id", "sold"]])
        for name, ((s, e), (ls, le)) in _windows(data_end).items():
            if name == "forecast_inputs" and not include_forecast_inputs:
                continue
            parts.setdefault(name, []).append(df[df["date"].between(s, e)])
            parts.setdefault(name + "_last_year", []).append(df[df["date"].between(ls, le)])
        print(f"{store}: {time.time() - t:.0f}s", flush=True)
    cat = lambda xs: pd.concat(xs, ignore_index=True)
    return data_end, cat(ref_parts), cat(target_ref), {k: cat(v) for k, v in parts.items()}


def build_report(include_forecast_inputs: bool = True) -> dict:
    data_end, reference, target_ref, w = collect(include_forecast_inputs)
    spans = _windows(data_end)
    names = [n for n in spans if n in w]

    feats = feature_drift(reference, {n: (w[n], w[n + "_last_year"]) for n in names})
    target = target_drift(target_ref, w["recent_actuals"][["cat_id", "sold"]])

    fc = pd.read_parquet(SERVING_DIR / "forecasts.parquet", columns=["id", "cat_id", "forecast"])
    act = pd.read_parquet(SERVING_DIR / "actuals_last28.parquet")
    cat_of = fc.drop_duplicates("id").set_index("id")["cat_id"]
    act["cat_id"] = act["id"].map(cat_of)
    pred = prediction_drift(fc, act)
    meta = json.loads((SERVING_DIR / "metadata.json").read_text(encoding="utf-8"))

    day = lambda t: str(pd.Timestamp(t).date())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": meta.get("model"),
        "reference": {"start": REFERENCE[0], "end": REFERENCE[1], "sample_fraction": SAMPLE_FRACTION,
                      "seed": SEED, "rows_sampled": len(reference)},
        "windows": {n: {"start": day(spans[n][0][0]), "end": day(spans[n][0][1]),
                        "last_year_start": day(spans[n][1][0]), "last_year_end": day(spans[n][1][1]),
                        "rows": len(w[n])} for n in names},
        "thresholds": {"psi_flag": PSI_FLAG, "psi_watch": PSI_WATCH, "prediction_gap_pct": PREDICTION_GAP_PCT},
        "features": _records(feats),
        "target": _records(target),
        "prediction": _records(pred),
        "flagged": [f"{r.feature}@{r.window}" for r in feats.itertuples() if r.status == "flagged"],
    }


def _records(df: pd.DataFrame) -> list:
    """JSON-safe records: NaN -> None, numpy scalars -> Python."""
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v.item() if hasattr(v, "item") else v)
             for k, v in row.items()} for row in df.to_dict("records")]


def print_summary(report: dict) -> None:
    feats = pd.DataFrame(report["features"])
    print(f"\nReference {report['reference']['start']}..{report['reference']['end']} "
          f"({report['reference']['rows_sampled']:,} sampled rows, seed {report['reference']['seed']})")
    for name, win in report["windows"].items():
        f = feats[feats["window"] == name].set_index("feature")
        counts = f["status"].value_counts().to_dict()
        print(f"\n{name} {win['start']}..{win['end']}: {counts}")
        print(f[["psi", "psi_vs_last_year", "ks_stat", "ref_mean", "cur_mean", "status"]]
              .sort_values("psi", ascending=False).round(3).to_string())
    print("\nTarget (mean daily units per series, zero share):")
    print(pd.DataFrame(report["target"]).round(3).to_string(index=False))
    print("\nPrediction vs recent actuals:")
    print(pd.DataFrame(report["prediction"]).round(3).to_string(index=False))
    print(f"\nFlagged (PSI > {PSI_FLAG}): {report['flagged'] or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-forecast-inputs", action="store_true",
                    help="skip the forecast-input window (reads stored features; faster)")
    args = ap.parse_args()
    report = build_report(include_forecast_inputs=not args.no_forecast_inputs)
    REPORT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print_summary(report)
    print(f"\nwrote {REPORT_PATH} ({REPORT_PATH.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
