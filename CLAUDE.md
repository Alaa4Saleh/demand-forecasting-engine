# Demand Forecasting & Inventory Optimization Engine

Portfolio project on Kaggle M5 data (hierarchical retail demand forecasting).
Goal: end-to-end system - baselines -> LightGBM -> PySpark pipeline -> FastAPI + Streamlit -> drift monitoring.
Environment: Windows, PyCharm, venv, Python 3.11. Evaluation metric: WRMSSE.

## Data pipeline (done)
- data/raw: sales_train_validation.csv, calendar.csv, sell_prices.csv
- Scope: 3,049 items x 10 stores (CA, TX, WI) = 30,490 series; 2011-01-29 -> 2016-04-24.
- notebooks/01_first_look.ipynb: first look; melt sales to long format, merge calendar.
- notebooks/02_eda_demand_patterns.ipynb: repeats melt/merge, casts to category/int32, adds is_store_closed
  (Dec 25), trims rows before item release week (first week with a price), and WRITES the parquet (cells 13-15).
- Result: data/processed/sales_long_clean.parquet, shape (46,027,957 x 20). Load it instead of rebuilding.
- Findings: 59.7% zero-sale rows after trim (HOBBIES most intermittent); Walmart closed every Dec 25.

## Modeling status
- Split by time: val = last 28 days (2016-03-28 -> 2016-04-24), train = everything before.
- Baselines (global, not WRMSSE): Naive RMSE 3.301 / MAE 1.350; Seasonal Naive RMSE 2.759 / MAE 1.232.
- WRMSSE (done): src/metrics.py `WRMSSEEvaluator.from_long(train, load_sell_prices())`, 12 M5 levels /
  42,840 series; scale = train one-step diffs from first non-zero sale; weights = last-28-train-days dollar sales.
- Baselines WRMSSE (val 2016-03-28 -> 2016-04-24): Naive 1.4864, Seasonal Naive 0.9228 (bar to beat).
- Reusable code: src/data.py (load_sales_long, time_split, to_wide), src/models/baselines.py.
  Load only needed columns, e.g. load_sales_long(EVAL_COLUMNS) - item_id/store_id are large_string.
- Features (done): src/features.py -> data/processed/features/<store>.parquet (gitignored; rebuild with
  `python -m src.features`, ~2 min). Feature sets (FEATURE_SETS):
  - base: lags 28/35/42, rolling mean/std 7/14/28 on sold shifted 28, calendar (dow, month, events, snap,
    is_store_closed), price (sell_price, week-over-week change, price / store-cat-week mean).
  - extended = base + mday, week_of_year, price_rmean_4w, price_vs_rmean_4w, zero_streak_28 (zero run ending at t-28).
  All sales-derived features look >= 28 days back.
- Global LightGBM (src/models/lgbm.py): training data streamed per store via lgb.Sequence (23M rows fit in RAM).
  Windows: fit train_start -> 2016-02-28; inner 2016-02-29 -> 2016-03-27 (early stopping + config selection by
  inner WRMSSE, evaluator built from data <= 2016-02-28); val 2016-03-28 -> 2016-04-24 scored ONLY with
  --score-val for the final config. Runs log to artifacts/experiments.jsonl; models in artifacts/ (gitignored).

### Experiment log (selection on inner WRMSSE; greedy, each step keeps the best so far)
Inner-window reference: Naive 1.6956, Seasonal Naive 0.8889.

| Exp | Change vs previous best | Fit rows | Best iter | Inner WRMSSE | Kept |
|---|---|---|---|---|---|
| E0 | lr 0.05, l2 0.1, min_leaf 100, patience 100, 2015 start, base features | 12.8M | 78 | 0.7177 | ref |
| E1 | lr 0.02, lambda_l2 1.0, min_data_in_leaf 500, patience 200 | 12.8M | 304 | 0.6904 | yes |
| E2 | train from 2014-01-01 | 23.1M | 304 | 0.6807 | yes |
| E3 | extended features | 23.1M | 351 | 0.6535 | yes (final) |
| E4 | tweedie_variance_power 1.3 (vs 1.1) | 23.1M | 345 | 0.6553 | no |

Final config = E3: `python -m src.models.lgbm --name final --features extended --train-start 2014-01-01
--lr 0.02 --lambda-l2 1.0 --min-data-in-leaf 500 --patience 200 --score-val` (~8 min; reproduces E3 exactly).
Top gain: rmean_28 75%, item_id 7%, rmean_14 7%, rstd_28 4%; new features each <= 0.5%.
Val predictions: artifacts/lgbm_final_val_pred.parquet (input to error analysis).

WRMSSE per level, scored val window 2016-03-28 -> 2016-04-24 (LightGBM v1 = first model, 2015 start, lr 0.05):

| Level | Naive | Seasonal Naive | LightGBM v1 | LightGBM final |
|---|---|---|---|---|
| total | 1.3843 | 0.6677 | 0.4133 | 0.3029 |
| state | 1.6089 | 0.7245 | 0.4730 | 0.3875 |
| store | 1.6322 | 0.8252 | 0.5184 | 0.4698 |
| cat | 1.4183 | 0.7543 | 0.4813 | 0.3516 |
| dept | 1.5939 | 0.9738 | 0.5904 | 0.4572 |
| state_cat | 1.5719 | 0.8347 | 0.5516 | 0.4560 |
| state_dept | 1.6359 | 0.9844 | 0.6530 | 0.5447 |
| store_cat | 1.6297 | 0.9205 | 0.6140 | 0.5554 |
| store_dept | 1.6808 | 1.0290 | 0.6994 | 0.6277 |
| item | 1.2699 | 1.1143 | 0.8891 | 0.8603 |
| item_state | 1.2361 | 1.1192 | 0.8732 | 0.8592 |
| item_store | 1.1752 | 1.1259 | 0.8527 | 0.8461 |
| WRMSSE | 1.4864 | 0.9228 | 0.6341 | 0.5599 |

## Conventions
- Never random split; always time-based. Avoid leakage in lag/target-encoding features.
- Use observed=True in groupby on category columns. Memory is tight (46M rows): keep category/int32 dtypes.
- Move reusable logic from notebooks into src/ modules; keep notebooks for exploration.
- Commit small, documented steps; update README with findings as we go.

## Next steps
1. ~~Finish WRMSSE and evaluate both baselines.~~ Done.
2. ARIMA/Prophet on category level, Croston for intermittent items.
3. ~~Feature engineering (lags, rolling stats, calendar, price) and global LightGBM.~~ Done (0.6341).
4. ~~LightGBM tuning: regularization/lower lr, 2014 start, extra features, tweedie power.~~ Done (0.5599).
5. Error analysis (src/analysis.py, notebooks/04_error_analysis.ipynb) - in progress.
6. Remaining LightGBM ideas: refit through 2016-03-27 at the chosen iteration count, per-store/per-dept models.
