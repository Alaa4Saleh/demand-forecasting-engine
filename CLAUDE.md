# Demand Forecasting & Inventory Optimization Engine

Portfolio project on Kaggle M5 data (hierarchical retail demand forecasting).
Goal: end-to-end system - baselines -> LightGBM -> PySpark pipeline -> FastAPI + Streamlit -> drift monitoring.
Environment: Windows, PyCharm, venv, Python 3.11. Evaluation metric: WRMSSE.

## Data pipeline (done)
- data/raw: sales_train_validation.csv, calendar.csv, sell_prices.csv
- notebooks/01_first_look.ipynb: melt sales to long format, merge calendar, cast to category/int32,
  add is_store_closed (Dec 25), trim rows before item release week (first week with a price).
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
  `python -m src.features`, ~2 min). Lags 28/35/42, rolling mean/std 7/14/28 on sold shifted 28,
  calendar, price (sell_price, week-over-week change, price / store-cat-week mean). All sales features >= 28 days back.
- Global LightGBM (src/models/lgbm.py, `python -m src.models.lgbm --train-start 2015-01-01`): Tweedie 1.1,
  lr 0.05, 127 leaves; fit 2015-01-01 -> 2016-02-28 (12.8M rows), early stop on 2016-02-29 -> 2016-03-27,
  scored on val. Model saved to artifacts/lgbm_tweedie.txt (gitignored).
  Result: WRMSSE 0.6341 (vs sNaive 0.9228, Naive 1.4864); best_iteration 78 (stopped very early);
  bottom levels ~0.85-0.89, aggregate levels 0.41-0.70. Gain dominated by rmean_28 (70%), then rmean_7, item_id.

## Conventions
- Never random split; always time-based. Avoid leakage in lag/target-encoding features.
- Use observed=True in groupby on category columns. Memory is tight (46M rows): keep category/int32 dtypes.
- Move reusable logic from notebooks into src/ modules; keep notebooks for exploration.
- Commit small, documented steps; update README with findings as we go.

## Next steps
1. ~~Finish WRMSSE and evaluate both baselines.~~ Done.
2. ARIMA/Prophet on category level, Croston for intermittent items.
3. ~~Feature engineering (lags, rolling stats, calendar, price) and global LightGBM.~~ Done (0.6341).
4. LightGBM follow-ups: why it stops at ~78 iterations (lower lr / more regularization), refit on full
   train at best_iteration, train from 2014, per-store models.
