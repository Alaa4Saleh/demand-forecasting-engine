# Demand Forecasting & Inventory Optimization Engine

End-to-end hierarchical retail demand forecasting on the Kaggle **M5** dataset (Walmart unit sales).
The goal is a complete system: baselines → LightGBM → PySpark pipeline → FastAPI + Streamlit → drift monitoring.
Models are evaluated with **WRMSSE**, the official M5 accuracy metric.

**Current result:** a global LightGBM model reaches **WRMSSE 0.6341** on a 28-day hold-out,
vs **0.9228** for Seasonal Naive and **1.4864** for Naive.

## Dataset

[M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) (Kaggle).
Daily unit sales for 3,049 products in 10 stores across 3 US states (CA, TX, WI), giving 30,490
item-store series from 2011-01-29 to 2016-04-24. Files used:

| File | Content |
|---|---|
| `sales_train_validation.csv` | Daily unit sales per item-store |
| `calendar.csv` | Dates, weekdays, events, SNAP days |
| `sell_prices.csv` | Weekly price per item-store |

Findings from EDA:
- After trimming each series to start at its release week, **59.7%** of rows are zero sales
  (HOBBIES is the most intermittent category).
- Stores are closed every December 25.

## Pipeline

1. **Cleaning** (notebooks 01–02): melt sales to long format, merge the calendar, cast to compact
   category/int32 dtypes, add `is_store_closed` (Dec 25), and drop rows before each item's release
   week (its first week with a price). Output: `data/processed/sales_long_clean.parquet`,
   46,027,957 rows × 20 columns.
2. **Split** (time-based, never random): validation = last 28 days (2016-03-28 → 2016-04-24),
   train = everything before.
3. **Metric** (`src/metrics.py`): WRMSSE over all 12 M5 aggregation levels (42,840 series).
   - Scale: mean squared one-step difference over train, from each series' first non-zero sale.
   - Weights: dollar sales (units × price) over the last 28 days of train.
   - Final score: the mean of the 12 level scores.
4. **Baselines** (`src/models/baselines.py`): Naive (repeat the last day) and Seasonal Naive (repeat the last week).
5. **Features** (`src/features.py`), built per store to limit memory:
   - Lags at 28, 35 and 42 days.
   - Rolling mean/std over 7, 14 and 28 days on sales shifted 28 days.
   - Calendar: day of week, month, events, SNAP, store closed.
   - Price: sell price, week-over-week change, price relative to the store-category mean.

   Every sales-derived feature looks at least 28 days back, so the whole 28-day horizon is
   forecast without leakage or recursive prediction.
6. **Global LightGBM** (`src/models/lgbm.py`): a single model for all series.
   - Objective: Tweedie (variance power 1.1), for intermittent count data.
   - Learning rate 0.05, 127 leaves.
   - Fit on 2015-01-01 → 2016-02-28 (12.8M rows).
   - Early stopping on the last 28 days of train (2016-02-29 → 2016-03-27). The scored validation
     window is never used for tuning.

## Results

WRMSSE on the validation window 2016-03-28 → 2016-04-24 (lower is better):

| Level | Naive | Seasonal Naive | LightGBM |
|---|---:|---:|---:|
| total | 1.3843 | 0.6677 | **0.4133** |
| state | 1.6089 | 0.7245 | **0.4730** |
| store | 1.6322 | 0.8252 | **0.5184** |
| cat | 1.4183 | 0.7543 | **0.4813** |
| dept | 1.5939 | 0.9738 | **0.5904** |
| state_cat | 1.5719 | 0.8347 | **0.5516** |
| state_dept | 1.6359 | 0.9844 | **0.6530** |
| store_cat | 1.6297 | 0.9205 | **0.6140** |
| store_dept | 1.6808 | 1.0290 | **0.6994** |
| item | 1.2699 | 1.1143 | **0.8891** |
| item_state | 1.2361 | 1.1192 | **0.8732** |
| item_store | 1.1752 | 1.1259 | **0.8527** |
| **WRMSSE** | **1.4864** | **0.9228** | **0.6341** |

Notes:
- LightGBM improves on both baselines at every level. The gain is largest at the aggregated
  levels (0.41–0.70) and smaller at the intermittent item levels (0.85–0.89).
- Feature gain is dominated by the 28-day rolling mean (~70%), followed by the 7-day rolling
  mean and `item_id`.
- Early stopping chose iteration 78, which is very early for a 0.05 learning rate.
  This is the first thing to investigate (see Roadmap).
- For reference, the global (non-WRMSSE) RMSE / MAE of the baselines is 3.301 / 1.350 for Naive
  and 2.759 / 1.232 for Seasonal Naive.

## Repository structure

```
├── data/
│   ├── raw/                  # M5 CSVs (not tracked)
│   └── processed/            # cleaned parquet + per-store features (not tracked)
├── notebooks/
│   ├── 01_first_look.ipynb           # initial exploration
│   ├── 02_eda_demand_patterns.ipynb  # EDA + cleaning, writes sales_long_clean.parquet
│   └── 03_baselines.ipynb            # time split, baselines, WRMSSE evaluation
├── src/
│   ├── data.py               # loaders, time split, long -> wide pivot
│   ├── metrics.py            # WRMSSEEvaluator (12-level M5 hierarchy)
│   ├── features.py           # lag / rolling / calendar / price features
│   └── models/
│       ├── baselines.py      # Naive, Seasonal Naive
│       └── lgbm.py           # global LightGBM: train, predict, evaluate
├── tests/test_metrics.py     # WRMSSE sanity checks
├── api/, dashboard/          # planned: FastAPI service, Streamlit app
└── requirements.txt
```

## How to reproduce

1. **Environment**

   ```bash
   python -m venv .venv
   ```
   ```bash
   .venv/Scripts/python -m pip install -r requirements.txt
   ```
   (`.venv/Scripts/` on Windows, `.venv/bin/` on Linux/macOS.)

2. **Data**: download the M5 files from the
   [Kaggle competition page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
   and put `sales_train_validation.csv`, `calendar.csv` and `sell_prices.csv` in `data/raw/`.

3. **Clean**: run `notebooks/02_eda_demand_patterns.ipynb` from the `notebooks/` folder. It
   writes `data/processed/sales_long_clean.parquet`.

4. **Features** (all 10 stores, about 2 minutes):

   ```bash
   python -m src.features
   ```

5. **Train and evaluate LightGBM**. This prints the best iteration, the top feature importances,
   and the per-level WRMSSE table for Naive, Seasonal Naive and LightGBM. The model is saved to
   `artifacts/lgbm_tweedie.txt`.

   ```bash
   python -m src.models.lgbm --train-start 2015-01-01
   ```

6. **Tests**

   ```bash
   python -m pytest tests
   ```

Memory: the full dataset is 46M rows, so the code keeps compact dtypes, builds features per
store, and loads only the columns it needs. Training from 2015-01-01 uses a 1.2 GB feature matrix.

## Roadmap

- [x] Data cleaning and EDA
- [x] WRMSSE metric and Naive / Seasonal Naive baselines
- [x] Feature engineering and global LightGBM (WRMSSE 0.6341)
- [ ] LightGBM follow-ups:
  - Investigate the early stop at iteration 78 (lower learning rate, more regularization).
  - Refit on the full train period.
  - Train from 2014.
  - Per-store models.
- [ ] Classical models: ARIMA / Prophet at category level, Croston for intermittent items
- [ ] Move the cleaning steps from the notebooks into a `src/` script
- [ ] PySpark feature pipeline
- [ ] FastAPI forecasting service and Streamlit dashboard
- [ ] Drift monitoring
