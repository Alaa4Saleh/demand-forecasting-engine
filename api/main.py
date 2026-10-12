"""Forecasting API over the files written by `python -m src.forecast`.

Run:  uvicorn api.main:app --port 8000      (data folder: SERVING_DIR, default data/serving)
Docs: http://localhost:8000/docs
"""
import json
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request

from api.data import ServingData
from api.schemas import (SIMULATED_NOTE, Aggregate, Alert, Alerts, Catalog, DailyValue, DriftReport, Forecast,
                         GroupTotal, Health, Item)

DEFAULT_SERVING_DIR = Path(__file__).resolve().parents[1] / "data" / "serving"


def _daily(dates, values) -> list[DailyValue]:
    return [DailyValue(date=d.date(), units=round(float(v), 3)) for d, v in zip(dates, values)]


def create_app(serving_dir: Optional[Path] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.serving_dir = Path(serving_dir or os.environ.get("SERVING_DIR", DEFAULT_SERVING_DIR))
        app.state.data = ServingData(app.state.serving_dir)
        yield

    app = FastAPI(
        title="Demand Forecasting API",
        description="28-day unit-sales forecasts for the M5 (Walmart) series from the global LightGBM model. "
                    "Alerts flag demand surges vs the recent run-rate against **simulated** stock - "
                    "M5 contains no inventory data.",
        version="1.0.0",
        lifespan=lifespan,
    )

    def data(request: Request) -> ServingData:
        return request.app.state.data

    @app.get("/health", response_model=Health)
    def health(request: Request):
        m = data(request).meta
        return Health(status="ok", **{k: m.get(k) for k in Health.model_fields if k != "status"})

    @app.get("/catalog", response_model=Catalog)
    def catalog(request: Request, store_id: Optional[str] = None, cat_id: Optional[str] = None):
        """Stores, states, categories, departments and items (optionally filtered by store / category)."""
        d = data(request)
        s = d.series
        for name, value in [("store_id", store_id), ("cat_id", cat_id)]:
            if value is not None:
                if value not in d.known[name]:
                    raise HTTPException(404, f"unknown {name} '{value}'")
                s = s[s[name] == value]
        items = s[["item_id", "dept_id", "cat_id"]].drop_duplicates("item_id").sort_values("item_id")
        return Catalog(stores=sorted(d.known["store_id"]), states=sorted(d.known["state_id"]),
                       categories=sorted(d.known["cat_id"]), departments=sorted(d.known["dept_id"]),
                       items=[Item(**r) for r in items.to_dict("records")])

    @app.get("/forecast", response_model=Forecast)
    def forecast(request: Request, item_id: str, store_id: str,
                 start: Optional[date] = Query(None, description="First forecast date (YYYY-MM-DD)"),
                 end: Optional[date] = Query(None, description="Last forecast date (YYYY-MM-DD)")):
        """Daily forecast for one item in one store, plus its last 28 days of actual sales."""
        d = data(request)
        if item_id not in d.known["item_id"]:
            raise HTTPException(404, f"unknown item_id '{item_id}'")
        if store_id not in d.known["store_id"]:
            raise HTTPException(404, f"unknown store_id '{store_id}'")
        i = d.row.get((item_id, store_id))
        if i is None:
            raise HTTPException(404, f"item '{item_id}' is not sold in store '{store_id}'")

        first, last = d.forecast_dates[0].date(), d.forecast_dates[-1].date()
        start, end = start or first, end or last
        if start > end:
            raise HTTPException(422, f"start {start} is after end {end}")
        if start < first or end > last:
            raise HTTPException(422, f"dates must be within the forecast window {first}..{last}")
        cols = (d.forecast_dates.date >= start) & (d.forecast_dates.date <= end)

        s = d.series.iloc[i]
        return Forecast(item_id=item_id, store_id=store_id, dept_id=s["dept_id"], cat_id=s["cat_id"],
                        state_id=s["state_id"], sell_price=round(float(s["sell_price"]), 2),
                        forecast_total=round(float(d.F[i, cols].sum()), 3),
                        forecast=_daily(d.forecast_dates[cols], d.F[i, cols]),
                        history=_daily(d.history_dates, d.H[i]))

    @app.get("/aggregate", response_model=Aggregate)
    def aggregate(request: Request, level: Literal["store", "cat", "dept", "state"],
                  key: Optional[str] = Query(None, description="e.g. CA_1, FOODS, FOODS_3, TX")):
        """Totals per group at one level, and the daily forecast vs actuals for one group (or all summed)."""
        d = data(request)
        groups = d.levels[level]
        if key is not None and key not in groups:
            raise HTTPException(404, f"unknown {level} '{key}'; valid: {sorted(groups)}")
        totals = []
        for g, (f, h) in groups.items():
            change = 100 * (f.sum() / h.sum() - 1) if h.sum() > 0 else None
            totals.append(GroupTotal(key=g, forecast_total=round(float(f.sum()), 1),
                                     history_total=round(float(h.sum()), 1),
                                     change_pct=None if change is None else round(float(change), 2)))
        f, h = groups[key] if key else (np.sum([v[0] for v in groups.values()], axis=0),
                                        np.sum([v[1] for v in groups.values()], axis=0))
        return Aggregate(level=level, key=key, groups=totals,
                         forecast=_daily(d.forecast_dates, f), history=_daily(d.history_dates, h))

    @app.get("/alerts", response_model=Alerts, description=SIMULATED_NOTE)
    def alerts(request: Request,
               days_of_cover: int = Query(..., ge=1, le=365, description="SIMULATED stock, in days of recent sales"),
               min_shortfall_pct: float = Query(20.0, ge=0, le=1000,
                                                description="Flag only if the shortfall exceeds this % of simulated stock"),
               store_id: Optional[str] = None, cat_id: Optional[str] = None,
               limit: int = Query(50, ge=1, le=1000)):
        """Demand surge vs recent run-rate (simulated stock), ranked by estimated lost revenue."""
        d = data(request)
        for name, value in [("store_id", store_id), ("cat_id", cat_id)]:
            if value is not None and value not in d.known[name]:
                raise HTTPException(404, f"unknown {name} '{value}'")
        a = d.alerts(days_of_cover, min_shortfall_pct, store_id, cat_id)
        rows = a.head(limit).round({"recent_daily_sales": 3, "simulated_stock": 2, "forecast_demand": 2,
                                    "shortfall_units": 2, "shortfall_pct": 1, "est_lost_revenue": 2,
                                    "sell_price": 2})
        return Alerts(days_of_cover=days_of_cover, min_shortfall_pct=min_shortfall_pct, forecast_start=d.forecast_dates[0].date(),
                      forecast_end=d.forecast_dates[-1].date(), n_evaluated=a.attrs["n_evaluated"],
                      n_at_risk=len(a), alerts=[Alert(**r) for r in rows.to_dict("records")])

    @app.get("/drift", response_model=DriftReport)
    def drift(request: Request):
        """Latest drift report (written by `python -m src.monitoring`; read on each request, so a new
        report is served without restarting the API)."""
        path = request.app.state.serving_dir / "drift_report.json"
        if not path.exists():
            raise HTTPException(404, "no drift report yet; run `python -m src.monitoring`")
        return json.loads(path.read_text(encoding="utf-8"))

    return app


app = create_app()
