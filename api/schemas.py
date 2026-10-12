"""Response models for the forecasting API."""
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

SIMULATED_NOTE = (
    "Demand surge vs recent run-rate (simulated stock). The M5 dataset has no inventory data: stock is "
    "assumed to equal days_of_cover x the series' average daily sales over the last 28 actual days, and a "
    "series is flagged when forecast demand over the 28-day forecast window exceeds that simulated stock by "
    "more than min_shortfall_pct percent. With days_of_cover of 28 or more this isolates series whose forecast "
    "runs above their recent sales rate; below 28 most steadily selling series qualify by construction. "
    "Series with no recent sales are not evaluated. Flags can also reflect model errors (e.g. a short sales "
    "spike that is still inside the >= 28-day-lagged features), not only genuine demand changes."
)


class Health(BaseModel):
    status: Literal["ok"]
    model: str
    best_iteration: int
    val_wrmsse: Optional[float] = Field(None, description="WRMSSE on the held-out validation window")
    forecast_start: date
    forecast_end: date
    history_start: date
    history_end: date
    n_series: int
    generated_at: str


class Item(BaseModel):
    item_id: str
    dept_id: str
    cat_id: str


class Catalog(BaseModel):
    stores: list[str]
    states: list[str]
    categories: list[str]
    departments: list[str]
    items: list[Item]


class DailyValue(BaseModel):
    date: date
    units: float


class Forecast(BaseModel):
    item_id: str
    store_id: str
    dept_id: str
    cat_id: str
    state_id: str
    sell_price: float = Field(description="Latest known shelf price (USD)")
    forecast_total: float
    forecast: list[DailyValue]
    history: list[DailyValue] = Field(description="Actual daily sales, last 28 days before the forecast")


class GroupTotal(BaseModel):
    key: str
    forecast_total: float = Field(description="Forecast units over the forecast window")
    history_total: float = Field(description="Actual units over the last 28 days")
    change_pct: Optional[float] = Field(None, description="Forecast vs last 28 days, in percent")


class Aggregate(BaseModel):
    level: Literal["store", "cat", "dept", "state"]
    key: Optional[str] = Field(None, description="Group shown in the daily series; all groups summed if empty")
    groups: list[GroupTotal]
    forecast: list[DailyValue]
    history: list[DailyValue]


class Alert(BaseModel):
    item_id: str
    store_id: str
    dept_id: str
    cat_id: str
    sell_price: float
    recent_daily_sales: float = Field(description="Average daily units over the last 28 actual days")
    simulated_stock: float = Field(description="SIMULATED: days_of_cover x recent_daily_sales")
    forecast_demand: float = Field(description="Forecast units over the forecast window")
    shortfall_units: float
    shortfall_pct: float = Field(description="Shortfall as a percentage of the simulated stock")
    est_lost_revenue: float = Field(description="shortfall_units x sell_price (USD)")
    stockout_date: date = Field(description="First day cumulative forecast demand exceeds the simulated stock")


class Alerts(BaseModel):
    simulated: Literal[True] = True
    note: str = SIMULATED_NOTE
    days_of_cover: int
    min_shortfall_pct: float
    forecast_start: date
    forecast_end: date
    n_evaluated: int = Field(description="Series evaluated (with sales in the last 28 days, after filters)")
    n_at_risk: int
    alerts: list[Alert] = Field(description="At-risk series, highest estimated lost revenue first")
