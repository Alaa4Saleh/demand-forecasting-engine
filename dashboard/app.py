"""Demand forecast dashboard. Reads everything from the forecasting API (API_URL), never from files.

Run:  streamlit run dashboard/app.py      (API at http://localhost:8000 unless API_URL is set)
"""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

STATE_NAMES = {"CA": "California", "TX": "Texas", "WI": "Wisconsin"}
LEVELS = {"Store": "store", "Category": "cat", "Department": "dept", "State": "state"}

st.set_page_config(page_title="Demand Forecast", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def api(path: str, **params):
    r = requests.get(f"{API_URL}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=30)
    r.raise_for_status()
    return r.json()


def store_label(store_id: str) -> str:
    return f"{store_id} ({STATE_NAMES.get(store_id.split('_')[0], '')})"


def nice(code: str) -> str:
    """FOODS_3 -> Foods 3, HOUSEHOLD -> Household (M5 has codes, not product names)."""
    return code.replace("_", " ").title()


def daily_frame(points, label):
    return pd.DataFrame([{"Date": pd.Timestamp(p["date"]), "Units": p["units"], "Series": label} for p in points])


def history_vs_forecast_chart(history, forecast, title):
    df = pd.concat([daily_frame(history, "Actual sales (last 28 days)"),
                    daily_frame(forecast, "Forecast (next 28 days)")])
    return (alt.Chart(df, title=title).mark_line(point=True)
            .encode(x=alt.X("Date:T", title=None), y=alt.Y("Units:Q", title="Units sold per day"),
                    color=alt.Color("Series:N", title=None,
                                    scale=alt.Scale(range=["#4c78a8", "#f58518"]),
                                    legend=alt.Legend(orient="top")),
                    tooltip=[alt.Tooltip("Date:T"), "Series:N", alt.Tooltip("Units:Q", format=",.1f")])
            .properties(height=360))


# --- Header and global filters -------------------------------------------------------------------
try:
    health = api("/health")
    catalog = api("/catalog")
except requests.RequestException as e:
    st.error(f"The forecasting service at {API_URL} is not reachable ({e}). Start the API and reload.")
    st.stop()

st.title("Demand Forecast")
st.caption(f"Forecast period **{health['forecast_start']} to {health['forecast_end']}** for "
           f"{health['n_series']:,} product-store combinations · compared with actual sales "
           f"{health['history_start']} to {health['history_end']} · model accuracy on held-out data: "
           f"WRMSSE {health['val_wrmsse']} (lower is better)")

with st.sidebar:
    st.header("Filters")
    store = st.selectbox("Store", [None] + catalog["stores"],
                         format_func=lambda s: "All stores" if s is None else store_label(s))
    category = st.selectbox("Category", [None] + catalog["categories"],
                            format_func=lambda c: "All categories" if c is None else nice(c))

tab_item, tab_totals, tab_risk = st.tabs(["Product forecast", "Store & category totals", "Stock risk (simulated)"])

# --- Product forecast ----------------------------------------------------------------------------
with tab_item:
    if store is None:
        st.info("Choose a store in the sidebar to see product-level forecasts.")
    else:
        items = api("/catalog", store_id=store, cat_id=category)["items"]
        item = st.selectbox(f"Product ({len(items):,} available)", [i["item_id"] for i in items],
                            format_func=lambda i: f"{i} ({nice(i.rsplit('_', 1)[0])})")
        fc = api("/forecast", item_id=item, store_id=store)
        hist_total = sum(p["units"] for p in fc["history"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Forecast, next 28 days", f"{fc['forecast_total']:,.0f} units")
        c2.metric("Actual, last 28 days", f"{hist_total:,.0f} units",)
        c3.metric("Change", f"{100 * (fc['forecast_total'] / hist_total - 1):+.1f}%" if hist_total else "n/a")
        c4.metric("Shelf price", f"${fc['sell_price']:,.2f}")
        st.altair_chart(history_vs_forecast_chart(fc["history"], fc["forecast"],
                                                  f"{item} at {store_label(store)}"), use_container_width=True)

# --- Totals by level -----------------------------------------------------------------------------
with tab_totals:
    level_name = st.radio("Group by", list(LEVELS), horizontal=True)
    agg = api("/aggregate", level=LEVELS[level_name])
    groups = pd.DataFrame(agg["groups"])
    label = store_label if level_name == "Store" else (lambda k: STATE_NAMES.get(k, k)) if level_name == "State" else nice
    groups[level_name] = groups["key"].map(label)
    bars = groups.melt(id_vars=[level_name], value_vars=["history_total", "forecast_total"],
                       var_name="Period", value_name="Units")
    bars["Period"] = bars["Period"].map({"history_total": "Actual, last 28 days", "forecast_total": "Forecast, next 28 days"})
    st.altair_chart(alt.Chart(bars, title=f"Units sold by {level_name.lower()}")
                    .mark_bar().encode(x=alt.X(f"{level_name}:N", title=None, sort="-y"),
                                       xOffset="Period:N", y=alt.Y("Units:Q", title="Units"),
                                       color=alt.Color("Period:N", title=None, scale=alt.Scale(range=["#4c78a8", "#f58518"]),
                                                       legend=alt.Legend(orient="top")),
                                       tooltip=[f"{level_name}:N", "Period:N", alt.Tooltip("Units:Q", format=",.0f")])
                    .properties(height=320), use_container_width=True)

    key = st.selectbox(f"Daily view for one {level_name.lower()}", [None] + list(groups["key"]),
                       format_func=lambda k: f"All {level_name.lower()}s combined" if k is None else label(k))
    one = api("/aggregate", level=LEVELS[level_name], key=key)
    st.altair_chart(history_vs_forecast_chart(one["history"], one["forecast"],
                                              "All combined" if key is None else label(key)), use_container_width=True)
    table = groups.rename(columns={"history_total": "Actual, last 28 days", "forecast_total": "Forecast, next 28 days",
                                   "change_pct": "Change (%)"})[[level_name, "Actual, last 28 days",
                                                                  "Forecast, next 28 days", "Change (%)"]]
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"Actual, last 28 days": st.column_config.NumberColumn(format="%.0f"),
                                "Forecast, next 28 days": st.column_config.NumberColumn(format="%.0f"),
                                "Change (%)": st.column_config.NumberColumn(format="%+.1f")})

# --- Simulated stock risk ------------------------------------------------------------------------
with tab_risk:
    st.warning("**Simulated inventory.** The M5 data contains no stock levels. Each product is assumed to hold "
               "stock equal to the chosen number of days of its average daily sales over the last 28 days. "
               "Products are flagged when forecast demand for the next 28 days is higher than that stock. "
               "Below 28 days of cover, almost every steadily selling product is flagged by construction.")
    c1, c2 = st.columns([3, 1])
    days = c1.slider("Simulated stock on hand (days of recent sales)", min_value=7, max_value=90, value=35, step=1)
    top_n = c2.number_input("Products to show", min_value=10, max_value=500, value=50, step=10)
    risk = api("/alerts", days_of_cover=days, store_id=store, cat_id=category, limit=int(top_n))
    rows = pd.DataFrame(risk["alerts"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Products at risk", f"{risk['n_at_risk']:,}")
    m2.metric("Share of products checked", f"{100 * risk['n_at_risk'] / max(risk['n_evaluated'], 1):.1f}%")
    m3.metric(f"Est. lost revenue, top {len(rows)}", f"${rows['est_lost_revenue'].sum():,.0f}" if len(rows) else "$0")
    if rows.empty:
        st.success("No products at risk with this much simulated stock.")
    else:
        rows["Store"] = rows["store_id"].map(store_label)
        rows["Category"] = rows["cat_id"].map(nice)
        shown = rows.rename(columns={
            "item_id": "Product", "recent_daily_sales": "Avg daily sales (last 28 days)",
            "simulated_stock": "Simulated stock (units)", "forecast_demand": "Forecast demand, 28 days (units)",
            "shortfall_units": "Shortfall (units)", "est_lost_revenue": "Est. lost revenue ($)",
            "stockout_date": "Projected stock-out date", "sell_price": "Shelf price ($)"})
        cols = ["Product", "Store", "Category", "Shelf price ($)", "Avg daily sales (last 28 days)",
                "Simulated stock (units)", "Forecast demand, 28 days (units)", "Shortfall (units)",
                "Est. lost revenue ($)", "Projected stock-out date"]
        st.dataframe(shown[cols], hide_index=True, use_container_width=True, column_config={
            "Shelf price ($)": st.column_config.NumberColumn(format="$%.2f"),
            "Avg daily sales (last 28 days)": st.column_config.NumberColumn(format="%.1f"),
            "Simulated stock (units)": st.column_config.NumberColumn(format="%.0f"),
            "Forecast demand, 28 days (units)": st.column_config.NumberColumn(format="%.0f"),
            "Shortfall (units)": st.column_config.NumberColumn(format="%.0f"),
            "Est. lost revenue ($)": st.column_config.NumberColumn(format="$%.0f"),
        })
    st.caption(risk["note"])
