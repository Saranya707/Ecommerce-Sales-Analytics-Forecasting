"""E-Commerce Sales Analytics & Historical One-Day-Ahead Predictions."""

from pathlib import Path
import zipfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="E-Commerce Sales Dashboard", page_icon="📊", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "ecommerce_public.csv.zip"
MODEL_FILE = BASE_DIR / "sales_forecasting_model.pkl"
FEATURES_FILE = BASE_DIR / "model_features.pkl"
REQUIRED_COLUMNS = {"InvoiceDate", "Description", "Country", "TotalSales"}
TEST_START = "2011-09-30"
TEST_END = "2011-12-09"


@st.cache_data(show_spinner="Loading sales data...")
def load_data():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset missing: {DATA_FILE.name}")
    # Finder-created ZIP archives can contain an additional __MACOSX directory.
    with zipfile.ZipFile(DATA_FILE) as archive:
        csv_names = [name for name in archive.namelist()
                     if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected exactly one CSV in ZIP; found {csv_names}")
        with archive.open(csv_names[0]) as source:
            data = pd.read_csv(source, low_memory=False)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {', '.join(sorted(missing))}")
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data["TotalSales"] = pd.to_numeric(data["TotalSales"], errors="coerce")
    data = data.dropna(subset=["InvoiceDate", "TotalSales"]).copy()
    if data.empty:
        raise ValueError("Dataset has no usable dated sales rows.")
    return data


@st.cache_resource(show_spinner="Loading forecasting model...")
def load_model():
    if not MODEL_FILE.is_file() or not FEATURES_FILE.is_file():
        raise FileNotFoundError("Model or feature list is missing from the project folder.")
    # Load only locally trained/trusted joblib files; pickle files can execute code.
    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURES_FILE)
    if not isinstance(features, (list, tuple, pd.Index)):
        raise ValueError("Saved model features must be a list of column names.")
    return model, list(features)


def prepare_daily_sales(data):
    daily = (data.groupby(data["InvoiceDate"].dt.normalize())["TotalSales"]
             .sum().sort_index().to_frame(name="Actual Sales"))
    daily = daily.asfreq("D", fill_value=0)  # Keep no-sale dates for fair evaluation.
    daily["DayOfWeek"] = daily.index.dayofweek
    daily["Month"] = daily.index.month
    for lag in (1, 7, 14):
        daily[f"Sales_Lag{lag}"] = daily["Actual Sales"].shift(lag)
    for window in (7, 14):
        daily[f"Rolling_Mean{window}"] = daily["Actual Sales"].shift(1).rolling(window).mean()
    return daily.dropna().copy()


st.title("📊 E-Commerce Sales Analytics Dashboard")
st.caption("Sales analysis and historical one-day-ahead machine-learning predictions")

try:
    full_data = load_data()
except (OSError, ValueError, zipfile.BadZipFile) as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()

st.sidebar.header("Dashboard filters")
min_date = full_data["InvoiceDate"].min().date()
max_date = full_data["InvoiceDate"].max().date()
date_range = st.sidebar.date_input("Select date range", (min_date, max_date),
                                    min_value=min_date, max_value=max_date)
if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Select a start and end date to view the sales dashboard.")
    st.stop()
start_date, end_date = date_range
if start_date > end_date:
    st.warning("Start date must be before end date.")
    st.stop()
filtered = full_data.loc[(full_data["InvoiceDate"].dt.date >= start_date) &
                         (full_data["InvoiceDate"].dt.date <= end_date)].copy()
if filtered.empty:
    st.warning("No transactions in the selected date range.")
    st.stop()

col1, col2 = st.columns(2)
col1.metric("Sales Records", f"{len(filtered):,}")
col2.metric("Total Revenue", f"£{filtered['TotalSales'].sum():,.2f}")
st.caption("Sales records count rows in the public dataset, not necessarily unique invoices.")

st.subheader("📈 Monthly Sales Trend")
monthly = filtered.groupby(filtered["InvoiceDate"].dt.to_period("M"))["TotalSales"].sum()
monthly.index = monthly.index.astype(str)
st.line_chart(monthly.rename("Revenue (£)"))

left, right = st.columns(2)
with left:
    st.subheader("🏆 Top 10 Products by Revenue")
    products = (filtered.dropna(subset=["Description"])
                .groupby("Description")["TotalSales"].sum().nlargest(10))
    st.bar_chart(products, horizontal=True)
    st.dataframe(products.rename("Revenue (£)").reset_index(), hide_index=True)
with right:
    st.subheader("🌍 Top 10 Countries by Revenue")
    countries = (filtered.dropna(subset=["Country"])
                 .groupby("Country")["TotalSales"].sum().nlargest(10))
    st.bar_chart(countries, horizontal=True)
    st.dataframe(countries.rename("Revenue (£)").reset_index(), hide_index=True)

st.subheader("Sales Dataset Preview")
st.dataframe(filtered.head(10), hide_index=True)
st.caption("Customer spending rankings are omitted because customer identifiers were removed from the public dataset.")

st.divider()
st.header("🤖 Historical Sales Prediction & Evaluation")
st.info("Predictions use actual previous-day sales from the historical dataset. "
        "This is one-day-ahead historical evaluation, not an untested multi-day future forecast. "
        "The sales date filter above applies only to analytics; model evaluation uses the full dataset.")

try:
    model, model_features = load_model()
    daily_sales = prepare_daily_sales(full_data)
    missing_features = [feature for feature in model_features if feature not in daily_sales.columns]
    if missing_features:
        raise ValueError(f"Model requires missing features: {missing_features}")
    if daily_sales.empty:
        raise ValueError("Not enough historical dates to generate lag features.")
    daily_sales["Predicted Sales"] = model.predict(daily_sales[model_features])
except (OSError, ValueError, KeyError, TypeError) as error:
    st.error(f"Unable to run saved model: {error}")
    st.stop()

st.subheader("Actual vs Predicted Daily Sales")
st.line_chart(daily_sales[["Actual Sales", "Predicted Sales"]])

test_data = daily_sales.loc[TEST_START:TEST_END].copy()
expected_dates = pd.date_range(TEST_START, TEST_END, freq="D")
missing_dates = expected_dates.difference(test_data.index)
if missing_dates.size:
    st.warning(f"Evaluation is missing {len(missing_dates)} expected calendar days; "
               "metrics may not match the original experiment.")
    st.write("Missing evaluation dates:", missing_dates.strftime("%Y-%m-%d").tolist())
if test_data.empty:
    st.error("No records found in the saved model's original test period.")
    st.stop()

y_actual = test_data["Actual Sales"]
y_predicted = test_data["Predicted Sales"]
mae = mean_absolute_error(y_actual, y_predicted)
rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
r2 = r2_score(y_actual, y_predicted) if len(test_data) >= 2 else float("nan")
baseline_mae = mean_absolute_error(y_actual, test_data["Sales_Lag1"])
improvement = (baseline_mae - mae) / baseline_mae * 100 if baseline_mae > 0 else float("nan")

st.subheader("📊 Model Performance on Original Test Period")
a, b, c = st.columns(3)
a.metric("MAE", f"£{mae:,.2f}")
b.metric("RMSE", f"£{rmse:,.2f}")
c.metric("R² Score", f"{r2:.3f}")
st.write(f"Evaluation: {test_data.index.min().date()} to "
         f"{test_data.index.max().date()} ({len(test_data)} calendar days)")

st.subheader("📈 ML Model vs Previous-Day Baseline")
comparison = pd.DataFrame({"Method": ["Linear Regression", "Previous-Day Baseline"],
                           "MAE (£)": [mae, baseline_mae]})
st.dataframe(comparison, hide_index=True)
st.bar_chart(comparison.set_index("Method"))
if np.isfinite(improvement):
    st.metric("MAE Improvement Over Baseline", f"{improvement:.2f}%")
    if improvement > 0:
        st.success("Linear Regression has lower MAE than the previous-day baseline on this test period.")
    else:
        st.warning("Linear Regression does not outperform the previous-day baseline on this test period.")
else:
    st.info("Baseline MAE is zero, so percentage improvement cannot be calculated.")

st.caption("Dataset: Online Retail II (credit the original dataset source and license in README). "
           "Evaluation figures reflect the saved model and the loaded dataset; results are not live forecasts.")
