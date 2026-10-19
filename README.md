# E-Commerce Sales Analytics & Forecasting

**Interactive sales analytics and historical one-day-ahead sales prediction using Python, Machine Learning, and Streamlit.**

[**Live Dashboard**](https://ecommerce-sales-analytics-forecasting-o8cmb7gysshtcwxq55ivm8.streamlit.app/) · [**GitHub Repository**](https://github.com/Saranya707/Ecommerce-Sales-Analytics-Forecasting)

## Project Overview

This project analyzes historical e-commerce transactions to identify sales trends, high-revenue products, and leading countries. It also evaluates a **Linear Regression** model for predicting daily sales from prior sales observations and calendar-based features. Results are presented through an interactive Streamlit dashboard.

## Problem Statement

Retail businesses need accessible ways to explore sales performance and assess whether historical sales patterns can help predict near-term demand. This project combines exploratory analysis with a reproducible historical prediction evaluation and a web-based dashboard.

## Features

- **Sales overview:** sales-record count and total revenue in GBP (£)
- **Date filter:** explore analytics within a selected date range
- **Monthly revenue:** interactive time-series visualization
- **Top 10 products:** rank product descriptions by revenue
- **Top 10 countries:** compare geographic revenue contributions
- **Dataset preview:** inspect sample transaction-level records
- **ML predictions:** compare actual and predicted historical daily sales
- **Model evaluation:** MAE, RMSE, R², and comparison against a previous-day-sales baseline

> **Scope:** The ML component is a **historical one-day-ahead prediction evaluation** using actual earlier sales as lag inputs. It is **not** a validated multi-day future forecasting system. The dashboard's date filter controls sales analytics; the ML evaluation uses the full historical dataset and a fixed test period.

## Tech Stack

| Area | Technologies |
| --- | --- |
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn (Linear Regression) |
| Model persistence | joblib |
| Web dashboard | Streamlit |
| Version control / hosting | GitHub, Streamlit Community Cloud |
| Exploratory analysis | Python and SQL during project development |

## Dataset

The project uses a processed version of the **Online Retail II** dataset. The dashboard reads `ecommerce_public.csv.zip`, which contains the CSV data required for the public-facing sales visualizations. The public dataset omits `CustomerID`, `InvoiceNo`, and `StockCode` fields; therefore, the dashboard does **not** present customer-level rankings or claim its row count is a count of unique invoices.

**Dataset attribution:** Online Retail II, UCI Machine Learning Repository. Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii. Please confirm that the uploaded processed dataset originates from this source and review its licensing/attribution requirements before reuse. Changes in this project include cleaning, selecting dashboard columns, and ZIP compression.

## Machine Learning Approach

1. Aggregate historical transaction revenue into daily totals.
2. Reindex the series to include every calendar day, filling no-sale dates with zero.
3. Create predictive features:
   - `DayOfWeek`, `Month`
   - `Sales_Lag1`, `Sales_Lag7`, `Sales_Lag14`
   - `Rolling_Mean7`, `Rolling_Mean14` (computed from prior days only)
4. Load the saved Linear Regression model and its saved feature order.
5. Compare model predictions with actual sales on the original held-out period.
6. Benchmark the model against predicting each day's sales using the previous day's actual sales.

### Evaluation

- **Test period:** 30 September 2011 – 9 December 2011 (71 calendar days)
- **Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R²
- **Baseline:** Previous-day actual sales
- **Reported Colab result:** approximately **£19,004.07 MAE** and **26.23% lower MAE** than the baseline in the original experiment.

The deployed app recomputes evaluation metrics from the bundled model and dataset. **Treat its displayed values as authoritative if they differ from the original Colab run.** These historical evaluation results do not establish performance for iterative future-day forecasts.

## Project Files

```text
Ecommerce-Sales-Analytics-Forecasting/
├── app.py                       # Streamlit dashboard
├── ecommerce_public.csv.zip     # Compressed dataset for dashboard
├── sales_forecasting_model.pkl  # Saved regression model
├── model_features.pkl           # Saved model feature names/order
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Run Locally

Requires Python 3 and the files shown above in the same project folder.

```bash
# From the project directory
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL shown in the terminal, typically `http://localhost:8501`.

> Only load the `.pkl` model files from a source you trust; joblib/pickle files can execute code when loaded.

## Deployment

Hosted on **Streamlit Community Cloud**, connected to this repository's `main` branch with `app.py` as the entry point.

**Live app:** https://ecommerce-sales-analytics-forecasting-o8cmb7gysshtcwxq55ivm8.streamlit.app/

## Limitations and Future Work

- The current model is evaluated with **observed historical lag values** rather than recursively forecasting unknown future days.
- Model performance may change over time and across retail settings.
- Customer-level analytics were intentionally excluded from the public dataset.
- Potential improvements include walk-forward validation, stronger seasonal baselines, model comparisons, and a separately validated future-forecasting workflow.

## Project Purpose

Developed as a practical Data Science and Machine Learning portfolio project demonstrating data preparation, feature engineering, historical model evaluation, dashboard development, and cloud deployment.
