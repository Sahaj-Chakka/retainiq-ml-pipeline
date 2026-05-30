"""
RetainIQ · src/forecasting/run_prophet.py
------------------------------------------
Aggregates honoured-booking revenue to a daily series, fits Prophet,
and compares forecast accuracy to a seasonal-naive baseline via
time-series cross-validation. All metrics are real (no hard-coding).
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings("ignore")
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

ROOT    = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
FORECAST_DAYS = 90


def build_daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Is_Cancelled"] == 0].copy()
    df["Check_In_Date"] = pd.to_datetime(df["Check_In_Date"])
    ts = (df.groupby("Check_In_Date")["Total_Amount"].sum()
            .rename("y").reset_index().rename(columns={"Check_In_Date": "ds"}))
    ts = ts.sort_values("ds").reset_index(drop=True)
    return ts


def seasonal_naive_rmse(ts: pd.DataFrame, horizon: int = 90, period: int = 7) -> float:
    """Baseline: predict each day as the value `period` days earlier.
    Evaluated on the same trailing horizon Prophet is scored on."""
    y = ts["y"].values
    test = y[-horizon:]
    preds = y[-horizon - period:-period]
    return float(np.sqrt(np.mean((test - preds) ** 2)))


def run(df: pd.DataFrame):
    ts = build_daily_revenue(df)
    print(f"Daily revenue series: {len(ts)} days "
          f"({ts['ds'].min().date()} → {ts['ds'].max().date()})")

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, interval_width=0.90,
                    changepoint_prior_scale=0.05)
    model.fit(ts)

    future   = model.make_future_dataframe(periods=FORECAST_DAYS, freq="D")
    forecast = model.predict(future)

    # ── Time-series cross-validation ────────────────────────────────────────────
    cv = cross_validation(model, initial="450 days", period="90 days",
                          horizon="90 days", parallel=None)
    pm = performance_metrics(cv)
    prophet_rmse = float(pm["rmse"].mean())
    prophet_mae  = float(pm["mae"].mean())
    prophet_mape = float(pm["mape"].mean())

    baseline_rmse = seasonal_naive_rmse(ts)
    improvement = (baseline_rmse - prophet_rmse) / baseline_rmse * 100

    print(f"\nProphet  RMSE={prophet_rmse:,.0f}  MAE={prophet_mae:,.0f}  MAPE={prophet_mape:.1%}")
    print(f"Seasonal-naive baseline RMSE={baseline_rmse:,.0f}")
    print(f"Prophet improvement over baseline: {improvement:.1f}%")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts["ds"], ts["y"], color="#1A1A2E", lw=0.8, label="Actual revenue")
    fc = forecast[forecast["ds"] > ts["ds"].max()]
    ax.plot(fc["ds"], fc["yhat"], color="#E85D04", lw=2, label="Prophet forecast")
    ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                    color="#E85D04", alpha=0.18, label="90% interval")
    ax.set_title("Median Inn — Daily Revenue & 90-Day Forecast (synthetic data)")
    ax.set_xlabel("Date"); ax.set_ylabel("Revenue (₹)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "prophet_forecast.png", dpi=140)
    plt.close(fig)

    comp = model.plot_components(forecast)
    comp.savefig(FIGURES / "prophet_components.png", dpi=140)
    plt.close(comp)

    summary = {"prophet_rmse": round(prophet_rmse, 2),
               "prophet_mae": round(prophet_mae, 2),
               "prophet_mape": round(prophet_mape, 4),
               "baseline_rmse": round(baseline_rmse, 2),
               "improvement_pct": round(improvement, 2)}
    with open(REPORTS / "forecast_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "synthetic" / "median_inn_synthetic.csv")
    run(df)
