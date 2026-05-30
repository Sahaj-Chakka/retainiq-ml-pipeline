"""
RetainIQ · src/models/train_revenue.py
---------------------------------------
Predicts booking revenue (Total_Amount) on honoured bookings.
Benchmarks linear, regularised, and tree models. Reports real R²/RMSE/MAE.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

ROOT    = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
RANDOM_STATE = 42


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge":             Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso":             Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000),
        "Random Forest":     RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
        "LightGBM":          lgb.LGBMRegressor(n_estimators=500, random_state=RANDOM_STATE, verbose=-1),
        "XGBoost":           xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                              subsample=0.9, colsample_bytree=0.9,
                                              random_state=RANDOM_STATE, n_jobs=-1),
    }


def run(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    rows = []
    best_name, best_rmse, best_model = None, np.inf, None
    for name, model in get_models().items():
        linear = name in ("Linear Regression", "Ridge", "Lasso")
        Xtr, Xte = (X_train_s, X_test_s) if linear else (X_train, X_test)
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae  = mean_absolute_error(y_test, pred)
        r2   = r2_score(y_test, pred)
        rows.append({"Model": name, "R2": round(r2, 4),
                     "RMSE": round(rmse, 2), "MAE": round(mae, 2)})
        print(f"  {name:20s}  R2={r2:.4f}  RMSE={rmse:8.2f}  MAE={mae:8.2f}")
        if rmse < best_rmse:
            best_name, best_rmse, best_model = name, rmse, model

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    results.to_csv(REPORTS / "revenue_benchmark.csv", index=False)
    print(f"\nBest model: {best_name}  (RMSE={best_rmse:.2f})")

    summary = {"best_model": best_name, "best_rmse": round(best_rmse, 2),
               "benchmark": results.to_dict(orient="records")}
    with open(REPORTS / "revenue_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return results, summary


if __name__ == "__main__":
    from src.features.build_features import build_revenue_features
    df = pd.read_csv(ROOT / "data" / "synthetic" / "median_inn_synthetic.csv")
    X, y, names = build_revenue_features(df)
    print(f"Revenue features: {X.shape[1]} cols, {len(X):,} honoured bookings, "
          f"mean ₹{y.mean():,.0f}\n")
    run(X, y, names)
