"""
RetainIQ · src/features/build_features.py
------------------------------------------
Turns the raw booking table into model-ready features for two tasks:
  • cancellation classification  (target: Is_Cancelled)
  • revenue regression           (target: Total_Amount, non-cancelled only)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CATEGORICAL = ["Booking_Source", "Room_Type", "Pax", "Payment_Mode", "Corporate_Name"]


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Check_In_Date"] = pd.to_datetime(df["Check_In_Date"])
    df["checkin_month"]    = df["Check_In_Date"].dt.month
    df["checkin_dow"]      = df["Check_In_Date"].dt.dayofweek
    df["checkin_is_weekend"] = (df["checkin_dow"] >= 4).astype(int)
    df["checkin_quarter"]  = df["Check_In_Date"].dt.quarter
    return df


def add_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Repeat-guest flag (count of prior bookings by same guest)
    df = df.sort_values("Check_In_Date")
    df["guest_booking_seq"] = df.groupby("Primary_Guest_Name").cumcount()
    df["is_repeat_guest"]   = (df["guest_booking_seq"] > 0).astype(int)

    # Hot-room flag
    hot = [202, 206, 208, 210, 301, 302, 309]
    df["is_hot_room"] = df["Room_No"].isin(hot).astype(int)

    # Number of adults / children parsed from Pax string
    df["n_adults"]   = df["Pax"].str.extract(r"(\d+)\(A\)").astype(float)
    df["n_children"] = df["Pax"].str.extract(r"(\d+)\(C\)").astype(float)
    return df


def build_cancellation_features(df: pd.DataFrame):
    """Returns X (DataFrame), y (Series), feature list. One-hot encoded."""
    df = add_datetime_features(df)
    df = add_behavioural_features(df)

    feature_cols = [
        "Booking_Source", "Room_Type", "Pax", "Payment_Mode",
        "Lead_Time_Days", "Per_Room_Night_Charges",
        "checkin_month", "checkin_dow", "checkin_is_weekend", "checkin_quarter",
        "is_repeat_guest", "is_hot_room", "n_adults", "n_children",
    ]
    X = df[feature_cols].copy()
    y = df["Is_Cancelled"].copy()

    # One-hot the categoricals present in feature_cols
    cat_in = [c for c in CATEGORICAL if c in X.columns]
    X = pd.get_dummies(X, columns=cat_in, drop_first=True)
    return X, y, list(X.columns)


def build_revenue_features(df: pd.DataFrame):
    """Regression on realised bookings only (exclude cancellations)."""
    df = df[df["Is_Cancelled"] == 0].copy()
    df = add_datetime_features(df)
    df = add_behavioural_features(df)

    feature_cols = [
        "Booking_Source", "Room_Type", "Pax",
        "Stayed_Room_Nights",
        "checkin_month", "checkin_is_weekend",
        "is_repeat_guest", "is_hot_room", "n_adults", "n_children",
    ]
    # NOTE: Per_Room_Night_Charges is deliberately EXCLUDED — it mechanically
    # determines Total_Amount (price × nights × tax), so including it would leak
    # the target. The model must instead infer pricing from room type, source,
    # seasonality and length of stay — the realistic forecasting task.
    X = df[feature_cols].copy()
    y = df["Total_Amount"].copy()

    cat_in = [c for c in CATEGORICAL if c in X.columns]
    X = pd.get_dummies(X, columns=cat_in, drop_first=True)
    return X, y, list(X.columns)
