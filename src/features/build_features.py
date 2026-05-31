"""
RetainIQ · src/features/build_features.py  (v2 — enriched schema)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT         = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42
TARGET       = "Is_Cancelled"
DROP_COLS    = ["Hotel","Reservation_ID","Primary_Guest_Name","Corporate_Name",
                "Booking_Date","Check_In_Date","Check_Out_Date",
                "Pax","Special_Request_Detail",         # text / identifier
                "Stayed_Room_Nights","Total_Amount",    # post-outcome leakage
                "Total_Lodging_Amount","Total_Lodging_Taxes",
                "Net_Revenue","Paid_at_Hotel_Amount","Paid_To_OTA_Amount",
                "Cancel_Days_Before_Arrival",           # reveals outcome
                "Room_Type_Assigned"]                   # same as reserved for most rows


def add_datetime_features(df):
    df = df.copy()
    df["Check_In_Date"] = pd.to_datetime(df["Check_In_Date"])
    df["checkin_month"]      = df["Check_In_Date"].dt.month
    df["checkin_dow"]        = df["Check_In_Date"].dt.dayofweek
    df["checkin_is_weekend"] = (df["Check_In_Date"].dt.dayofweek >= 4).astype(int)
    df["checkin_quarter"]    = df["Check_In_Date"].dt.quarter
    return df


def add_derived_features(df):
    df = df.copy()
    df["is_hot_room"]              = df["Room_No"].isin([202,206,208,210,301,302,309]).astype(int)
    df["has_children"]             = (df["N_Children"] > 0).astype(int)
    df["is_overpriced"]            = (df["Rate_Premium_Pct"] > 10).astype(int)
    df["cancellation_history_ratio"] = (
        df["Previous_Cancellations"] /
        (df["Previous_Cancellations"] + df["Previous_Successful_Bookings"] + 1)
    )
    df["lead_time_sq"] = df["Lead_Time_Days"] ** 2   # nonlinear signal
    df["high_lead"]    = (df["Lead_Time_Days"] > 30).astype(int)
    return df


def build_cancellation_features(df):
    df = add_datetime_features(df)
    df = add_derived_features(df)
    df = add_derived_features(df)
    cat_cols = ["Booking_Source","Market_Segment","Room_Type_Reserved","Meal_Plan",
                "Payment_Mode"]
    feature_df = df.drop(columns=DROP_COLS, errors="ignore")
    feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)
    feature_df = feature_df.drop(columns=[TARGET], errors="ignore")
    y = df[TARGET]
    return feature_df, y, list(feature_df.columns)


def build_revenue_features(df):
    """Regression on honoured bookings: predict Total_Amount."""
    df = df[df["Is_Cancelled"] == 0].copy()
    df = add_datetime_features(df)
    df = add_derived_features(df)
    cat_cols = ["Booking_Source","Market_Segment","Room_Type_Reserved","Meal_Plan"]
    keep = ["Nights_Booked","N_Adults","N_Children",
            "N_Special_Requests","Got_Room_Upgrade","OTA_Commission_Pct",
            "checkin_month","checkin_is_weekend","checkin_quarter",
            "is_hot_room"] + cat_cols
    X = pd.get_dummies(df[keep], columns=cat_cols, drop_first=True)
    # Remove per-night rate to avoid near-tautological R² — model must learn pricing from context
    y = df["Total_Amount"]
    return X, y, list(X.columns)


def split(X, y, test_size=0.25):
    return train_test_split(X, y, test_size=test_size,
                            stratify=y if y.nunique()==2 else None,
                            random_state=RANDOM_STATE)
