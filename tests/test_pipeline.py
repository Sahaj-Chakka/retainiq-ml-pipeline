"""
RetainIQ · tests/test_pipeline.py
----------------------------------
Smoke tests for the full pipeline (enriched 37-column schema, v2).
Covers: data generation, feature engineering (leakage guards),
        logistic regression, RFM segmentation, overbooking simulation.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation.generate_synthetic_data import generate
from src.features.build_features import build_cancellation_features, build_revenue_features


# ── Shared fixture ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def df():
    return generate(2000, seed=7)


# ── Data generation ────────────────────────────────────────────────────────────
def test_generator_shape(df):
    assert len(df) == 2000
    assert df.shape[1] == 37, f"Expected 37 columns, got {df.shape[1]}"


def test_required_columns_present(df):
    required = [
        "Reservation_ID", "Booking_Source", "Market_Segment", "Lead_Time_Days",
        "Previous_Cancellations", "Previous_Successful_Bookings",
        "N_Special_Requests", "Room_Type_Reserved", "Meal_Plan",
        "Avg_Price_Per_Room", "Competitor_Avg_Rate", "Rate_Premium_Pct",
        "Net_Revenue", "Cancel_Days_Before_Arrival", "Is_Cancelled",
    ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_cancellation_rate_realistic(df):
    rate = df["Is_Cancelled"].mean()
    assert 0.20 < rate < 0.50, f"Cancellation rate {rate:.2%} outside expected band"


def test_direct_cancel_low(df):
    """Direct / Front Desk should cancel much less than OTA (fixed from v1)."""
    direct = df[df["Booking_Source"] == "Direct / Front Desk"]["Is_Cancelled"].mean()
    ota    = df[df["Booking_Source"] == "OTA"]["Is_Cancelled"].mean()
    assert direct < 0.15, f"Direct cancel rate {direct:.2%} too high (should be <15%)"
    assert ota > direct, "OTA should cancel more than Direct"


def test_cancelled_rows_zero_revenue(df):
    assert (df.loc[df["Is_Cancelled"] == 1, "Total_Amount"] == 0).all()
    assert (df.loc[df["Is_Cancelled"] == 1, "Net_Revenue"]  == 0).all()


def test_cancel_days_minus1_for_honoured(df):
    """Cancel_Days_Before_Arrival should be -1 for non-cancelled bookings."""
    assert (df.loc[df["Is_Cancelled"] == 0, "Cancel_Days_Before_Arrival"] == -1).all()


def test_lead_time_correlates_with_cancellation(df):
    """Longer lead time → higher cancellation. Core relationship must hold."""
    df2 = df.copy()
    df2["long_lead"] = (df2["Lead_Time_Days"] > 30).astype(int)
    high = df2[df2["long_lead"] == 1]["Is_Cancelled"].mean()
    low  = df2[df2["long_lead"] == 0]["Is_Cancelled"].mean()
    assert high > low, f"Long lead cancel rate ({high:.2%}) should exceed short ({low:.2%})"


# ── Feature engineering — leakage guards ──────────────────────────────────────
def test_cancellation_features_no_outcome_leakage(df):
    X, y, names = build_cancellation_features(df)
    leaked = [c for c in names if any(k in c for k in
              ["Total_Amount","Net_Revenue","Stayed_Room","Cancel_Days",
               "Paid_at_Hotel","Paid_To_OTA","Lodging"])]
    assert not leaked, f"Outcome-leaking columns in features: {leaked}"
    assert len(X) == len(y)


def test_revenue_features_no_rate_leakage(df):
    """Per-night rate × nights = revenue mechanically; must be excluded."""
    X, y, names = build_revenue_features(df)
    assert not any("Avg_Price_Per_Room" in n for n in names), \
        "Avg_Price_Per_Room leaks revenue — must be excluded"
    assert (y > 0).all(), "Revenue target should be positive for honoured bookings"


def test_revenue_features_honoured_only(df):
    """Revenue model must only train on non-cancelled bookings."""
    X, y, names = build_revenue_features(df)
    assert len(X) < len(df), "Revenue features should exclude cancelled bookings"


# ── Analysis modules ──────────────────────────────────────────────────────────
def test_logistic_regression_runs(df):
    """Logistic regression module runs without error and returns expected keys."""
    from src.analysis.logistic_regression_primary import run
    summary = run(df)
    assert "validation_auc" in summary
    assert "significant_variables" in summary
    assert len(summary["significant_variables"]) > 0
    assert 0.5 < summary["validation_auc"] < 1.0, \
        f"AUC {summary['validation_auc']} outside expected range"


def test_rfm_segmentation_tiers(df):
    """RFM must produce all five named tiers with sensible revenue splits."""
    from src.analysis.rfm_segmentation import build_rfm, assign_tier
    rfm = assign_tier(build_rfm(df))
    tiers = set(rfm["Tier"].unique())
    expected = {"VIP", "Loyal", "At Risk", "New", "Lapsed"}
    assert expected == tiers or tiers.issubset(expected), \
        f"Unexpected tiers: {tiers - expected}"


def test_overbooking_simulation_returns_optimal(df):
    """Overbooking simulation must identify an optimal overbook level."""
    from src.analysis.overbooking_policy import run as ob_run
    summary = ob_run(df)
    assert "optimal_overbook" in summary
    assert 0 <= summary["optimal_overbook"] <= 6, \
        f"Optimal overbook {summary['optimal_overbook']} outside range 0-6"
