"""Smoke tests for the RetainIQ pipeline."""
import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation.generate_synthetic_data import generate
from src.features.build_features import build_cancellation_features, build_revenue_features


@pytest.fixture(scope="module")
def df():
    return generate(2000, seed=7)


def test_generator_shape(df):
    assert len(df) == 2000
    assert "Is_Cancelled" in df.columns
    assert df["Is_Cancelled"].isin([0, 1]).all()


def test_cancellation_rate_realistic(df):
    rate = df["Is_Cancelled"].mean()
    assert 0.25 < rate < 0.45, f"cancellation rate {rate:.2%} outside expected band"


def test_cancelled_rows_have_zero_amount(df):
    assert (df.loc[df["Is_Cancelled"] == 1, "Total_Amount"] == 0).all()


def test_cancellation_features_no_leakage(df):
    X, y, names = build_cancellation_features(df)
    assert "Total_Amount" not in names
    assert len(X) == len(y)


def test_revenue_features_exclude_per_night_rate(df):
    X, y, names = build_revenue_features(df)
    # leakage guard: the per-night rate must not be a feature
    assert not any("Per_Room_Night" in n for n in names)
    assert (y > 0).all()
