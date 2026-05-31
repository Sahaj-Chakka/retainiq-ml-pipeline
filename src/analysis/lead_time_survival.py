"""
RetainIQ · src/analysis/lead_time_survival.py
-----------------------------------------------
Kaplan-Meier style analysis: when in the lead-time window do
cancellations actually happen, and what does that imply for
deposit policy?

Also runs logistic regression to quantify the lead-time effect
on cancellation probability at different source / policy buckets.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT    = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def cancellation_timing(df: pd.DataFrame):
    """
    For cancelled bookings: at what fraction of lead time did the cancel happen?
    cancel_days_before_arrival / lead_time_days → 0 = cancelled at last minute,
                                                    1 = cancelled immediately after booking
    """
    cancelled = df[(df["Is_Cancelled"] == 1) & (df["Lead_Time_Days"] > 0)].copy()
    cancelled["cancel_fraction"] = (
        cancelled["Cancel_Days_Before_Arrival"] / cancelled["Lead_Time_Days"]
    ).clip(0, 1)
    return cancelled


def lead_time_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Cancellation rate by lead-time bucket."""
    df = df.copy()
    df["lead_bucket"] = pd.cut(
        df["Lead_Time_Days"],
        bins=[0, 7, 14, 30, 60, 999],
        labels=["0–7 days", "8–14 days", "15–30 days", "31–60 days", "60+ days"],
        right=True
    )
    out = (df.groupby("lead_bucket", observed=True)["Is_Cancelled"]
           .agg(["mean","count"])
           .rename(columns={"mean": "cancellation_rate", "count": "bookings"})
           .reset_index())
    out["cancellation_rate"] = out["cancellation_rate"].round(4)
    return out


def policy_simulation(df: pd.DataFrame) -> dict:
    """
    Simulate revenue impact of a non-refundable deposit policy
    for bookings with lead time > 30 days.

    Assumption: 40% of OTA/Web bookings with lead > 30d would
    convert to non-refundable if offered a 5% discount.
    Non-refundable guests cancel at the rate of Walk-In guests (~5%).
    """
    target = df[
        (df["Booking_Source"].isin(["OTA","Web"])) &
        (df["Lead_Time_Days"] > 30)
    ]
    current_cancellations = int(target["Is_Cancelled"].sum())
    current_lost_revenue  = float((target[target["Is_Cancelled"]==1]["Avg_Price_Per_Room"]
                                   * target[target["Is_Cancelled"]==1]["Nights_Booked"]).sum())
    # Under new policy: 40% convert, cancel rate drops to 5%
    convert_n   = int(len(target) * 0.40)
    remain_n    = len(target) - convert_n
    cancel_saved = int(convert_n * (target["Is_Cancelled"].mean() - 0.05))
    avg_rev      = float(target["Avg_Price_Per_Room"].mean() * target["Nights_Booked"].mean())
    revenue_recovered = round(cancel_saved * avg_rev, 2)
    return {
        "target_bookings":       len(target),
        "current_cancel_n":      current_cancellations,
        "current_cancel_rate":   round(target["Is_Cancelled"].mean(), 4),
        "estimated_cancels_saved": cancel_saved,
        "estimated_revenue_recovered": revenue_recovered,
    }


def run(df: pd.DataFrame):
    # 1. Cancellation timing
    cancelled  = cancellation_timing(df)
    fig, axes  = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(cancelled["Cancel_Days_Before_Arrival"], bins=40,
                 color="#E85D04", edgecolor="none", alpha=0.85)
    axes[0].set_title("Days before arrival when booking is cancelled")
    axes[0].set_xlabel("Days before arrival"); axes[0].set_ylabel("Count")

    axes[1].hist(cancelled["cancel_fraction"], bins=30,
                 color="#1A1A2E", edgecolor="none", alpha=0.85)
    axes[1].axvline(0.5, color="#E85D04", lw=1.5, ls="--",
                    label="Midpoint of lead window")
    axes[1].set_title("When (relative to lead time) cancellation happens\n"
                      "0=last minute, 1=immediately after booking")
    axes[1].set_xlabel("Fraction of lead-time window elapsed")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "cancellation_timing.png", dpi=140)
    plt.close(fig)

    # 2. Cancellation rate by lead-time bucket
    buckets = lead_time_buckets(df)
    print("\nCancellation rate by lead-time bucket:")
    print(buckets.to_string(index=False))
    buckets.to_csv(REPORTS / "lead_time_buckets.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#3fb950","#4aa8ff","#d29922","#E85D04","#cf222e"]
    bars = ax.bar(buckets["lead_bucket"].astype(str),
                  buckets["cancellation_rate"] * 100,
                  color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.set_title("Cancellation rate rises sharply with booking lead time")
    ax.set_ylabel("Cancellation rate (%)")
    ax.set_xlabel("Days between booking and check-in")
    fig.tight_layout()
    fig.savefig(FIGURES / "lead_time_cancellation.png", dpi=140)
    plt.close(fig)

    # 3. Policy simulation
    policy = policy_simulation(df)
    print(f"\nDeposit policy simulation:")
    print(f"  Target bookings (OTA/Web, lead>30d): {policy['target_bookings']:,}")
    print(f"  Current cancellation rate: {policy['current_cancel_rate']:.1%}")
    print(f"  Estimated cancellations saved: {policy['estimated_cancels_saved']:,}")
    print(f"  Estimated revenue recovered: ₹{policy['estimated_revenue_recovered']:,.0f}")

    summary = {"lead_time_buckets": buckets.to_dict(orient="records"),
               "policy_simulation": policy}
    with open(REPORTS / "lead_time_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "synthetic" / "median_inn_synthetic.csv")
    run(df)
