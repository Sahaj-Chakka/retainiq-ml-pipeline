"""
RetainIQ · src/evaluation/make_eda_figures.py
----------------------------------------------
Generates portfolio EDA charts from the synthetic dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140, "font.size": 10})
ORANGE, NAVY, GREY = "#E85D04", "#1A1A2E", "#9AA0A6"


def main():
    df = pd.read_csv(ROOT / "data" / "synthetic" / "median_inn_synthetic.csv")
    df["Check_In_Date"] = pd.to_datetime(df["Check_In_Date"])

    # 1. Cancellation rate by booking source
    cr = df.groupby("Booking_Source")["Is_Cancelled"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(cr.index, cr.values, color=ORANGE)
    ax.set_title("Cancellation rate by booking source")
    ax.set_xlabel("Cancellation rate")
    for i, v in enumerate(cr.values):
        ax.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURES / "cancellation_by_source.png"); plt.close(fig)

    # 2. Monthly revenue seasonality (honoured bookings)
    rev = (df[df.Is_Cancelled == 0]
           .assign(m=df["Check_In_Date"].dt.to_period("M").astype(str))
           .groupby("m")["Total_Amount"].sum())
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rev.index, rev.values, color=NAVY, marker="o", ms=3)
    ax.set_title("Monthly revenue (synthetic)")
    ax.set_ylabel("Revenue (₹)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "monthly_revenue.png"); plt.close(fig)

    # 3. Room type mix
    # Cancellation rate by room type (more insightful than a demand donut)
    rt_cancel = df.groupby("Room_Type_Reserved")["Is_Cancelled"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(rt_cancel.index, rt_cancel.values * 100, color=[NAVY, ORANGE], width=0.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_title("Cancellation rate by room type")
    ax.set_ylabel("Cancellation rate (%)")
    fig.tight_layout(); fig.savefig(FIGURES / "cancellation_by_room.png"); plt.close(fig)

    # 4. Revenue concentration (Pareto)
    g = df.groupby("Primary_Guest_Name")["Total_Amount"].sum().sort_values(ascending=False)
    cum = g.cumsum() / g.sum()
    x = np.arange(1, len(g) + 1) / len(g)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, cum.values, color=ORANGE, lw=2)
    ax.axvline(0.18, color=GREY, ls="--", lw=1)
    ax.axhline(cum.values[int(len(g) * 0.18)], color=GREY, ls="--", lw=1)
    ax.set_title("Revenue concentration (Pareto curve)")
    ax.set_xlabel("Cumulative share of guests")
    ax.set_ylabel("Cumulative share of revenue")
    fig.tight_layout(); fig.savefig(FIGURES / "revenue_concentration.png"); plt.close(fig)

    print("Saved 4 EDA figures to", FIGURES)


if __name__ == "__main__":
    main()
