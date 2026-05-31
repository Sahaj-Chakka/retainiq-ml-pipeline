"""
RetainIQ · src/analysis/rfm_segmentation.py
--------------------------------------------
RFM (Recency · Frequency · Monetary) scoring with named guest tiers:
  VIP         → retain, upsell, personalise
  At Risk     → re-engage before lapse
  Loyal       → reward, protect
  New         → nurture, convert to repeat
  Lapsed      → win-back campaign or write off
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT    = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

SNAPSHOT_DATE = pd.Timestamp("2024-01-01")   # "today" for recency calc


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """One row per guest with R, F, M scores."""
    df  = df.copy()
    df["Check_In_Date"] = pd.to_datetime(df["Check_In_Date"])
    hon = df[df["Is_Cancelled"] == 0]

    rfm = hon.groupby("Primary_Guest_Name").agg(
        last_stay      = ("Check_In_Date", "max"),
        frequency      = ("Reservation_ID", "count"),
        monetary       = ("Total_Amount",   "sum"),
        avg_nights     = ("Stayed_Room_Nights", "mean"),
        cancel_rate    = ("Is_Cancelled",    "mean"),   # always 0 here; use full df below
        n_special_req  = ("N_Special_Requests", "mean"),
    ).reset_index()

    # Re-join cancellation rate from full dataset
    cancel_hist = df.groupby("Primary_Guest_Name")["Is_Cancelled"].mean().reset_index()
    cancel_hist.columns = ["Primary_Guest_Name", "cancel_rate"]
    rfm = rfm.drop(columns=["cancel_rate"]).merge(cancel_hist, on="Primary_Guest_Name")

    rfm["recency_days"] = (SNAPSHOT_DATE - rfm["last_stay"]).dt.days

    # Quintile scoring: 5 = best, 1 = worst
    rfm["R"] = pd.qcut(rfm["recency_days"], 5, labels=[5,4,3,2,1]).astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5,
                        labels=[1,2,3,4,5]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 5,
                        labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_Score"] = rfm["R"] + rfm["F"] + rfm["M"]
    return rfm


def assign_tier(rfm: pd.DataFrame) -> pd.DataFrame:
    """Rule-based tier assignment on RFM scores."""
    def tier(row):
        r, f, m, score = row["R"], row["F"], row["M"], row["RFM_Score"]
        if score >= 13:                          return "VIP"
        if r >= 4 and f >= 3:                   return "Loyal"
        if r <= 2 and f >= 3:                   return "At Risk"
        if f == 1 and r >= 3:                   return "New"
        return "Lapsed"
    rfm["Tier"] = rfm.apply(tier, axis=1)
    return rfm


def run(df: pd.DataFrame):
    rfm = build_rfm(df)
    rfm = assign_tier(rfm)

    # Tier summary
    tier_summary = rfm.groupby("Tier").agg(
        n_guests      = ("Primary_Guest_Name", "count"),
        avg_revenue   = ("monetary",    "mean"),
        total_revenue = ("monetary",    "sum"),
        avg_recency   = ("recency_days","mean"),
        avg_frequency = ("frequency",   "mean"),
        avg_cancel    = ("cancel_rate", "mean"),
    ).round(2).reset_index()
    tier_summary["pct_guests"]  = (tier_summary["n_guests"]
                                    / tier_summary["n_guests"].sum() * 100).round(1)
    tier_summary["pct_revenue"] = (tier_summary["total_revenue"]
                                    / tier_summary["total_revenue"].sum() * 100).round(1)
    tier_summary = tier_summary.sort_values("avg_revenue", ascending=False)
    print("\n=== RFM Tier Profiles ===")
    print(tier_summary.to_string(index=False))

    # Recommended action per tier
    actions = {
        "VIP":      "Personalised loyalty reward + early access to offers",
        "Loyal":    "Targeted upsell (room upgrade, meal plan)",
        "At Risk":  "Proactive win-back email within 30 days",
        "New":      "Welcome sequence + incentive for second stay",
        "Lapsed":   "Reactivation offer or remove from active CRM",
    }
    tier_summary["recommended_action"] = tier_summary["Tier"].map(actions)

    # Save
    rfm.to_csv(REPORTS / "rfm_scores.csv", index=False)
    tier_summary.to_csv(REPORTS / "rfm_tier_summary.csv", index=False)

    # Visualisation: tier revenue contribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"VIP":"#E85D04","Loyal":"#4aa8ff","At Risk":"#d29922",
              "New":"#3fb950","Lapsed":"#888"}
    ts_sorted = tier_summary.sort_values("pct_revenue", ascending=False)
    bars = axes[0].bar(ts_sorted["Tier"], ts_sorted["pct_revenue"],
                       color=[colors[t] for t in ts_sorted["Tier"]])
    axes[0].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    axes[0].set_title("Revenue share by guest tier")
    axes[0].set_ylabel("% of total revenue")

    axes[1].scatter(rfm["recency_days"], rfm["monetary"],
                    c=[{"VIP":"#E85D04","Loyal":"#4aa8ff","At Risk":"#d29922",
                        "New":"#3fb950","Lapsed":"#aaa"}[t] for t in rfm["Tier"]],
                    alpha=0.25, s=6)
    axes[1].set_xlabel("Recency (days since last stay)")
    axes[1].set_ylabel("Lifetime revenue (₹)")
    axes[1].set_title("Guest map: recency vs lifetime value")
    for tier, color in colors.items():
        axes[1].scatter([], [], c=color, label=tier, s=30)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "rfm_tiers.png", dpi=140)
    plt.close(fig)

    summary = {"tier_profiles": json.loads(tier_summary.to_json(orient="records")),
               "n_guests": int(len(rfm))}
    with open(REPORTS / "rfm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "synthetic" / "median_inn_synthetic.csv")
    run(df)
