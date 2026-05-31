"""
RetainIQ · src/analysis/pricing_sensitivity.py
-----------------------------------------------
Analyses price elasticity of demand by booking source and season.
Key question: when we price above competitors, do OTA bookings drop?
Also computes RevPAR (Revenue Per Available Room) by month.

Models:
  • Log-log regression (price elasticity)
  • OLS with interaction terms (elasticity varies by source)
  • RevPAR decomposition: ADR × Occupancy rate
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

ROOT    = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

INVENTORY_ROOMS = 14   # total hotel capacity (real Median Inn: 14 rooms)


def price_elasticity(df: pd.DataFrame) -> dict:
    """
    Log-log regression: ln(nights_booked) ~ ln(avg_price / competitor_avg)
    Coefficient = price elasticity of demand.
    Separate regressions per booking source.
    """
    results = {}
    hon = df[df["Is_Cancelled"] == 0].copy()
    hon = hon[hon["Avg_Price_Per_Room"] > 0]
    hon["log_price_ratio"] = np.log(hon["Avg_Price_Per_Room"]
                                    / hon["Competitor_Avg_Rate"].clip(1))
    hon["log_nights"]      = np.log(hon["Nights_Booked"].clip(1))

    for src in df["Booking_Source"].unique():
        sub = hon[hon["Booking_Source"] == src]
        if len(sub) < 100:
            continue
        X = sub[["log_price_ratio"]].values
        y = sub["log_nights"].values
        model = LinearRegression().fit(X, y)
        results[src] = {
            "elasticity":  round(float(model.coef_[0]), 3),
            "r2":          round(float(model.score(X, y)), 4),
            "n":           int(len(sub)),
            "interpretation": (
                "inelastic (demand insensitive to price)"
                if abs(model.coef_[0]) < 0.3
                else "elastic (demand falls when overpriced)"
            )
        }
    return results


def revpar_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    RevPAR = ADR × Occupancy Rate
    ADR    = average daily rate (among honoured bookings)
    Occupancy = rooms sold / (14 rooms × days in month)
    """
    df  = df.copy()
    df["Check_In_Date"] = pd.to_datetime(df["Check_In_Date"])
    hon = df[df["Is_Cancelled"] == 0]

    monthly = hon.groupby(hon["Check_In_Date"].dt.to_period("M")).agg(
        adr          = ("Avg_Price_Per_Room", "mean"),
        rooms_sold   = ("Stayed_Room_Nights", "sum"),
    ).reset_index()
    monthly["Check_In_Date"] = monthly["Check_In_Date"].astype(str)
    monthly["days_in_month"] = monthly["Check_In_Date"].apply(
        lambda m: pd.Period(m).days_in_month)
    monthly["capacity"]  = INVENTORY_ROOMS * monthly["days_in_month"]
    monthly["occupancy"] = (monthly["rooms_sold"] / monthly["capacity"]).clip(0, 1)
    monthly["revpar"]    = (monthly["adr"] * monthly["occupancy"]).round(2)
    monthly["adr"]       = monthly["adr"].round(2)
    monthly["occupancy"] = monthly["occupancy"].round(4)
    return monthly


def net_revenue_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare gross vs net revenue (after OTA commission) per booking source.
    Shows true cost of OTA dependency.
    """
    hon = df[df["Is_Cancelled"] == 0]
    src = hon.groupby("Booking_Source").agg(
        bookings         = ("Reservation_ID", "count"),
        gross_revenue    = ("Total_Amount",   "sum"),
        net_revenue      = ("Net_Revenue",    "sum"),
        avg_commission   = ("OTA_Commission_Pct", "mean"),
    ).reset_index()
    src["commission_paid"] = (src["gross_revenue"] - src["net_revenue"]).round(2)
    src["net_per_booking"] = (src["net_revenue"] / src["bookings"]).round(2)
    src["gross_per_booking"] = (src["gross_revenue"] / src["bookings"]).round(2)
    return src.sort_values("net_revenue", ascending=False)


def run(df: pd.DataFrame):
    # 1. Price elasticity
    elasticity = price_elasticity(df)
    print("\n=== Price Elasticity by Booking Source ===")
    for src, res in elasticity.items():
        print(f"  {src:22s}  elasticity={res['elasticity']:+.3f}  R²={res['r2']}  → {res['interpretation']}")

    # 2. RevPAR decomposition
    revpar = revpar_decomposition(df)
    print("\n=== RevPAR by Month (sample) ===")
    print(revpar[["Check_In_Date","adr","occupancy","revpar"]].tail(6).to_string(index=False))
    revpar.to_csv(REPORTS / "revpar_monthly.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    months = revpar["Check_In_Date"]
    axes[0].plot(months, revpar["adr"], color="#E85D04", marker="o", ms=3, label="ADR (₹)")
    axes[0].set_ylabel("Avg Daily Rate (₹)"); axes[0].legend()
    axes[0].set_title("RevPAR decomposition: ADR and Occupancy rate")
    axes[1].plot(months, revpar["occupancy"] * 100, color="#4aa8ff", marker="o", ms=3,
                 label="Occupancy %")
    axes[1].set_ylabel("Occupancy (%)"); axes[1].legend()
    axes[1].tick_params(axis="x", rotation=75, labelsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "revpar_decomposition.png", dpi=140)
    plt.close(fig)

    # 3. Net revenue by source (OTA commission cost)
    net_rev = net_revenue_by_source(df)
    print("\n=== Net Revenue by Source (after OTA commission) ===")
    print(net_rev.to_string(index=False))
    net_rev.to_csv(REPORTS / "net_revenue_by_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(net_rev))
    w = 0.35
    ax.bar(x - w/2, net_rev["gross_per_booking"], w, label="Gross per booking", color="#E85D04")
    ax.bar(x + w/2, net_rev["net_per_booking"],   w, label="Net per booking",   color="#4aa8ff")
    ax.set_xticks(x); ax.set_xticklabels(net_rev["Booking_Source"], rotation=25, ha="right")
    ax.set_ylabel("Revenue per booking (₹)")
    ax.set_title("Gross vs net revenue per booking — OTA commission cost")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIGURES / "net_revenue_by_source.png", dpi=140)
    plt.close(fig)

    summary = {"elasticity": elasticity,
               "revpar_tail": json.loads(revpar.tail(6).to_json(orient="records")),
               "net_revenue_by_source": json.loads(net_rev.to_json(orient="records"))}
    with open(REPORTS / "pricing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "synthetic" / "median_inn_synthetic.csv")
    run(df)
