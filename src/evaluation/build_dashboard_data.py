"""
RetainIQ · src/evaluation/build_dashboard_data.py
--------------------------------------------------
Produces (1) a JSON of real aggregates for the HTML dashboard and
(2) tidy CSV extracts ready to drop into Tableau / Power BI.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
DASH      = ROOT / "dashboard"
TABLEAU   = DASH / "tableau_ready"
DASH.mkdir(exist_ok=True); TABLEAU.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(ROOT / "data" / "synthetic" / "median_inn_synthetic.csv")
    df["Check_In_Date"] = pd.to_datetime(df["Check_In_Date"])
    honoured = df[df.Is_Cancelled == 0]

    # Monthly revenue
    monthly = (honoured.assign(m=honoured["Check_In_Date"].dt.to_period("M").astype(str))
               .groupby("m")["Total_Amount"].sum().reset_index())
    monthly.columns = ["month", "revenue"]

    # Cancellation by source
    cxs = (df.groupby("Booking_Source")["Is_Cancelled"].mean()
           .sort_values(ascending=False).reset_index())
    cxs.columns = ["booking_source", "cancellation_rate"]

    # Revenue by source (honoured)
    rxs = (honoured.groupby("Booking_Source")["Total_Amount"].sum()
           .sort_values(ascending=False).reset_index())
    rxs.columns = ["booking_source", "revenue"]
    rxs["pct_of_revenue"] = (rxs["revenue"] / rxs["revenue"].sum() * 100).round(1)

    # Cancellation rate by room type (replaces the uninformative demand donut)
    room = (df.groupby("Room_Type_Reserved")["Is_Cancelled"].mean()
            .mul(100).round(1).reset_index())
    room.columns = ["room_type", "cancel_rate"]

    # Pareto curve (downsampled to 100 points)
    g = df.groupby("Primary_Guest_Name")["Total_Amount"].sum().sort_values(ascending=False)
    cum = (g.cumsum() / g.sum()).values
    idx = np.linspace(0, len(cum) - 1, 100).astype(int)
    pareto = [{"x": round((i + 1) / len(cum), 4), "y": round(cum[i], 4)} for i in idx]

    # Lead-time buckets (cancellation rate climbs with lead time)
    df2 = df.copy()
    df2["lead_bucket"] = pd.cut(df2["Lead_Time_Days"], bins=[0,7,14,30,60,999],
        labels=["0-7d","8-14d","15-30d","31-60d","60d+"])
    lead = (df2.groupby("lead_bucket", observed=True)["Is_Cancelled"].mean()
            .mul(100).round(1).reset_index())
    lead.columns = ["bucket", "cancel_rate"]

    # Model benchmark + analyses (from reports)
    cancel = json.loads((ROOT / "reports" / "cancellation_summary.json").read_text())
    rev    = json.loads((ROOT / "reports" / "revenue_summary.json").read_text())
    fcast  = json.loads((ROOT / "reports" / "forecast_summary.json").read_text())
    seg    = json.loads((ROOT / "reports" / "segmentation_summary.json").read_text())
    logit  = json.loads((ROOT / "reports" / "logistic_summary.json").read_text())
    rfm    = json.loads((ROOT / "reports" / "rfm_summary.json").read_text())
    overb  = json.loads((ROOT / "reports" / "overbooking_summary.json").read_text())

    data = {
        "kpis": {
            "bookings": int(len(df)),
            "cancellation_rate": round(float(df.Is_Cancelled.mean()), 3),
            "total_revenue": int(honoured.Total_Amount.sum()),
            "avg_booking_value": int(honoured.Total_Amount.mean()),
            "logit_auc": logit["validation_auc"],
            "logit_threshold": logit["optimal_threshold"],
            "best_cancel_model": cancel["best_model"],
            "best_cancel_auc": cancel["best_auc"],
            "best_revenue_r2": rev["benchmark"][0]["R2"],
            "forecast_improvement": fcast["improvement_pct"],
            "top18_revenue_share": seg["concentration"]["top_18pct_revenue_share"],
            "optimal_overbook": overb["optimal_overbook"],
        },
        "monthly_revenue": monthly.to_dict(orient="records"),
        "cancellation_by_source": cxs.to_dict(orient="records"),
        "revenue_by_source": rxs.to_dict(orient="records"),
        "cancellation_by_room": room.to_dict(orient="records"),
        "lead_time": lead.to_dict(orient="records"),
        "pareto": pareto,
        "cancel_benchmark": cancel["benchmark"],
        "revenue_benchmark": rev["benchmark"],
        "rfm_tiers": rfm["tier_profiles"],
        "n_significant_vars": len(logit["significant_variables"]),
    }
    (DASH / "dashboard_data.json").write_text(json.dumps(data, indent=2))

    # Tableau / Power BI tidy extracts
    monthly.to_csv(TABLEAU / "monthly_revenue.csv", index=False)
    cxs.to_csv(TABLEAU / "cancellation_by_source.csv", index=False)
    rxs.to_csv(TABLEAU / "revenue_by_source.csv", index=False)
    lead.to_csv(TABLEAU / "lead_time_cancellation.csv", index=False)
    (honoured.groupby(["Booking_Source", honoured["Check_In_Date"].dt.to_period("M").astype(str)])
        .agg(bookings=("Reservation_ID", "count"), revenue=("Total_Amount", "sum"))
        .reset_index().rename(columns={"Check_In_Date": "month"})
        .to_csv(TABLEAU / "source_month_fact.csv", index=False))
    pd.DataFrame(cancel["benchmark"]).to_csv(TABLEAU / "model_benchmark.csv", index=False)

    print("Dashboard data + Tableau extracts written to", DASH)
    return data


if __name__ == "__main__":
    main()
