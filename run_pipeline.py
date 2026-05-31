"""
RetainIQ · run_pipeline.py
--------------------------
Runs the full analysis pipeline end to end, in the correct analytical order:

  0. (optional) regenerate synthetic data
  1. EDA figures
  2. PRIMARY: Logistic regression  → significance, VIF, parsimonious model, threshold
  3. Cancellation ML benchmark      (9 models, on the significant feature set)
  4. Revenue regression             (6 models)
  5. Revenue forecasting            (Prophet vs baseline)
  6. Customer segmentation          (K-Means + PCA)
  7. Business analyses              (RFM tiers, lead-time/deposit, pricing, overbooking)
  8. Dashboard                      (data + self-contained HTML)

    python run_pipeline.py            # uses existing synthetic data
    python run_pipeline.py --regen    # regenerate data first
"""
import argparse
import subprocess
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def banner(txt):
    print("\n" + "=" * 64 + f"\n  {txt}\n" + "=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="regenerate synthetic data")
    ap.add_argument("--rows", type=int, default=70000)
    args = ap.parse_args()

    if args.regen:
        subprocess.run([sys.executable, "-m",
                        "src.data_generation.generate_synthetic_data",
                        "--rows", str(args.rows)], check=True)

    from src.ingestion.load_data import load
    from src.features.build_features import (build_cancellation_features,
                                             build_revenue_features)
    from src.models import train_cancellation, train_revenue
    from src.forecasting import run_prophet
    from src.segmentation import cluster
    from src.analysis import (logistic_regression_primary, lead_time_survival,
                              rfm_segmentation, pricing_sensitivity,
                              overbooking_policy)
    from src.evaluation import (make_eda_figures, build_dashboard_data,
                                build_dashboard_html)

    df = load()

    banner("1 · EDA FIGURES")
    make_eda_figures.main()

    banner("2 · PRIMARY ANALYSIS — LOGISTIC REGRESSION (significance → VIF → threshold)")
    logistic_regression_primary.run(df)

    banner("3 · CANCELLATION ML BENCHMARK (9 models)")
    Xc, yc, nc = build_cancellation_features(df)
    train_cancellation.run(Xc, yc, nc)

    banner("4 · REVENUE REGRESSION (6 models)")
    Xr, yr, nr = build_revenue_features(df)
    train_revenue.run(Xr, yr, nr)

    banner("5 · REVENUE FORECASTING (Prophet vs baseline)")
    run_prophet.run(df)

    banner("6 · CUSTOMER SEGMENTATION (K-Means + PCA)")
    cluster.run(df)

    banner("7a · RFM SEGMENTATION (named tiers + actions)")
    rfm_segmentation.run(df)

    banner("7b · LEAD-TIME & DEPOSIT POLICY")
    lead_time_survival.run(df)

    banner("7c · PRICING SENSITIVITY & RevPAR")
    pricing_sensitivity.run(df)

    banner("7d · OVERBOOKING POLICY (Monte Carlo)")
    overbooking_policy.run(df)

    banner("8 · DASHBOARD")
    build_dashboard_data.main()
    build_dashboard_html.main()

    print("\n✓ Pipeline complete.")
    print("  Metrics  → reports/*.json, reports/*.csv")
    print("  Figures  → reports/figures/")
    print("  Dashboard→ dashboard/index.html")


if __name__ == "__main__":
    main()
