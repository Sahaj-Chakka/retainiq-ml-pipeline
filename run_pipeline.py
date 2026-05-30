"""
RetainIQ · run_pipeline.py
--------------------------
Runs the full pipeline end to end and prints a consolidated summary.

    python run_pipeline.py            # uses existing synthetic data
    python run_pipeline.py --regen    # regenerate data first
"""
import argparse
import subprocess
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="regenerate synthetic data")
    ap.add_argument("--rows", type=int, default=70000)
    args = ap.parse_args()

    if args.regen:
        subprocess.run([sys.executable, "-m", "src.data_generation.generate_synthetic_data",
                        "--rows", str(args.rows)], check=True)

    from src.ingestion.load_data import load
    from src.features.build_features import build_cancellation_features, build_revenue_features
    from src.models import train_cancellation, train_revenue
    from src.forecasting import run_prophet
    from src.segmentation import cluster
    from src.evaluation import make_eda_figures

    df = load()

    print("\n" + "=" * 60 + "\n  EDA FIGURES\n" + "=" * 60)
    make_eda_figures.main()

    print("\n" + "=" * 60 + "\n  CANCELLATION CLASSIFICATION\n" + "=" * 60)
    Xc, yc, nc = build_cancellation_features(df)
    train_cancellation.run(Xc, yc, nc)

    print("\n" + "=" * 60 + "\n  REVENUE REGRESSION\n" + "=" * 60)
    Xr, yr, nr = build_revenue_features(df)
    train_revenue.run(Xr, yr, nr)

    print("\n" + "=" * 60 + "\n  REVENUE FORECASTING (PROPHET)\n" + "=" * 60)
    run_prophet.run(df)

    print("\n" + "=" * 60 + "\n  CUSTOMER SEGMENTATION\n" + "=" * 60)
    cluster.run(df)

    print("\n" + "=" * 60 + "\n  DASHBOARD\n" + "=" * 60)
    from src.evaluation import build_dashboard_data, build_dashboard_html
    build_dashboard_data.main()
    build_dashboard_html.main()

    print("\n✓ Pipeline complete. Metrics in reports/, figures in reports/figures/, "
          "dashboard in dashboard/index.html.")


if __name__ == "__main__":
    main()
