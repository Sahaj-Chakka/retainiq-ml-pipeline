"""
RetainIQ · src/ingestion/load_data.py
--------------------------------------
Loads the synthetic booking table (CSV or, if configured, SQL).
"""
import os
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV  = ROOT / "data" / "synthetic" / "median_inn_synthetic.csv"

EXPECTED = ["reservation_id", "booking_source", "room_type", "total_amount",
            "is_cancelled", "check_in_date"]


def load(source: str = "csv") -> pd.DataFrame:
    if source == "sql":
        from sqlalchemy import create_engine
        engine = create_engine(os.environ["DATABASE_URL"])
        df = pd.read_sql("SELECT * FROM bookings", engine)
    else:
        if not CSV.exists():
            raise FileNotFoundError(
                f"{CSV} not found. Run: "
                "python -m src.data_generation.generate_synthetic_data")
        df = pd.read_csv(CSV)
    return df


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df):,} rows × {df.shape[1]} cols")
    print(df.head())
