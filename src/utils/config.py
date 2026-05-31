"""
RetainIQ · src/utils/config.py
-------------------------------
Shared paths and a small logging helper used across the pipeline.
"""
import logging
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[2]
DATA_DIR      = ROOT / "data"
SYNTHETIC_CSV = DATA_DIR / "synthetic" / "median_inn_synthetic.csv"
SAMPLE_CSV    = DATA_DIR / "sample" / "sample_50_rows.csv"
REPORTS_DIR   = ROOT / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"
MODELS_DIR    = ROOT / "models"

# Reproducibility
RANDOM_STATE = 42

# Schema constants (single source of truth)
BOOKING_SOURCES = ["OTA", "Corporate", "Walk - In", "Travel Agent", "Web"]
ROOM_TYPES      = ["Maple (Deluxe)", "Mahogany (Premium)"]
HOT_ROOMS       = [202, 206, 208, 210, 301, 302, 309]
CATEGORICAL     = ["Booking_Source", "Market_Segment", "Room_Type_Reserved",
                   "Meal_Plan", "Payment_Mode", "Corporate_Name"]


def get_logger(name: str = "retainiq", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
    return logger
