"""
RetainIQ · src/data_generation/generate_synthetic_data.py
----------------------------------------------------------
Generates a SYNTHETIC hotel-booking dataset modeled on the real
'Median Inn' schema (an anonymised hotel group project).

The data is NOT real. Relationships (cancellation drivers, seasonality,
room/source pricing, corporate revenue concentration) are deliberately
encoded so models find genuine, explainable signal — but every metric
reported downstream is whatever the models actually score on this data.

Usage:
    python -m src.data_generation.generate_synthetic_data --rows 70000 --seed 42
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

ROOT      = Path(__file__).resolve().parents[2]
OUT_DIR   = ROOT / "data" / "synthetic"
SAMPLE    = ROOT / "data" / "sample"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE.mkdir(parents=True, exist_ok=True)

# ── Schema values (taken from the real anonymised filter outputs) ───────────────
BOOKING_SOURCES = ["OTA", "Corporate", "Walk - In", "Travel Agent", "Web"]
SOURCE_WEIGHTS  = [0.46, 0.22, 0.18, 0.08, 0.06]

CORPORATES = ["makemytrip", "goibibo", "agoda", "booking-com", "easemytrip",
              "IIFL FINANCE LIMITED", "INFOSYS", "TATA TECHNOLOGIES",
              "JETWAYS TRAVELS", "TRAVLOGUE INDIA", "B2B BULK"]

ROOM_TYPES   = ["Maple (Deluxe)", "Mahogany (Premium)"]
ROOM_WEIGHTS = [0.92, 0.08]                      # real finding: 92% deluxe

PAX_LEVELS   = ["1(A) 0(C)", "2(A) 0(C)", "2(A) 1(C)", "2(A) 2(C)", "3(A) 0(C)"]
PAX_WEIGHTS  = [0.30, 0.42, 0.12, 0.08, 0.08]

PAYMENT_MODES = ["Prepaid", "Pay at Hotel"]

# Room numbers (floors 2-3, the ones that show up in the real analysis)
ROOM_NUMBERS = [201, 202, 203, 204, 205, 206, 208, 210,
                301, 302, 303, 308, 309, 310]
# Real finding: these rooms get booked most
HOT_ROOMS    = [202, 206, 208, 210, 301, 302, 309]

# Monthly demand multiplier (real finding: Apr & Sep high; May & Jun low)
MONTH_DEMAND = {1: 0.85, 2: 0.90, 3: 1.00, 4: 1.30, 5: 0.70, 6: 0.65,
                7: 0.95, 8: 1.05, 9: 1.30, 10: 1.15, 11: 1.00, 12: 1.10}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_rows: int, seed: int = 42,
             start="2022-01-01", end="2023-12-31") -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Build a date pool weighted by monthly demand + weekend lift ─────────────
    all_days = pd.date_range(start, end, freq="D")
    day_weight = np.array([
        MONTH_DEMAND[d.month] * (1.25 if d.weekday() >= 4 else 1.0)
        for d in all_days
    ])
    day_weight = day_weight / day_weight.sum()
    check_in = rng.choice(all_days, size=n_rows, p=day_weight)
    check_in = pd.to_datetime(check_in)

    # ── Core categoricals ───────────────────────────────────────────────────────
    booking_source = rng.choice(BOOKING_SOURCES, size=n_rows, p=SOURCE_WEIGHTS)
    room_type      = rng.choice(ROOM_TYPES, size=n_rows, p=ROOM_WEIGHTS)
    pax            = rng.choice(PAX_LEVELS, size=n_rows, p=PAX_WEIGHTS)
    payment_mode   = rng.choice(PAYMENT_MODES, size=n_rows, p=[0.55, 0.45])

    # Room number — bias toward hot rooms
    room_pool   = np.array(ROOM_NUMBERS)
    room_w      = np.where(np.isin(room_pool, HOT_ROOMS), 2.2, 1.0)
    room_w      = room_w / room_w.sum()
    room_no     = rng.choice(room_pool, size=n_rows, p=room_w)

    # Corporate name only populated for Corporate / OTA sources
    corporate_name = np.array([None] * n_rows, dtype=object)
    for i in range(n_rows):
        if booking_source[i] == "Corporate":
            corporate_name[i] = rng.choice(
                ["IIFL FINANCE LIMITED", "INFOSYS", "TATA TECHNOLOGIES",
                 "JETWAYS TRAVELS", "TRAVLOGUE INDIA", "B2B BULK"])
        elif booking_source[i] == "OTA":
            corporate_name[i] = rng.choice(
                ["makemytrip", "goibibo", "agoda", "booking-com", "easemytrip"])

    # ── Stayed room nights (0 = will become cancellation) ───────────────────────
    base_nights = rng.choice([1, 2, 3, 4, 5, 6, 7],
                             size=n_rows, p=[0.42, 0.26, 0.14, 0.08, 0.05, 0.03, 0.02])
    # Corporate stays trend slightly longer
    corp_mask = booking_source == "Corporate"
    base_nights[corp_mask] = np.clip(base_nights[corp_mask] + rng.integers(0, 2, corp_mask.sum()), 1, 7)

    # Lead time (days between booking and check-in) — drives cancellation
    lead_time = rng.gamma(shape=2.0, scale=10.0, size=n_rows).astype(int)  # ~0-60 days

    # ── Pricing ─────────────────────────────────────────────────────────────────
    # Per-room-night base by room type
    base_rate = np.where(room_type == "Mahogany (Premium)",
                         rng.normal(4200, 350, n_rows),
                         rng.normal(2750, 280, n_rows))
    # Corporate negotiated rates a touch lower per night but bulk → higher totals
    base_rate[corp_mask] *= 0.95
    # Seasonal price lift
    season_mult = np.array([MONTH_DEMAND[d.month] for d in check_in])
    per_night = np.clip(base_rate * (0.9 + 0.2 * season_mult), 1500, 8000).round(2)

    total_lodging = (per_night * base_nights).round(2)
    lodging_tax   = (total_lodging * 0.12).round(2)         # 12% GST proxy
    total_amount  = (total_lodging + lodging_tax).round(2)

    # ── Cancellation model (the REAL signal we want models to recover) ──────────
    # Higher cancellation if: OTA/Web source, long lead time, prepaid, more pax,
    # off-peak month. Lower for corporate. Plus noise.
    src_effect = pd.Series(booking_source).map({
        "OTA": 0.55, "Web": 0.65, "Walk - In": -0.9,
        "Travel Agent": 0.15, "Corporate": -0.7}).values
    lead_effect   = 0.020 * (lead_time - 15)
    season_effect = -0.6 * (season_mult - 1.0)
    pax_effect    = 0.10 * (pd.Series(pax).map(
        {"1(A) 0(C)": 0, "2(A) 0(C)": 1, "2(A) 1(C)": 2,
         "2(A) 2(C)": 3, "3(A) 0(C)": 2}).values)
    prepaid_effect = np.where(payment_mode == "Prepaid", 0.25, -0.25)

    # Nonlinear interaction pocket: OTA/Web + Prepaid + long lead time cancels a lot.
    # Linear models can't capture this AND-interaction well; trees can — which is
    # exactly why gradient boosting wins the benchmark downstream.
    online = np.isin(booking_source, ["OTA", "Web"])
    interaction = np.where(online & (payment_mode == "Prepaid") & (lead_time > 25), 1.6, 0.0)
    # A second pocket: corporate + short lead time almost never cancels
    interaction += np.where((booking_source == "Corporate") & (lead_time < 7), -1.2, 0.0)

    noise = rng.normal(0, 0.45, n_rows)

    logit = (-0.95 + src_effect + lead_effect + season_effect
             + pax_effect + prepaid_effect + interaction + noise)
    cancel_prob = _sigmoid(logit)
    is_cancelled = (rng.random(n_rows) < cancel_prob).astype(int)

    # When cancelled → zero out amounts and nights (mirrors real data convention)
    stayed_nights = np.where(is_cancelled == 1, 0, base_nights)
    total_lodging = np.where(is_cancelled == 1, 0.0, total_lodging)
    lodging_tax   = np.where(is_cancelled == 1, 0.0, lodging_tax)
    total_amount  = np.where(is_cancelled == 1, 0.0, total_amount)

    # Split who collected the money (Hotel vs Treebo/OTA)
    paid_at_hotel = np.where(np.isin(booking_source, ["Walk - In", "Corporate"]),
                             total_amount, 0.0)
    paid_to_ota   = total_amount - paid_at_hotel
    invoice_by    = np.where(paid_at_hotel > 0, "Hotel", "Treebo")

    # ── Repeat customers (real finding: ~11% of guests are repeaters) ───────────
    # Large guest universe (many one-time OTA travellers over 2 years) so that
    # only a realistic minority of bookings come from returning guests.
    n_unique = int(n_rows * 4.1)
    guest_ids = rng.integers(1, n_unique + 1, size=n_rows)
    guest_name = np.array([f"Guest_{g:06d}" for g in guest_ids])

    check_out = check_in + pd.to_timedelta(np.where(stayed_nights > 0, stayed_nights, 1), unit="D")

    df = pd.DataFrame({
        "Hotel":               "Median Inn",
        "Reservation_ID":      [f"R{100000+i}" for i in range(n_rows)],
        "Booking_Source":      booking_source,
        "Primary_Guest_Name":  guest_name,
        "Corporate_Name":      corporate_name,
        "Room_No":             room_no,
        "Room_Type":           room_type,
        "Pax":                 pax,
        "Payment_Mode":        payment_mode,
        "Check_In_Date":       check_in,
        "Check_Out_Date":      check_out,
        "Lead_Time_Days":      lead_time,
        "Stayed_Room_Nights":  stayed_nights,
        "Per_Room_Night_Charges": per_night,
        "Total_Lodging_Amount": total_lodging,
        "Total_Lodging_Taxes":  lodging_tax,
        "Total_Amount":         total_amount,
        "Paid_at_Hotel_Amount": paid_at_hotel.round(2),
        "Paid_To_OTA_Amount":   paid_to_ota.round(2),
        "Guest_Invoice_Issued_By": invoice_by,
        "Is_Cancelled":         is_cancelled,
    })

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=70000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = generate(args.rows, args.seed)

    out_path = OUT_DIR / "median_inn_synthetic.csv"
    df.to_csv(out_path, index=False)

    # Small committed sample (anonymised already — synthetic)
    df.head(50).to_csv(SAMPLE / "sample_50_rows.csv", index=False)

    print(f"Generated {len(df):,} rows  →  {out_path}")
    print(f"Cancellation rate: {df['Is_Cancelled'].mean():.2%}")
    print(f"Date range: {df['Check_In_Date'].min().date()} → {df['Check_In_Date'].max().date()}")
    print(f"Room mix:\n{df['Room_Type'].value_counts(normalize=True).round(3)}")
    print(f"Repeat guests: {(df['Primary_Guest_Name'].value_counts() > 1).sum():,} "
          f"({(df['Primary_Guest_Name'].duplicated().sum() / len(df)):.1%} of rows)")


if __name__ == "__main__":
    main()
