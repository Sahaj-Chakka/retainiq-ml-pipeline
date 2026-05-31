"""
RetainIQ · src/data_generation/generate_synthetic_data.py  (v2)
---------------------------------------------------------------
Enriched schema: 35 columns vs original 21.
New columns add genuine behavioural, temporal, and operational signal
that enables the four new analyses:
  • Lead-time survival analysis (cancellation timing)
  • RFM segmentation with actionable tiers
  • Dynamic pricing / RevPAR sensitivity
  • Overbooking policy simulation

All cancellation relationships are encoded explicitly so models find
REAL, EXPLAINABLE signal — not noise. Every downstream metric is
whatever the models actually score.

Usage:
    python -m src.data_generation.generate_synthetic_data --rows 70000 --seed 42
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "synthetic"
SAMPLE  = ROOT / "data" / "sample"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE.mkdir(parents=True, exist_ok=True)

# ── Schema constants (anchored to real Median Inn filter outputs) ──────────────
BOOKING_SOURCES  = ["OTA", "Corporate", "Direct / Front Desk", "Travel Agent", "Web"]
SOURCE_WEIGHTS   = [0.46, 0.22, 0.18, 0.08, 0.06]

ROOM_TYPES   = ["Maple (Deluxe)", "Mahogany (Premium)"]
ROOM_WEIGHTS = [0.92, 0.08]

# Market segment maps to booking source — finer-grained for RFM and pricing
MARKET_SEGMENTS = {
    "OTA":                  "Online Leisure",
    "Web":                  "Online Leisure",
    "Corporate":            "Corporate",
    "Travel Agent":         "Group / Agent",
    "Direct / Front Desk":  "Direct",
}

MEAL_PLANS = ["Breakfast Included", "No Meal", "Half Board", "Full Board"]
MEAL_WEIGHTS = [0.48, 0.38, 0.09, 0.05]

SPECIAL_REQUESTS = ["None", "Early Check-In", "Late Check-Out",
                    "High Floor", "Extra Bed", "Quiet Room"]

ROOM_NUMBERS = [201,202,203,204,205,206,208,210,
                301,302,303,308,309,310]
HOT_ROOMS    = [202,206,208,210,301,302,309]

# Monthly demand (real project: Apr & Sep high; May/Jun low)
MONTH_DEMAND = {1:0.85,2:0.90,3:1.00,4:1.30,5:0.70,6:0.65,
                7:0.95,8:1.05,9:1.30,10:1.15,11:1.00,12:1.10}

COMPETITORS = ["Hotel Sunrise Inn", "Grand Plaza Suites",
               "City Center Residency", "Business Inn Express"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_rows: int, seed: int = 42,
             start: str = "2022-01-01", end: str = "2023-12-31") -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Check-in date (weighted by demand + weekend lift) ─────────────────────
    all_days   = pd.date_range(start, end, freq="D")
    day_weight = np.array([
        MONTH_DEMAND[d.month] * (1.25 if d.weekday() >= 4 else 1.0)
        for d in all_days])
    day_weight /= day_weight.sum()
    check_in = pd.to_datetime(rng.choice(all_days, size=n_rows, p=day_weight))

    # ── Booking source and derived segments ───────────────────────────────────
    booking_source = rng.choice(BOOKING_SOURCES, size=n_rows, p=SOURCE_WEIGHTS)
    market_segment = np.array([MARKET_SEGMENTS[s] for s in booking_source])

    # ── Lead time: distribution varies meaningfully by source ─────────────────
    # Corporate books short-notice; OTA/Web book further out
    lead_time = np.zeros(n_rows, dtype=int)
    for i, src in enumerate(booking_source):
        if src == "Corporate":
            lead_time[i] = int(rng.gamma(1.2, 5))        # mostly <14d
        elif src == "Direct / Front Desk":
            lead_time[i] = int(rng.gamma(1.0, 2))        # very short
        elif src in ("OTA", "Web"):
            lead_time[i] = int(rng.gamma(2.5, 10))       # longer tail
        else:  # Travel Agent
            lead_time[i] = int(rng.gamma(2.0, 12))
    lead_time = np.clip(lead_time, 0, 180)

    # Booking date = check-in minus lead time
    booking_date = check_in - pd.to_timedelta(lead_time, unit="D")

    # ── Room attributes ───────────────────────────────────────────────────────
    room_type = rng.choice(ROOM_TYPES, size=n_rows, p=ROOM_WEIGHTS)
    room_w    = np.where(np.isin(ROOM_NUMBERS, HOT_ROOMS), 2.2, 1.0)
    room_w   /= room_w.sum()
    room_no   = rng.choice(ROOM_NUMBERS, size=n_rows, p=room_w)

    # Reserved room type may differ from assigned (upgrade / downgrade)
    reserved_room_type = room_type.copy()
    upgrade_mask = rng.random(n_rows) < 0.06   # 6% get upgraded
    reserved_room_type[upgrade_mask] = "Maple (Deluxe)"

    # ── Stay length and guest count ───────────────────────────────────────────
    corp_mask = booking_source == "Corporate"
    nights_booked = rng.choice([1,2,3,4,5,6,7], size=n_rows,
                                p=[0.40,0.27,0.14,0.09,0.05,0.03,0.02])
    nights_booked[corp_mask] = np.clip(
        nights_booked[corp_mask] + rng.integers(0,2, corp_mask.sum()), 1, 7)

    n_adults   = rng.choice([1,2,3], size=n_rows, p=[0.30, 0.60, 0.10])
    n_children = rng.choice([0,1,2], size=n_rows, p=[0.78, 0.14, 0.08])
    pax = np.array([f"{a}(A) {c}(C)" for a, c in zip(n_adults, n_children)])

    # ── Special requests (0-5; more requests → more invested guest → less cancel)
    n_special_requests = rng.choice([0,1,2,3,4,5], size=n_rows,
                                     p=[0.35,0.32,0.18,0.09,0.04,0.02])
    special_request_detail = rng.choice(SPECIAL_REQUESTS, size=n_rows)

    # ── Meal plan ─────────────────────────────────────────────────────────────
    meal_plan = rng.choice(MEAL_PLANS, size=n_rows, p=MEAL_WEIGHTS)
    # Corp guests skew toward breakfast
    corp_idx = np.where(corp_mask)[0]
    meal_plan[corp_idx] = rng.choice(
        ["Breakfast Included","No Meal"], size=len(corp_idx), p=[0.72,0.28])

    # ── Guest history (repeat behaviour) ──────────────────────────────────────
    n_unique = int(n_rows * 4.1)
    guest_ids = rng.integers(1, n_unique + 1, size=n_rows)
    guest_name = np.array([f"Guest_{g:06d}" for g in guest_ids])

    # Simulate previous cancellations and bookings per guest
    # (in real data you'd compute these from history; here we assign plausible values)
    prev_cancellations = rng.choice([0,1,2,3], size=n_rows,
                                     p=[0.70, 0.18, 0.08, 0.04])
    prev_successful    = rng.choice([0,1,2,3,4,5], size=n_rows,
                                     p=[0.42, 0.26, 0.16, 0.09, 0.05, 0.02])
    # Repeat guests (same guest_id appearing >1 time) get non-zero prev_successful
    # This is captured implicitly via guest_ids; downstream RFM computes per-guest stats

    # ── Pricing ───────────────────────────────────────────────────────────────
    season_mult = np.array([MONTH_DEMAND[d.month] for d in check_in])
    weekend_mult = np.where(check_in.dayofweek >= 4, 1.18, 1.0)

    base_rate = np.where(room_type == "Mahogany (Premium)",
                         rng.normal(4200, 350, n_rows),
                         rng.normal(2750, 280, n_rows))
    base_rate[corp_mask] *= 0.93   # negotiated corporate discount

    avg_price_per_room = np.clip(
        base_rate * (0.88 + 0.24 * season_mult) * weekend_mult, 1500, 8500
    ).round(2)

    # Competitor pricing (market context — used in dynamic pricing analysis)
    competitor_avg_rate = np.clip(
        avg_price_per_room * rng.normal(1.04, 0.08, n_rows), 1400, 10000
    ).round(2)

    # Rate premium vs competitor (positive = our rate is higher)
    rate_premium_pct = ((avg_price_per_room - competitor_avg_rate)
                        / competitor_avg_rate * 100).round(2)

    # ── Revenue amounts ───────────────────────────────────────────────────────
    total_lodging = (avg_price_per_room * nights_booked).round(2)
    lodging_tax   = (total_lodging * 0.12).round(2)
    total_amount  = (total_lodging + lodging_tax).round(2)

    payment_mode = np.where(
        np.isin(booking_source, ["Direct / Front Desk", "Corporate"]),
        "Pay at Hotel",
        rng.choice(["Prepaid","Pay at Hotel"], size=n_rows, p=[0.62, 0.38])
    )

    # ── Cancellation model (richer signal with new features) ──────────────────
    src_effect = pd.Series(booking_source).map({
        "OTA": 0.55, "Web": 0.65,
        "Direct / Front Desk": -1.20,   # near-zero cancel (fix from v1)
        "Travel Agent": 0.15, "Corporate": -0.70}).values

    lead_effect    = 0.022 * (lead_time - 15)
    season_effect  = -0.55 * (season_mult - 1.0)
    prepaid_effect = np.where(payment_mode == "Prepaid", 0.30, -0.30)
    history_effect = -0.35 * prev_successful + 0.45 * prev_cancellations
    request_effect = -0.12 * n_special_requests   # more requests = more invested
    meal_effect    = np.where(meal_plan == "No Meal", 0.20, -0.15)
    premium_effect = np.where(rate_premium_pct > 10, 0.25, 0.0)  # overpriced → cancels more

    # Nonlinear pocket: OTA/Web + Prepaid + long lead = high cancel
    online   = np.isin(booking_source, ["OTA","Web"])
    interact = np.where(online & (payment_mode == "Prepaid") & (lead_time > 25), 1.5, 0.0)
    # Corporate + short lead + repeat = almost never cancels
    interact += np.where(
        corp_mask & (lead_time < 7) & (prev_successful > 0), -1.3, 0.0)

    noise = rng.normal(0, 0.42, n_rows)
    logit = (-0.90 + src_effect + lead_effect + season_effect + prepaid_effect
             + history_effect + request_effect + meal_effect + premium_effect
             + interact + noise)

    cancel_prob  = _sigmoid(logit)
    is_cancelled = (rng.random(n_rows) < cancel_prob).astype(int)

    # Cancelled bookings → zero revenue
    stayed_nights  = np.where(is_cancelled, 0, nights_booked)
    total_lodging  = np.where(is_cancelled, 0.0, total_lodging)
    lodging_tax    = np.where(is_cancelled, 0.0, lodging_tax)
    total_amount   = np.where(is_cancelled, 0.0, total_amount)

    # For cancellations: days_before_arrival when cancellation happened
    cancel_days_before = np.where(
        is_cancelled,
        np.clip(
            (lead_time * rng.beta(2.5, 1.5, n_rows)).astype(int), 0, lead_time
        ),
        -1   # -1 = not cancelled
    )

    # ── OTA commission / net revenue ──────────────────────────────────────────
    ota_commission_pct = np.where(
        np.isin(booking_source, ["OTA","Web"]), rng.uniform(0.15, 0.20, n_rows), 0.0)
    net_revenue = (total_amount * (1 - ota_commission_pct)).round(2)

    paid_at_hotel = np.where(
        np.isin(booking_source, ["Direct / Front Desk","Corporate"]),
        total_amount, 0.0).round(2)
    paid_to_ota   = (total_amount - paid_at_hotel).round(2)

    # ── RevPAR and occupancy proxy ────────────────────────────────────────────
    # Total hotel capacity: 14 rooms. RevPAR = revenue / total_capacity_nights
    # We attach a daily_occupancy_pct as an aggregate feature
    # (downstream analysis groups by date to compute this properly)
    inventory_rooms = 14

    # ── Check-out date ────────────────────────────────────────────────────────
    check_out = check_in + pd.to_timedelta(
        np.where(stayed_nights > 0, stayed_nights, 1), unit="D")

    # ── Corporate name ────────────────────────────────────────────────────────
    corporate_name = np.array([None]*n_rows, dtype=object)
    for i in range(n_rows):
        if booking_source[i] == "Corporate":
            corporate_name[i] = rng.choice([
                "IIFL FINANCE LIMITED","INFOSYS","TATA TECHNOLOGIES",
                "JETWAYS TRAVELS","TRAVLOGUE INDIA","B2B BULK"])
        elif booking_source[i] == "OTA":
            corporate_name[i] = rng.choice([
                "makemytrip","goibibo","agoda","booking-com","easemytrip"])

    # ── Assemble dataframe ────────────────────────────────────────────────────
    df = pd.DataFrame({
        # Identifiers
        "Hotel":                    "Median Inn",
        "Reservation_ID":           [f"R{100000+i}" for i in range(n_rows)],
        "Booking_Date":             pd.to_datetime(booking_date).date,
        "Check_In_Date":            check_in,
        "Check_Out_Date":           check_out,

        # Booking characteristics
        "Booking_Source":           booking_source,
        "Market_Segment":           market_segment,
        "Corporate_Name":           corporate_name,
        "Lead_Time_Days":           lead_time,

        # Guest
        "Primary_Guest_Name":       guest_name,
        "N_Adults":                 n_adults,
        "N_Children":               n_children,
        "Pax":                      pax,
        "Previous_Cancellations":   prev_cancellations,
        "Previous_Successful_Bookings": prev_successful,
        "N_Special_Requests":       n_special_requests,
        "Special_Request_Detail":   special_request_detail,

        # Room
        "Room_No":                  room_no,
        "Room_Type_Reserved":       reserved_room_type,
        "Room_Type_Assigned":       room_type,
        "Got_Room_Upgrade":         upgrade_mask.astype(int),
        "Meal_Plan":                meal_plan,

        # Stay
        "Nights_Booked":            nights_booked,
        "Stayed_Room_Nights":       stayed_nights,

        # Pricing
        "Avg_Price_Per_Room":       avg_price_per_room,
        "Competitor_Avg_Rate":      competitor_avg_rate,
        "Rate_Premium_Pct":         rate_premium_pct,  # our price vs competitor (%)
        "Payment_Mode":             payment_mode,
        "OTA_Commission_Pct":       (ota_commission_pct * 100).round(1),

        # Revenue
        "Total_Lodging_Amount":     total_lodging,
        "Total_Lodging_Taxes":      lodging_tax,
        "Total_Amount":             total_amount,
        "Net_Revenue":              net_revenue,           # after OTA commission
        "Paid_at_Hotel_Amount":     paid_at_hotel,
        "Paid_To_OTA_Amount":       paid_to_ota,

        # Outcome
        "Is_Cancelled":             is_cancelled,
        "Cancel_Days_Before_Arrival": cancel_days_before,  # -1 if honoured
    })

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=70000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = generate(args.rows, args.seed)

    out = OUT_DIR / "median_inn_synthetic.csv"
    df.to_csv(out, index=False)
    df.head(50).to_csv(SAMPLE / "sample_50_rows.csv", index=False)

    print(f"Generated {len(df):,} rows, {df.shape[1]} columns → {out}")
    print(f"Cancellation rate:   {df['Is_Cancelled'].mean():.2%}")
    print(f"Direct cancel rate:  {df[df.Booking_Source=='Direct / Front Desk']['Is_Cancelled'].mean():.2%}")
    print(f"OTA cancel rate:     {df[df.Booking_Source=='OTA']['Is_Cancelled'].mean():.2%}")
    print(f"Corporate cancel:    {df[df.Booking_Source=='Corporate']['Is_Cancelled'].mean():.2%}")
    print(f"Repeat guests:       {(df.Primary_Guest_Name.value_counts()>1).sum():,} "
          f"({df.Primary_Guest_Name.duplicated().sum()/len(df):.1%} of rows)")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
