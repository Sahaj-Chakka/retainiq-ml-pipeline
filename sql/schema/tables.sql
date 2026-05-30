-- =============================================================
-- RetainIQ · Schema  (matches data/synthetic/median_inn_synthetic.csv)
-- Synthetic hotel-booking data modeled on an anonymised real project.
-- =============================================================

CREATE TABLE IF NOT EXISTS bookings (
    reservation_id          VARCHAR(16)  PRIMARY KEY,
    hotel                   VARCHAR(64),
    booking_source          VARCHAR(32),     -- OTA / Corporate / Walk - In / Travel Agent / Web
    primary_guest_name      VARCHAR(64),
    corporate_name          VARCHAR(64),     -- NULL for non-corporate/OTA
    room_no                 INT,
    room_type               VARCHAR(32),     -- Maple (Deluxe) / Mahogany (Premium)
    pax                     VARCHAR(16),     -- e.g. '2(A) 0(C)'
    payment_mode            VARCHAR(16),     -- Prepaid / Pay at Hotel
    check_in_date           DATE,
    check_out_date          DATE,
    lead_time_days          INT,
    stayed_room_nights      INT,
    per_room_night_charges  NUMERIC(10,2),
    total_lodging_amount    NUMERIC(12,2),
    total_lodging_taxes     NUMERIC(12,2),
    total_amount            NUMERIC(12,2),
    paid_at_hotel_amount    NUMERIC(12,2),
    paid_to_ota_amount      NUMERIC(12,2),
    guest_invoice_issued_by VARCHAR(16),
    is_cancelled            SMALLINT         -- 1 = cancelled (total_amount = 0)
);

CREATE INDEX IF NOT EXISTS idx_bookings_checkin  ON bookings(check_in_date);
CREATE INDEX IF NOT EXISTS idx_bookings_source   ON bookings(booking_source);
CREATE INDEX IF NOT EXISTS idx_bookings_guest    ON bookings(primary_guest_name);
CREATE INDEX IF NOT EXISTS idx_bookings_cancel   ON bookings(is_cancelled);
