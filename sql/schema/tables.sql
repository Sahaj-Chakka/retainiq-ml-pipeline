-- =============================================================================
-- RetainIQ · sql/schema/tables.sql
-- Schema matching the enriched 37-column synthetic dataset (v2)
-- Run: psql -f sql/schema/tables.sql
-- =============================================================================

CREATE TABLE IF NOT EXISTS bookings (
    -- Identifiers
    reservation_id              VARCHAR(16)     PRIMARY KEY,
    hotel                       VARCHAR(64),
    booking_date                DATE,
    check_in_date               DATE            NOT NULL,
    check_out_date              DATE,

    -- Booking characteristics
    booking_source              VARCHAR(32),        -- OTA / Corporate / Direct Front Desk / Travel Agent / Web
    market_segment              VARCHAR(32),        -- Online Leisure / Corporate / Direct / Group Agent
    corporate_name              VARCHAR(64),        -- OTA platform or corporate account name; NULL for others
    lead_time_days              INT,                -- days between booking_date and check_in_date

    -- Guest
    primary_guest_name          VARCHAR(64),
    n_adults                    SMALLINT,
    n_children                  SMALLINT,
    pax                         VARCHAR(16),        -- formatted "2(A) 1(C)"
    previous_cancellations      SMALLINT,
    previous_successful_bookings SMALLINT,
    n_special_requests          SMALLINT,
    special_request_detail      VARCHAR(64),

    -- Room
    room_no                     SMALLINT,
    room_type_reserved          VARCHAR(32),        -- Maple (Deluxe) / Mahogany (Premium)
    room_type_assigned          VARCHAR(32),
    got_room_upgrade            SMALLINT,           -- 1 = upgraded at check-in
    meal_plan                   VARCHAR(32),        -- Breakfast Included / No Meal / Half Board / Full Board

    -- Stay
    nights_booked               SMALLINT,
    stayed_room_nights          SMALLINT,           -- 0 if cancelled

    -- Pricing
    avg_price_per_room          NUMERIC(10,2),
    competitor_avg_rate         NUMERIC(10,2),
    rate_premium_pct            NUMERIC(8,2),       -- (our price / competitor - 1) * 100
    payment_mode                VARCHAR(16),        -- Prepaid / Pay at Hotel
    ota_commission_pct          NUMERIC(5,1),       -- 0 for non-OTA bookings

    -- Revenue (0 if cancelled)
    total_lodging_amount        NUMERIC(12,2),
    total_lodging_taxes         NUMERIC(12,2),
    total_amount                NUMERIC(12,2),
    net_revenue                 NUMERIC(12,2),      -- total_amount after OTA commission
    paid_at_hotel_amount        NUMERIC(12,2),
    paid_to_ota_amount          NUMERIC(12,2),

    -- Outcome
    is_cancelled                SMALLINT            NOT NULL, -- 1 = cancelled
    cancel_days_before_arrival  SMALLINT            -- days before check-in; -1 if honoured
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_bookings_checkin     ON bookings(check_in_date);
CREATE INDEX IF NOT EXISTS idx_bookings_source      ON bookings(booking_source);
CREATE INDEX IF NOT EXISTS idx_bookings_segment     ON bookings(market_segment);
CREATE INDEX IF NOT EXISTS idx_bookings_guest       ON bookings(primary_guest_name);
CREATE INDEX IF NOT EXISTS idx_bookings_cancelled   ON bookings(is_cancelled);
CREATE INDEX IF NOT EXISTS idx_bookings_lead_time   ON bookings(lead_time_days);
CREATE INDEX IF NOT EXISTS idx_bookings_corporate   ON bookings(corporate_name);
