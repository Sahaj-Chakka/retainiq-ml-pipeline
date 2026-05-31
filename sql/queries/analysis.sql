-- =============================================================================
-- RetainIQ · sql/queries/analysis.sql
-- Analysis queries — mirrors the Python analysis modules in src/analysis/
-- All column names match the enriched synthetic schema (v2)
-- =============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- 1. LOGISTIC REGRESSION INPUT: pre-booking cancellation signals
--    Used as inputs for the primary logistic regression analysis
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    reservation_id,
    lead_time_days,
    n_special_requests,
    previous_cancellations,
    previous_successful_bookings,
    n_adults,
    n_children,
    booking_source,
    market_segment,
    room_type_reserved,
    meal_plan,
    payment_mode,
    avg_price_per_room,
    rate_premium_pct,
    nights_booked,
    got_room_upgrade,
    EXTRACT(MONTH FROM check_in_date)::INT          AS checkin_month,
    EXTRACT(DOW   FROM check_in_date)::INT          AS checkin_dow,
    CASE WHEN EXTRACT(DOW FROM check_in_date) >= 5
         THEN 1 ELSE 0 END                          AS is_weekend,
    CASE WHEN EXTRACT(MONTH FROM check_in_date) IN (4,9,10,12)
         THEN 1 ELSE 0 END                          AS is_peak_month,
    is_cancelled
FROM bookings;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. CANCELLATION RATE BY BOOKING SOURCE
--    Key finding: Direct/Corporate cancel far less than OTA/Web
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    booking_source,
    market_segment,
    COUNT(*)                                        AS total_bookings,
    SUM(is_cancelled)                               AS cancellations,
    ROUND(AVG(is_cancelled)::NUMERIC, 4)            AS cancellation_rate,
    ROUND(AVG(lead_time_days)::NUMERIC, 1)          AS avg_lead_time_days
FROM bookings
GROUP BY booking_source, market_segment
ORDER BY cancellation_rate DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. LEAD-TIME BUCKETS: cancellation risk by booking horizon
--    Supports: non-refundable deposit policy simulation
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    CASE
        WHEN lead_time_days BETWEEN 0  AND 7   THEN '0-7 days'
        WHEN lead_time_days BETWEEN 8  AND 14  THEN '8-14 days'
        WHEN lead_time_days BETWEEN 15 AND 30  THEN '15-30 days'
        WHEN lead_time_days BETWEEN 31 AND 60  THEN '31-60 days'
        ELSE '60+ days'
    END                                             AS lead_time_bucket,
    COUNT(*)                                        AS bookings,
    SUM(is_cancelled)                               AS cancellations,
    ROUND(AVG(is_cancelled)::NUMERIC, 4)            AS cancellation_rate
FROM bookings
GROUP BY 1
ORDER BY MIN(lead_time_days);


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. MONTHLY REVENUE (honoured bookings)
--    Feeds Prophet time-series forecasting; shows April/September seasonality
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    DATE_TRUNC('month', check_in_date)::DATE        AS month,
    COUNT(*)                                        AS bookings,
    SUM(total_amount)                               AS gross_revenue,
    SUM(net_revenue)                                AS net_revenue_after_commission,
    ROUND(AVG(total_amount)::NUMERIC, 2)            AS avg_booking_value,
    ROUND(AVG(avg_price_per_room)::NUMERIC, 2)      AS avg_daily_rate
FROM bookings
WHERE is_cancelled = 0
GROUP BY DATE_TRUNC('month', check_in_date)
ORDER BY month;


-- ─────────────────────────────────────────────────────────────────────────────
-- 5. REVENUE VS NET REVENUE BY SOURCE (OTA commission cost)
--    Key finding: OTA bookings net 15-20% less after commission
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    booking_source,
    COUNT(*)                                        AS bookings,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct_of_bookings,
    SUM(total_amount)                               AS gross_revenue,
    SUM(net_revenue)                                AS net_revenue,
    SUM(total_amount - net_revenue)                 AS commission_paid,
    ROUND(100.0 * SUM(net_revenue)
          / SUM(SUM(net_revenue)) OVER(), 1)        AS pct_of_net_revenue,
    ROUND(AVG(ota_commission_pct)::NUMERIC, 1)      AS avg_commission_pct
FROM bookings
WHERE is_cancelled = 0
GROUP BY booking_source
ORDER BY net_revenue DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- 6. REVPAR: Revenue Per Available Room (14-room inventory)
--    ADR × Occupancy rate — standard hotel KPI
-- ─────────────────────────────────────────────────────────────────────────────
WITH daily AS (
    SELECT
        check_in_date,
        SUM(stayed_room_nights)                     AS rooms_sold,
        AVG(avg_price_per_room)                     AS adr
    FROM bookings
    WHERE is_cancelled = 0
    GROUP BY check_in_date
)
SELECT
    check_in_date                                   AS date,
    rooms_sold,
    ROUND(adr::NUMERIC, 2)                          AS avg_daily_rate,
    ROUND((rooms_sold::NUMERIC / 14), 4)            AS occupancy_rate,
    ROUND((adr * rooms_sold::NUMERIC / 14)::NUMERIC, 2) AS revpar
FROM daily
ORDER BY check_in_date;


-- ─────────────────────────────────────────────────────────────────────────────
-- 7. RFM GUEST FEATURES
--    Input to RFM segmentation (Recency, Frequency, Monetary)
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    primary_guest_name,
    CURRENT_DATE - MAX(check_in_date)               AS recency_days,
    COUNT(DISTINCT reservation_id)                  AS frequency,
    SUM(total_amount)                               AS monetary_total,
    ROUND(AVG(total_amount)::NUMERIC, 2)            AS avg_booking_value,
    ROUND(AVG(is_cancelled)::NUMERIC, 4)            AS historical_cancel_rate,
    ROUND(AVG(n_special_requests)::NUMERIC, 2)      AS avg_special_requests
FROM bookings
GROUP BY primary_guest_name
ORDER BY monetary_total DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- 8. ROOM PERFORMANCE (occupancy + cancellation by room number)
--    Which specific rooms are under/over-booked?
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    room_no,
    room_type_assigned,
    COUNT(*)                                        AS total_bookings,
    SUM(stayed_room_nights)                         AS total_nights,
    ROUND(AVG(is_cancelled)::NUMERIC, 3)            AS cancellation_rate,
    ROUND(AVG(avg_price_per_room)::NUMERIC, 2)      AS avg_nightly_rate,
    SUM(net_revenue)                                AS total_net_revenue
FROM bookings
GROUP BY room_no, room_type_assigned
ORDER BY total_bookings DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- 9. GUEST HISTORY: previous cancellations vs outcome
--    Validates the logistic regression finding (OR 1.52× per prior cancellation)
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    previous_cancellations,
    COUNT(*)                                        AS bookings,
    ROUND(AVG(is_cancelled)::NUMERIC, 4)            AS cancellation_rate
FROM bookings
GROUP BY previous_cancellations
ORDER BY previous_cancellations;


-- ─────────────────────────────────────────────────────────────────────────────
-- 10. REVENUE CONCENTRATION (Pareto)
--     Top 18% of guests drive ~50% of revenue
-- ─────────────────────────────────────────────────────────────────────────────
WITH guest_value AS (
    SELECT
        primary_guest_name,
        SUM(total_amount)                           AS lifetime_revenue,
        COUNT(*)                                    AS bookings
    FROM bookings
    WHERE is_cancelled = 0
    GROUP BY primary_guest_name
),
ranked AS (
    SELECT *,
        NTILE(100) OVER (ORDER BY lifetime_revenue DESC) AS centile
    FROM guest_value
)
SELECT
    CASE WHEN centile <= 18 THEN 'Top 18%' ELSE 'Bottom 82%' END AS guest_tier,
    COUNT(*)                                        AS guests,
    SUM(lifetime_revenue)                           AS revenue,
    ROUND(100.0 * SUM(lifetime_revenue)
          / SUM(SUM(lifetime_revenue)) OVER(), 1)   AS pct_of_total_revenue
FROM ranked
GROUP BY 1
ORDER BY revenue DESC;
