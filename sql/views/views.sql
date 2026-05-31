-- =============================================================================
-- RetainIQ · sql/views/views.sql
-- Materialised views for the dashboard, BI tools, and forecasting pipeline
-- All column names match enriched synthetic schema (v2)
-- =============================================================================


-- Prophet forecasting input (ds / y convention)
CREATE OR REPLACE VIEW v_revenue_daily AS
SELECT
    check_in_date                                   AS ds,
    SUM(total_amount)                               AS y,
    SUM(net_revenue)                                AS net_y,
    COUNT(*)                                        AS bookings,
    ROUND(AVG(avg_price_per_room)::NUMERIC, 2)      AS adr
FROM bookings
WHERE is_cancelled = 0
GROUP BY check_in_date
ORDER BY check_in_date;


-- Booking source performance — used in dashboard + Tableau extracts
CREATE OR REPLACE VIEW v_source_performance AS
SELECT
    booking_source,
    market_segment,
    COUNT(*)                                        AS bookings,
    SUM(is_cancelled)                               AS cancellations,
    ROUND(AVG(is_cancelled)::NUMERIC, 4)            AS cancellation_rate,
    SUM(total_amount)                               AS gross_revenue,
    SUM(net_revenue)                                AS net_revenue,
    ROUND(AVG(lead_time_days)::NUMERIC, 1)          AS avg_lead_time
FROM bookings
GROUP BY booking_source, market_segment;


-- Lead-time risk buckets — supports deposit policy simulation
CREATE OR REPLACE VIEW v_lead_time_risk AS
SELECT
    CASE
        WHEN lead_time_days BETWEEN 0  AND 7   THEN '0-7 days'
        WHEN lead_time_days BETWEEN 8  AND 14  THEN '8-14 days'
        WHEN lead_time_days BETWEEN 15 AND 30  THEN '15-30 days'
        WHEN lead_time_days BETWEEN 31 AND 60  THEN '31-60 days'
        ELSE '60+ days'
    END                                             AS lead_bucket,
    COUNT(*)                                        AS bookings,
    ROUND(AVG(is_cancelled)::NUMERIC, 4)            AS cancellation_rate,
    SUM(CASE WHEN is_cancelled=1 THEN total_amount ELSE 0 END) AS revenue_at_risk
FROM bookings
GROUP BY 1;


-- Guest lifetime value — feeds RFM segmentation
CREATE OR REPLACE VIEW v_guest_lifetime_value AS
SELECT
    primary_guest_name,
    COUNT(DISTINCT reservation_id)                  AS total_bookings,
    SUM(total_amount)                               AS lifetime_revenue,
    ROUND(AVG(total_amount)::NUMERIC, 2)            AS avg_booking_value,
    MAX(check_in_date)                              AS last_stay_date,
    CURRENT_DATE - MAX(check_in_date)               AS recency_days,
    ROUND(AVG(is_cancelled)::NUMERIC, 4)            AS cancel_rate,
    SUM(n_special_requests)                         AS total_special_requests
FROM bookings
GROUP BY primary_guest_name;


-- Monthly RevPAR — ADR × Occupancy (14-room inventory)
CREATE OR REPLACE VIEW v_monthly_revpar AS
SELECT
    DATE_TRUNC('month', check_in_date)::DATE        AS month,
    ROUND(AVG(avg_price_per_room)::NUMERIC, 2)      AS adr,
    SUM(stayed_room_nights)                         AS rooms_sold,
    EXTRACT(DAYS FROM
        DATE_TRUNC('month', check_in_date)
        + INTERVAL '1 month'
        - DATE_TRUNC('month', check_in_date)
    ) * 14                                          AS total_capacity,
    ROUND(
        SUM(stayed_room_nights)::NUMERIC /
        NULLIF(EXTRACT(DAYS FROM
            DATE_TRUNC('month', check_in_date)
            + INTERVAL '1 month'
            - DATE_TRUNC('month', check_in_date)
        ) * 14, 0), 4)                              AS occupancy_rate,
    ROUND((AVG(avg_price_per_room) *
        SUM(stayed_room_nights)::NUMERIC /
        NULLIF(EXTRACT(DAYS FROM
            DATE_TRUNC('month', check_in_date)
            + INTERVAL '1 month'
            - DATE_TRUNC('month', check_in_date)
        ) * 14, 0))::NUMERIC, 2)                    AS revpar
FROM bookings
WHERE is_cancelled = 0
GROUP BY DATE_TRUNC('month', check_in_date)
ORDER BY month;
