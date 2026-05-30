-- =============================================================
-- RetainIQ · Views for forecasting & BI dashboards
-- =============================================================

-- Daily revenue series feeding the Prophet forecaster (ds / y convention)
CREATE OR REPLACE VIEW v_revenue_daily AS
SELECT
    check_in_date                AS ds,
    SUM(total_amount)            AS y,
    COUNT(*)                     AS bookings
FROM bookings
WHERE is_cancelled = 0
GROUP BY check_in_date
ORDER BY check_in_date;

-- Booking-source performance (for the BI dashboard)
CREATE OR REPLACE VIEW v_source_performance AS
SELECT
    booking_source,
    COUNT(*)                                          AS bookings,
    SUM(is_cancelled)                                 AS cancellations,
    ROUND(AVG(is_cancelled)::numeric, 4)              AS cancellation_rate,
    SUM(total_amount)                                 AS revenue
FROM bookings
GROUP BY booking_source;

-- Guest lifetime value tiers
CREATE OR REPLACE VIEW v_guest_value AS
SELECT
    primary_guest_name,
    COUNT(*)                                          AS bookings,
    SUM(total_amount)                                 AS lifetime_revenue,
    ROUND(AVG(is_cancelled)::numeric, 3)              AS cancellation_rate
FROM bookings
GROUP BY primary_guest_name;
