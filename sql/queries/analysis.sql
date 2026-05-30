-- =============================================================
-- RetainIQ · Analysis queries
-- =============================================================

-- 1. Cancellation rate by booking source
SELECT
    booking_source,
    COUNT(*)                                          AS total_bookings,
    SUM(is_cancelled)                                 AS cancellations,
    ROUND(AVG(is_cancelled)::numeric, 4)              AS cancellation_rate
FROM bookings
GROUP BY booking_source
ORDER BY cancellation_rate DESC;


-- 2. Revenue by month (honoured bookings only) — seasonality
SELECT
    DATE_TRUNC('month', check_in_date)::date          AS month,
    COUNT(*)                                          AS bookings,
    SUM(total_amount)                                 AS revenue,
    ROUND(AVG(total_amount)::numeric, 2)              AS avg_booking_value
FROM bookings
WHERE is_cancelled = 0
GROUP BY DATE_TRUNC('month', check_in_date)
ORDER BY month;


-- 3. Revenue concentration — top guests by lifetime value
WITH guest_value AS (
    SELECT primary_guest_name,
           SUM(total_amount)                          AS lifetime_revenue,
           COUNT(*)                                   AS bookings
    FROM bookings
    WHERE is_cancelled = 0
    GROUP BY primary_guest_name
),
ranked AS (
    SELECT *,
           NTILE(100) OVER (ORDER BY lifetime_revenue DESC) AS revenue_percentile
    FROM guest_value
)
SELECT
    CASE WHEN revenue_percentile <= 18 THEN 'Top 18%' ELSE 'Bottom 82%' END AS tier,
    COUNT(*)                                          AS guests,
    SUM(lifetime_revenue)                             AS revenue,
    ROUND(100.0 * SUM(lifetime_revenue)
          / SUM(SUM(lifetime_revenue)) OVER (), 1)    AS pct_of_total_revenue
FROM ranked
GROUP BY 1
ORDER BY revenue DESC;


-- 4. Room performance — which rooms drive bookings & revenue
SELECT
    room_no,
    room_type,
    COUNT(*)                                          AS bookings,
    SUM(total_amount)                                 AS revenue,
    ROUND(AVG(is_cancelled)::numeric, 3)              AS cancellation_rate
FROM bookings
GROUP BY room_no, room_type
ORDER BY bookings DESC;


-- 5. Corporate revenue contribution vs booking volume
SELECT
    booking_source,
    COUNT(*)                                          AS bookings,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_bookings,
    SUM(total_amount)                                 AS revenue,
    ROUND(100.0 * SUM(total_amount)
          / SUM(SUM(total_amount)) OVER (), 1)        AS pct_of_revenue
FROM bookings
WHERE is_cancelled = 0
GROUP BY booking_source
ORDER BY revenue DESC;


-- 6. Repeat-guest rate
SELECT
    COUNT(*) FILTER (WHERE booking_count > 1)         AS repeat_guests,
    COUNT(*)                                          AS total_guests,
    ROUND(100.0 * COUNT(*) FILTER (WHERE booking_count > 1)
          / COUNT(*), 1)                              AS repeat_rate_pct
FROM (
    SELECT primary_guest_name, COUNT(*) AS booking_count
    FROM bookings
    GROUP BY primary_guest_name
) t;
