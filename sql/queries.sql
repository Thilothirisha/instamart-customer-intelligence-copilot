-- Each query starts with a "-- name: Qn" marker; src/part_a_sql.py splits on it.
-- Only completed orders count as revenue / spend. All amounts are INR.

-- name: Q1
-- WHERE filters rows before grouping; HAVING filters the groups afterwards.
SELECT customer_id,
       COUNT(*)          AS big_orders,
       SUM(order_amount) AS big_order_total
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY big_order_total DESC;

-- name: Q2
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products  p ON p.product_id  = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- name: Q3
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- name: Q4
-- The status filter lives in the ON clause so customers without completed orders survive the LEFT JOIN.
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
       ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- name: Q5
WITH spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total FROM spend
)
SELECT s.customer_id,
       ROUND(s.total_spend, 2) AS total_spend,
       ROUND(a.avg_total, 2)   AS avg_total
FROM spend s
CROSS JOIN avg_spend a
WHERE s.total_spend > a.avg_total
ORDER BY s.total_spend DESC;

-- name: Q6
WITH spend AS (
    SELECT c.customer_id, c.city, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend
ORDER BY city, row_num;

-- name: Q7
WITH daily AS (
    SELECT substr(order_ts, 1, 10) AS day,
           ROUND(SUM(order_amount), 2) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT day, revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day;
