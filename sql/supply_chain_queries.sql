-- ============================================================================
-- MODULE 3: SQL ANALYTICS & STAR-SCHEMA LAYER (sql/supply_chain_queries.sql)
-- Syngenta Crop Protection (CP P&S, India) - Supply Chain Decision Support
-- Dialect: PostgreSQL 15+ (with ANSI/MySQL 8.0+ compatible window functions)
-- ============================================================================

-- ============================================================================
-- QUERY 1: AUTOMATED DATA QUALITY AUDIT QUERY
-- Identifies duplicate order-line keys, orphan SKUs missing from sku_master,
-- impossible dates (promised_date < order_date or delivered_date < order_date),
-- negative quantities, missing master lead times, and non-standard UOM strings.
-- ============================================================================
WITH order_line_ranked AS (
    SELECT
        o.order_id,
        o.line_id,
        o.sku_id,
        o.plant_id,
        o.order_date,
        o.promised_date,
        o.dispatched_date,
        o.delivered_date,
        o.qty_ordered,
        o.qty_delivered,
        ROW_NUMBER() OVER (
            PARTITION BY o.order_id, o.line_id, o.sku_id, o.order_date, o.qty_delivered
            ORDER BY o.order_id
        ) AS dup_rank
    FROM orders_deliveries o
),
dq_violations AS (
    SELECT
        '1_DUPLICATE_ORDER_LINE_KEY' AS dq_rule_code,
        'Uniqueness' AS dq_dimension,
        COUNT(*) AS violation_count,
        (SELECT COUNT(*) FROM orders_deliveries) AS total_records_evaluated,
        'Deduplicate exact SAP IDoc retry records keeping dup_rank = 1' AS remediation_action
    FROM order_line_ranked
    WHERE dup_rank > 1

    UNION ALL

    SELECT
        '2_ORPHAN_SKU_MISSING_IN_MASTER' AS dq_rule_code,
        'Validity / Referential Integrity' AS dq_dimension,
        COUNT(*) AS violation_count,
        (SELECT COUNT(*) FROM orders_deliveries) AS total_records_evaluated,
        'Quarantine unmapped legacy SKU codes (SYN-SKU-998/999) to ERP exception queue' AS remediation_action
    FROM orders_deliveries o
    LEFT JOIN sku_master s ON o.sku_id = s.sku_id
    WHERE s.sku_id IS NULL

    UNION ALL

    SELECT
        '3_IMPOSSIBLE_DATES_CHRONOLOGY' AS dq_rule_code,
        'Validity / Timeliness' AS dq_dimension,
        COUNT(*) AS violation_count,
        (SELECT COUNT(*) FROM orders_deliveries) AS total_records_evaluated,
        'Exclude negative transit/promise rows from lead-time distribution modeling' AS remediation_action
    FROM orders_deliveries o
    WHERE o.promised_date < o.order_date
       OR o.delivered_date < o.order_date
       OR o.delivered_date < o.dispatched_date

    UNION ALL

    SELECT
        '4_NEGATIVE_ORDER_OR_DELIVERY_QTY' AS dq_rule_code,
        'Validity' AS dq_dimension,
        COUNT(*) AS violation_count,
        (SELECT COUNT(*) FROM orders_deliveries) AS total_records_evaluated,
        'Filter out credit-memo reversal lines (qty <= 0) from outbound OTIF fact table' AS remediation_action
    FROM orders_deliveries o
    WHERE o.qty_ordered <= 0 OR o.qty_delivered < 0

    UNION ALL

    SELECT
        '5_MISSING_MASTER_LEAD_TIME' AS dq_rule_code,
        'Completeness' AS dq_dimension,
        COUNT(*) AS violation_count,
        (SELECT COUNT(*) FROM sku_master) AS total_records_evaluated,
        'Impute missing master_lead_time_days using product_family median before ERP sync' AS remediation_action
    FROM sku_master s
    WHERE s.master_lead_time_days IS NULL

    UNION ALL

    SELECT
        '6_INCONSISTENT_UOM_STRINGS' AS dq_rule_code,
        'Consistency' AS dq_dimension,
        COUNT(*) AS violation_count,
        (SELECT COUNT(*) FROM sku_master) AS total_records_evaluated,
        'Standardize Litre/ltr -> L and Kg/kg -> KG via uppercase canonical mapping' AS remediation_action
    FROM sku_master s
    WHERE s.uom NOT IN ('L', 'KG')
)
SELECT
    dq_rule_code,
    dq_dimension,
    violation_count,
    total_records_evaluated,
    ROUND(100.0 * violation_count / NULLIF(total_records_evaluated, 0), 2) AS failure_rate_pct,
    remediation_action
FROM dq_violations
ORDER BY dq_rule_code;


-- ============================================================================
-- QUERY 2: MONTHLY OTIF & SPLIT-SHIPMENT AGGREGATION BY PLANT / TOLLER
-- Properly handles split deliveries by pre-aggregating shipment lines to the
-- (order_id, sku_id) grain BEFORE evaluating On-Time (last delivery <= promised)
-- and In-Full (sum of delivered qty >= ordered qty).
-- ============================================================================
WITH deduplicated_lines AS (
    SELECT DISTINCT
        o.order_id,
        o.line_id,
        o.sku_id,
        o.plant_id,
        o.toller_flag,
        o.customer_region,
        o.shipping_mode,
        o.order_date,
        o.promised_date,
        o.dispatched_date,
        o.delivered_date,
        o.qty_ordered,
        o.qty_delivered
    FROM orders_deliveries o
    INNER JOIN sku_master s ON o.sku_id = s.sku_id
    WHERE o.qty_ordered > 0
      AND o.qty_delivered >= 0
      AND o.promised_date >= o.order_date
      AND o.delivered_date >= o.order_date
),
order_level_fulfillment AS (
    SELECT
        d.order_id,
        d.sku_id,
        d.plant_id,
        d.toller_flag,
        SUBSTR(d.order_date, 1, 7) AS order_year_month,
        MAX(d.order_date) AS order_date,
        MAX(d.promised_date) AS promised_date,
        MAX(d.delivered_date) AS final_delivered_date,
        COUNT(d.line_id) AS shipment_splits_count,
        MAX(d.qty_ordered) AS qty_ordered,
        SUM(COALESCE(d.qty_delivered, 0)) AS total_qty_delivered,
        CASE
            WHEN MAX(d.delivered_date) <= MAX(d.promised_date) THEN 1
            ELSE 0
        END AS is_on_time,
        CASE
            WHEN SUM(COALESCE(d.qty_delivered, 0)) >= MAX(d.qty_ordered) THEN 1
            ELSE 0
        END AS is_in_full,
        CASE
            WHEN MAX(d.delivered_date) <= MAX(d.promised_date)
             AND SUM(COALESCE(d.qty_delivered, 0)) >= MAX(d.qty_ordered) THEN 1
            ELSE 0
        END AS is_otif
    FROM deduplicated_lines d
    GROUP BY
        d.order_id,
        d.sku_id,
        d.plant_id,
        d.toller_flag,
        SUBSTR(d.order_date, 1, 7)
)
SELECT
    order_year_month,
    plant_id,
    CASE WHEN toller_flag = 1 THEN 'External_Toller' ELSE 'In_House_Plant' END AS manufacturing_type,
    COUNT(order_id) AS total_orders,
    SUM(CASE WHEN shipment_splits_count > 1 THEN 1 ELSE 0 END) AS split_shipment_orders,
    ROUND(100.0 * SUM(is_on_time) / NULLIF(COUNT(order_id), 0), 2) AS on_time_pct,
    ROUND(100.0 * SUM(is_in_full) / NULLIF(COUNT(order_id), 0), 2) AS in_full_pct,
    ROUND(100.0 * SUM(is_otif) / NULLIF(COUNT(order_id), 0), 2) AS otif_pct,
    ROUND(
        100.0 * SUM(total_qty_delivered) / NULLIF(SUM(qty_ordered), 0),
        2
    ) AS unit_fill_rate_pct
FROM order_level_fulfillment
GROUP BY
    order_year_month,
    plant_id,
    toller_flag
ORDER BY
    order_year_month DESC,
    plant_id;


-- ============================================================================
-- QUERY 3: ABC / XYZ SKU SEGMENTATION IN SQL
-- Ranks 50 SKUs by cumulative revenue contribution (A: <=80%, B: 80-95%, C: >95%)
-- and monthly demand Coefficient of Variation CV = sigma / mu
-- (X: CV <= 0.45, Y: 0.45 < CV <= 0.80, Z: CV > 0.80).
-- ============================================================================
WITH sku_stats AS (
    SELECT
        i.sku_id,
        s.sku_name,
        s.product_family,
        s.formulation_type,
        s.unit_price_inr,
        SUM(i.distributor_order_qty) AS total_36m_units,
        SUM(i.distributor_order_qty * s.unit_price_inr) AS total_36m_revenue_inr,
        AVG(i.distributor_order_qty) AS mean_monthly_demand,
        SQRT(MAX(0.0, AVG(i.distributor_order_qty * i.distributor_order_qty)
                    - AVG(i.distributor_order_qty) * AVG(i.distributor_order_qty))) AS std_monthly_demand
    FROM inventory_snapshots i
    INNER JOIN sku_master s ON i.sku_id = s.sku_id
    GROUP BY i.sku_id, s.sku_name, s.product_family, s.formulation_type, s.unit_price_inr
),
pareto_calc AS (
    SELECT
        sku_id,
        sku_name,
        product_family,
        formulation_type,
        total_36m_units,
        ROUND(total_36m_revenue_inr / 10000000.0, 3) AS revenue_inr_crore,
        ROUND(mean_monthly_demand, 1) AS mean_monthly_demand,
        ROUND(std_monthly_demand, 1) AS std_monthly_demand,
        ROUND(std_monthly_demand / NULLIF(mean_monthly_demand, 0), 3) AS demand_cv,
        DENSE_RANK() OVER (ORDER BY total_36m_revenue_inr DESC) AS revenue_rank,
        SUM(total_36m_revenue_inr) OVER (
            ORDER BY total_36m_revenue_inr DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) / NULLIF(SUM(total_36m_revenue_inr) OVER (), 0) AS cum_revenue_share
    FROM sku_stats
)
SELECT
    sku_id,
    sku_name,
    product_family,
    formulation_type,
    revenue_rank,
    revenue_inr_crore,
    ROUND(100.0 * cum_revenue_share, 2) AS cum_revenue_pct,
    demand_cv,
    CASE
        WHEN cum_revenue_share <= 0.80 THEN 'A'
        WHEN cum_revenue_share <= 0.95 THEN 'B'
        ELSE 'C'
    END AS abc_class,
    CASE
        WHEN demand_cv <= 0.45 THEN 'X'
        WHEN demand_cv <= 0.80 THEN 'Y'
        ELSE 'Z'
    END AS xyz_class,
    (CASE
        WHEN cum_revenue_share <= 0.80 THEN 'A'
        WHEN cum_revenue_share <= 0.95 THEN 'B'
        ELSE 'C'
    END) || (CASE
        WHEN demand_cv <= 0.45 THEN 'X'
        WHEN demand_cv <= 0.80 THEN 'Y'
        ELSE 'Z'
    END) AS abc_xyz_segment
FROM pareto_calc
ORDER BY revenue_rank;


-- ============================================================================
-- QUERY 4: DAYS OF SUPPLY (DOH) & SLOW-MOVING / OBSOLETE STOCK (SLOB / FEFO)
-- Identifies active batches in the latest inventory snapshot (2025-12) where
-- current Days of Supply (DOH) exceeds remaining shelf life, quantifying exact
-- expiry write-off exposure in INR.
-- ============================================================================
WITH trailing_90d_demand AS (
    SELECT
        sku_id,
        AVG(distributor_order_qty) / 30.0 AS avg_daily_demand_units
    FROM inventory_snapshots
    WHERE snapshot_month IN ('2025-10', '2025-11', '2025-12')
    GROUP BY sku_id
),
latest_inventory AS (
    SELECT
        i.snapshot_month,
        i.plant_id,
        i.sku_id,
        s.sku_name,
        s.product_family,
        i.batch_id,
        i.on_hand_qty,
        i.batch_age_days,
        i.remaining_shelf_life_days,
        s.shelf_life_days AS total_shelf_life_days,
        s.unit_cost_inr,
        COALESCE(t.avg_daily_demand_units, 0.1) AS avg_daily_demand_units,
        ROUND(i.on_hand_qty / NULLIF(COALESCE(t.avg_daily_demand_units, 0.1), 0), 1) AS days_of_supply_doh
    FROM inventory_snapshots i
    INNER JOIN sku_master s ON i.sku_id = s.sku_id
    LEFT JOIN trailing_90d_demand t ON i.sku_id = t.sku_id
    WHERE i.snapshot_month = '2025-12'
)
SELECT
    snapshot_month,
    plant_id,
    sku_id,
    sku_name,
    product_family,
    batch_id,
    on_hand_qty,
    ROUND(avg_daily_demand_units, 2) AS avg_daily_demand,
    days_of_supply_doh,
    remaining_shelf_life_days,
    ROUND(days_of_supply_doh - remaining_shelf_life_days, 1) AS excess_days_beyond_expiry,
    ROUND(
        MAX(0.0, on_hand_qty - (avg_daily_demand_units * remaining_shelf_life_days)),
        0
    ) AS projected_expired_units,
    ROUND(
        MAX(0.0, on_hand_qty - (avg_daily_demand_units * remaining_shelf_life_days)) * unit_cost_inr,
        2
    ) AS write_off_risk_inr,
    CASE
        WHEN days_of_supply_doh > remaining_shelf_life_days AND remaining_shelf_life_days <= 90
            THEN 'CRITICAL_FEFO_INTERVENTION'
        WHEN days_of_supply_doh > remaining_shelf_life_days
            THEN 'HIGH_SLOB_WRITE_OFF_RISK'
        WHEN days_of_supply_doh > 90
            THEN 'WATCH_OVERSTOCK'
        ELSE 'HEALTHY_TURNOVER'
    END AS fefo_action_status
FROM latest_inventory
WHERE days_of_supply_doh > remaining_shelf_life_days
ORDER BY write_off_risk_inr DESC;


-- ============================================================================
-- QUERY 5: MULTI-ECHELON BULLWHIP EFFECT DETECTOR
-- Compares demand Coefficient of Variation (CV) across Retail Secondary Sales,
-- Distributor Primary Orders, and Plant Production Orders per SKU and Family,
-- computing the Bullwhip Amplification Ratio (CV_Plant / CV_Retail).
-- ============================================================================
WITH echelon_variability AS (
    SELECT
        i.sku_id,
        s.sku_name,
        s.product_family,
        AVG(i.retail_secondary_sales_qty) AS mean_retail,
        SQRT(MAX(0.0, AVG(i.retail_secondary_sales_qty * i.retail_secondary_sales_qty)
                    - AVG(i.retail_secondary_sales_qty) * AVG(i.retail_secondary_sales_qty))) AS std_retail,
        AVG(i.distributor_order_qty) AS mean_distributor,
        SQRT(MAX(0.0, AVG(i.distributor_order_qty * i.distributor_order_qty)
                    - AVG(i.distributor_order_qty) * AVG(i.distributor_order_qty))) AS std_distributor,
        AVG(i.plant_production_order_qty) AS mean_plant_prod,
        SQRT(MAX(0.0, AVG(i.plant_production_order_qty * i.plant_production_order_qty)
                    - AVG(i.plant_production_order_qty) * AVG(i.plant_production_order_qty))) AS std_plant_prod
    FROM inventory_snapshots i
    INNER JOIN sku_master s ON i.sku_id = s.sku_id
    GROUP BY i.sku_id, s.sku_name, s.product_family
),
cv_summary AS (
    SELECT
        sku_id,
        sku_name,
        product_family,
        ROUND(std_retail / NULLIF(mean_retail, 0), 3) AS cv_retail_secondary,
        ROUND(std_distributor / NULLIF(mean_distributor, 0), 3) AS cv_distributor_primary,
        ROUND(std_plant_prod / NULLIF(mean_plant_prod, 0), 3) AS cv_plant_production
    FROM echelon_variability
)
SELECT
    sku_id,
    sku_name,
    product_family,
    cv_retail_secondary,
    cv_distributor_primary,
    cv_plant_production,
    ROUND(cv_distributor_primary / NULLIF(cv_retail_secondary, 0), 2) AS distributor_bullwhip_ratio,
    ROUND(cv_plant_production / NULLIF(cv_retail_secondary, 0), 2) AS total_supply_chain_bullwhip_ratio,
    CASE
        WHEN (cv_plant_production / NULLIF(cv_retail_secondary, 0)) >= 2.5 THEN 'SEVERE_BATCHING_WHIPLASH'
        WHEN (cv_plant_production / NULLIF(cv_retail_secondary, 0)) >= 1.5 THEN 'MODERATE_AMPLIFICATION'
        ELSE 'STABLE_SYNCHRONIZED'
    END AS bullwhip_diagnostic
FROM cv_summary
ORDER BY total_supply_chain_bullwhip_ratio DESC;


-- ============================================================================
-- QUERY 6: MOM DEMAND GROWTH & 3-MONTH ROLLING AVERAGES PER SKU
-- Uses LAG() and windowed AVG() OVER (ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
-- to track seasonal transitions into Kharif and Rabi peaks.
-- ============================================================================
WITH monthly_series AS (
    SELECT
        i.sku_id,
        s.sku_name,
        s.product_family,
        i.snapshot_month AS year_month,
        i.distributor_order_qty AS monthly_demand_qty,
        LAG(i.distributor_order_qty, 1) OVER (
            PARTITION BY i.sku_id
            ORDER BY i.snapshot_month
        ) AS prev_month_demand_qty,
        LAG(i.distributor_order_qty, 12) OVER (
            PARTITION BY i.sku_id
            ORDER BY i.snapshot_month
        ) AS yoy_same_month_demand_qty,
        AVG(i.distributor_order_qty) OVER (
            PARTITION BY i.sku_id
            ORDER BY i.snapshot_month
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_3m_avg_demand
    FROM inventory_snapshots i
    INNER JOIN sku_master s ON i.sku_id = s.sku_id
)
SELECT
    sku_id,
    sku_name,
    product_family,
    year_month,
    monthly_demand_qty,
    prev_month_demand_qty,
    ROUND(
        100.0 * (monthly_demand_qty - prev_month_demand_qty)
        / NULLIF(prev_month_demand_qty, 0),
        2
    ) AS mom_growth_pct,
    ROUND(
        100.0 * (monthly_demand_qty - yoy_same_month_demand_qty)
        / NULLIF(yoy_same_month_demand_qty, 0),
        2
    ) AS yoy_growth_pct,
    ROUND(rolling_3m_avg_demand, 1) AS rolling_3m_avg_demand
FROM monthly_series
ORDER BY sku_id, year_month;