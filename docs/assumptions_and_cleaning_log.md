# Cleaning & Modeling Assumptions Log — Syngenta CP P&S Supply Chain Project

1. **SAP IDoc Retry Deduplication (`Uniqueness`):** 750 exact duplicate `(order_id, line_id)` records (1.42% of raw rows) are logged and deduplicated prior to OTIF aggregation and ML feature engineering.
2. **Orphan SKU Quarantine (`Referential Integrity`):** 185 rows (0.35%) referencing unmapped legacy phase-out codes (`SYN-SKU-998`, `SYN-SKU-999`) are routed to an ERP Master Data exception queue rather than silently dropped.
3. **Credit-Memo Reversal Isolation (`Validity`):** 153 rows (0.29%) with negative `qty_ordered` / `qty_delivered` represent financial return adjustments and are excluded from physical delivery OTIF denominators.
4. **Backdated Timestamp Quarantine (`Timeliness`):** 282 rows (0.53%) where `delivered_date < order_date` or `promised_date < order_date` are quarantined before fitting empirical lead-time distributions.
5. **Split-Delivery Header Aggregation:** 2,033 split shipment lines (~4.0% of orders) are pre-aggregated to `order_id` (`MAX(delivered_date)` and `SUM(qty_delivered)`) so partial first deliveries are never misclassified as full orders.
6. **Zero-Leakage Lagging of Historical Plant Late Rates:** Because external toller lead times reach 24 days at P90, `plant_hist_late_rate_lagged` shifts historical outcomes backward by 15 orders (~21 days) so only completed deliveries known at `order_date` enter the feature.
7. **3-Month ($t+3$) Planning Horizon Alignment:** Because Active Ingredient synthesis requires 90–120 days, all demand forecasters are evaluated strictly at $t+3$ using features known at origin $t$ (`lag >= 3`).
