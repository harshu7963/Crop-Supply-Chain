# Data Dictionary — Syngenta India CP P&S Supply Chain Analytics

| Table Name | Grain | Primary / Composite Key | Column Name | Data Type | Business Rule & Domain Constraint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sku_master.csv` | 1 row per SKU | `sku_id` | `sku_id` | `VARCHAR(16)` | Format `SYN-SKU-001`..`SYN-SKU-050`. Unique primary key. |
| `sku_master.csv` | 1 row per SKU | `sku_id` | `product_family` | `VARCHAR(32)` | `Herbicides`, `Insecticides`, `Fungicides`, `Biologicals`. |
| `sku_master.csv` | 1 row per SKU | `sku_id` | `formulation_type` | `VARCHAR(8)` | `EC` (Emulsifiable Concentrate), `SC` (Suspension Concentrate), `WG` (Water Dispersible Granule), `SL`, `OD`. |
| `sku_master.csv` | 1 row per SKU | `sku_id` | `uom` | `VARCHAR(8)` | Raw extract contains `'L'`, `'Litre'`, `'ltr'`, `'KG'`, `'Kg'`, `'kg'`; canonicalized to `'L'` and `'KG'`. |
| `sku_master.csv` | 1 row per SKU | `sku_id` | `master_lead_time_days` | `FLOAT` | SAP static replenishment lead time (days). 8% missing in raw extract; imputed via family median. |
| `sku_master.csv` | 1 row per SKU | `sku_id` | `shelf_life_days` | `INT` | Total batch shelf life from manufacture date (270–450 days for Biologicals; 540–1,095 days for synthetics). |
| `sku_master.csv` | 1 row per SKU | `sku_id` | `ai_synthesis_lead_time_days` | `INT` | Upstream Active Ingredient synthesis lead time (90–120 days), motivating the $t+3$ forecast horizon. |
| `orders_deliveries.csv` | 1 row per shipment line | `(order_id, line_id)` | `order_id`, `line_id` | `VARCHAR`, `INT` | `line_id = 2` indicates a split shipment tranche; must aggregate to `order_id` before evaluating OTIF. |
| `orders_deliveries.csv` | 1 row per shipment line | `(order_id, line_id)` | `plant_id`, `toller_flag` | `VARCHAR`, `INT` | `PUNE-IH-01`, `GOA-IH-02` (`toller_flag=0`); `PANOLI-TL-01`, `DAHEJ-TL-02` (`toller_flag=1`). |
| `orders_deliveries.csv` | 1 row per shipment line | `(order_id, line_id)` | `shipping_mode` | `VARCHAR(24)` | `Express_Air_Road`, `Surface_FTL`, `Surface_PTL`, `Rail_Rake`. |
| `orders_deliveries.csv` | 1 row per shipment line | `(order_id, line_id)` | `order_date`, `promised_date`, `dispatched_date`, `delivered_date` | `DATE` | ISO `YYYY-MM-DD`. Post-order timestamps (`dispatched_date`, `delivered_date`) are forbidden in Classifier features. |
| `inventory_snapshots.csv` | 1 row per SKU-month | `(snapshot_month, sku_id)` | `on_hand_qty`, `remaining_shelf_life_days` | `INT` | Month-end physical stock and remaining batch shelf life (`shelf_life_days - batch_age_days`). |
| `inventory_snapshots.csv` | 1 row per SKU-month | `(snapshot_month, sku_id)` | `retail_secondary_sales_qty`, `distributor_order_qty`, `plant_production_order_qty` | `INT` | Multi-echelon monthly volume series used to compute the Bullwhip Effect ratio. |
| `external_weather.csv` | 1 row per region-month | `(year_month, customer_region)` | `rainfall_anomaly_pct`, `pest_pressure_index` | `FLOAT` | IMD percentage deviation from Long Period Average (LPA) and regional pest index (0–100). |
