-- ============================================================================
-- CONFORMED STAR-SCHEMA DDL FOR QLIK SAAS & POWER BI (dashboard/star_schema_ddl.sql)
-- Eliminates Synthetic Keys ($Syn) in Qlik Associative Engine via Canonical Bridge Key
-- ============================================================================

CREATE TABLE Dim_Calendar (
    date_key            DATE PRIMARY KEY,
    year_month          VARCHAR(7) NOT NULL,
    calendar_year       INT NOT NULL,
    calendar_quarter    VARCHAR(2) NOT NULL,
    month_num           INT NOT NULL,
    agri_season         VARCHAR(16) NOT NULL -- 'Kharif' (Jun-Oct), 'Rabi' (Nov-Mar), 'Zaid' (Apr-May)
);

CREATE TABLE Dim_SKU (
    sku_id                      VARCHAR(16) PRIMARY KEY,
    sku_name                    VARCHAR(64) NOT NULL,
    product_family              VARCHAR(32) NOT NULL,
    formulation_type            VARCHAR(8) NOT NULL,
    pack_size                   NUMERIC(8,2) NOT NULL,
    canonical_uom               VARCHAR(8) NOT NULL,
    master_lead_time_days       NUMERIC(6,1) NOT NULL,
    moq                         INT NOT NULL,
    shelf_life_days             INT NOT NULL,
    unit_cost_inr               NUMERIC(12,2) NOT NULL,
    unit_price_inr              NUMERIC(12,2) NOT NULL,
    ai_synthesis_lead_time_days INT NOT NULL,
    abc_class                   CHAR(1) NOT NULL,
    xyz_class                   CHAR(1) NOT NULL,
    abc_xyz_segment             CHAR(2) NOT NULL
);

CREATE TABLE Dim_Plant_Toller (
    plant_id            VARCHAR(16) PRIMARY KEY,
    plant_name          VARCHAR(64) NOT NULL,
    state_location      VARCHAR(32) NOT NULL,
    toller_flag         SMALLINT NOT NULL,
    manufacturing_type  VARCHAR(24) NOT NULL -- 'In_House_Plant' or 'External_Toller'
);

CREATE TABLE Dim_Region_Weather (
    region_ym_key                   VARCHAR(32) PRIMARY KEY,
    year_month                      VARCHAR(7) NOT NULL,
    customer_region                 VARCHAR(32) NOT NULL,
    imd_rainfall_mm                 NUMERIC(8,1) NOT NULL,
    long_period_avg_mm              NUMERIC(8,1) NOT NULL,
    rainfall_anomaly_pct            NUMERIC(6,2) NOT NULL,
    monsoon_onset_deviation_days    INT NOT NULL,
    pest_pressure_index             NUMERIC(5,1) NOT NULL
);

CREATE TABLE Fact_Orders_Fulfillment (
    order_id                VARCHAR(24) PRIMARY KEY,
    sku_plant_month_key     VARCHAR(64) NOT NULL,
    sku_id                  VARCHAR(16) NOT NULL REFERENCES Dim_SKU(sku_id),
    plant_id                VARCHAR(16) NOT NULL REFERENCES Dim_Plant_Toller(plant_id),
    region_ym_key           VARCHAR(32) NOT NULL REFERENCES Dim_Region_Weather(region_ym_key),
    order_date              DATE NOT NULL REFERENCES Dim_Calendar(date_key),
    promised_date           DATE NOT NULL,
    dispatched_date         DATE NOT NULL,
    final_delivered_date    DATE NOT NULL,
    shipping_mode           VARCHAR(24) NOT NULL,
    split_shipments_count   INT NOT NULL,
    qty_ordered             INT NOT NULL,
    total_qty_delivered     INT NOT NULL,
    actual_lead_time_days   INT NOT NULL,
    promised_lead_time_days INT NOT NULL,
    is_on_time              SMALLINT NOT NULL,
    is_in_full              SMALLINT NOT NULL,
    is_otif                 SMALLINT NOT NULL,
    pred_late_prob          NUMERIC(5,4),
    dq_clean_flag           SMALLINT NOT NULL DEFAULT 1
);

CREATE TABLE Fact_Inventory_Forecast (
    snapshot_sku_key            VARCHAR(32) PRIMARY KEY,
    sku_plant_month_key         VARCHAR(64) NOT NULL,
    snapshot_month              VARCHAR(7) NOT NULL,
    sku_id                      VARCHAR(16) NOT NULL REFERENCES Dim_SKU(sku_id),
    plant_id                    VARCHAR(16) NOT NULL REFERENCES Dim_Plant_Toller(plant_id),
    batch_id                    VARCHAR(24) NOT NULL,
    on_hand_qty                 INT NOT NULL,
    batch_age_days              INT NOT NULL,
    remaining_shelf_life_days   INT NOT NULL,
    actual_demand_qty           INT NOT NULL,
    fcst_snaive_t3_qty          NUMERIC(12,1),
    fcst_ets_croston_t3_qty     NUMERIC(12,1),
    fcst_lgbm_t3_qty            NUMERIC(12,1),
    avg_daily_demand            NUMERIC(10,2),
    projected_sell_through_days NUMERIC(8,1),
    ss_sap_static_units         NUMERIC(12,0),
    ss_dynamic_actual_units     NUMERIC(12,0),
    rop_dynamic_actual_units    NUMERIC(12,0),
    is_expiry_risk              SMALLINT NOT NULL
);
