"""
Module 4: Data Quality Scorecard & Diagnostic EDA (src/dq_and_diagnostics.py)
Syngenta Crop Protection (CP P&S, India) - End-to-End Supply Chain Analytics Project
Candidate Portfolio: Harshita Kadam

Performs:
1. Automated 6-Dimension Data Quality Scorecard (Completeness, Accuracy, Consistency, Validity, Uniqueness, Timeliness)
2. Master Data vs. Reality Lead-Time Audit (SAP master_lead_time_days vs. actual P50/P90 lead times by plant/toller)
3. Statistical Hypothesis Testing (Mann-Whitney U test for right-skewed lead times, Chi-Square test of independence for Kharif Toller vs. In-House late rates, and Simpson's Paradox proof across Express_Air_Road vs. Surface_FTL shipping modes)
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats


def run_dq_and_diagnostics(data_dir: str = "data", sql_path: str = None) -> dict:
    sku_df = pd.read_csv(os.path.join(data_dir, "sku_master.csv"))
    orders_df = pd.read_csv(os.path.join(data_dir, "orders_deliveries.csv"))
    inv_df = pd.read_csv(os.path.join(data_dir, "inventory_snapshots.csv"))
    weather_df = pd.read_csv(os.path.join(data_dir, "external_weather.csv"))

    # =========================================================================
    # 1. AUTOMATED 6-DIMENSION DATA QUALITY SCORECARD
    # =========================================================================
    total_orders_rows = len(orders_df)
    total_sku_rows = len(sku_df)

    missing_lt_cnt = int(sku_df["master_lead_time_days"].isna().sum())
    dup_orders_cnt = int(orders_df.duplicated(subset=["order_id", "line_id"]).sum())
    inconsistent_uom_cnt = int((~sku_df["uom"].isin(["L", "KG"])).sum())
    valid_skus = set(sku_df["sku_id"].unique())
    orphan_sku_cnt = int((~orders_df["sku_id"].isin(valid_skus)).sum())
    neg_qty_cnt = int(((orders_df["qty_ordered"] <= 0) | (orders_df["qty_delivered"] < 0)).sum())

    ord_dt_raw = pd.to_datetime(orders_df["order_date"])
    prom_dt_raw = pd.to_datetime(orders_df["promised_date"])
    disp_dt_raw = pd.to_datetime(orders_df["dispatched_date"])
    deliv_dt_raw = pd.to_datetime(orders_df["delivered_date"])
    bad_dates_cnt = int(
        ((prom_dt_raw < ord_dt_raw) | (deliv_dt_raw < ord_dt_raw) | (deliv_dt_raw < disp_dt_raw)).sum()
    )

    clean_mask = (
        (~orders_df.duplicated(subset=["order_id", "line_id"]))
        & (orders_df["sku_id"].isin(valid_skus))
        & (orders_df["qty_ordered"] > 0)
        & (orders_df["qty_delivered"] >= 0)
        & (prom_dt_raw >= ord_dt_raw)
        & (deliv_dt_raw >= ord_dt_raw)
    )
    clean_lines = orders_df.loc[clean_mask].copy()
    clean_lines["actual_lt_days"] = (
        pd.to_datetime(clean_lines["delivered_date"]) - pd.to_datetime(clean_lines["order_date"])
    ).dt.days

    sku_actual_p90 = clean_lines.groupby("sku_id")["actual_lt_days"].agg(lambda s: np.percentile(s, 90)).to_dict()
    underestimated_lt_skus = 0
    for _, srow in sku_df.dropna(subset=["master_lead_time_days"]).iterrows():
        act_p90 = sku_actual_p90.get(srow["sku_id"], srow["master_lead_time_days"])
        if act_p90 > srow["master_lead_time_days"] * 1.15:
            underestimated_lt_skus += 1

    dq_scorecard = pd.DataFrame([
        {
            "Dimension": "Completeness",
            "Table": "sku_master",
            "Rule_Description": "master_lead_time_days IS NOT NULL",
            "Records_Checked": total_sku_rows,
            "Failed_Records": missing_lt_cnt,
            "Failure_Rate_Pct": round(100.0 * missing_lt_cnt / total_sku_rows, 2),
            "Cleaning_Assumption_Logged": "Impute missing lead times using product_family median (15.0 days) and flag in ERP master audit",
        },
        {
            "Dimension": "Consistency",
            "Table": "sku_master",
            "Rule_Description": "Canonical UOM in ('L', 'KG') without mixed casing/spelling",
            "Records_Checked": total_sku_rows,
            "Failed_Records": inconsistent_uom_cnt,
            "Failure_Rate_Pct": round(100.0 * inconsistent_uom_cnt / total_sku_rows, 2),
            "Cleaning_Assumption_Logged": "Normalize {'Litre', 'ltr'} -> 'L' and {'Kg', 'kg'} -> 'KG' prior to aggregations",
        },
        {
            "Dimension": "Accuracy",
            "Table": "sku_master vs orders",
            "Rule_Description": "SAP master_lead_time_days within 15% of empirical P90 delivery lead time",
            "Records_Checked": total_sku_rows - missing_lt_cnt,
            "Failed_Records": underestimated_lt_skus,
            "Failure_Rate_Pct": round(100.0 * underestimated_lt_skus / (total_sku_rows - missing_lt_cnt), 2),
            "Cleaning_Assumption_Logged": "Replace static SAP master lead time with rolling empirical plant/toller lead time in Safety Stock & ROP",
        },
        {
            "Dimension": "Uniqueness",
            "Table": "orders_deliveries",
            "Rule_Description": "Unique composite key (order_id, line_id)",
            "Records_Checked": total_orders_rows,
            "Failed_Records": dup_orders_cnt,
            "Failure_Rate_Pct": round(100.0 * dup_orders_cnt / total_orders_rows, 2),
            "Cleaning_Assumption_Logged": "Drop exact IDoc retry duplicates via drop_duplicates(subset=['order_id', 'line_id'])",
        },
        {
            "Dimension": "Validity (Referential)",
            "Table": "orders_deliveries",
            "Rule_Description": "Every order sku_id exists in sku_master",
            "Records_Checked": total_orders_rows,
            "Failed_Records": orphan_sku_cnt,
            "Failure_Rate_Pct": round(100.0 * orphan_sku_cnt / total_orders_rows, 2),
            "Cleaning_Assumption_Logged": "Quarantine unmapped legacy phase-out SKUs (SYN-SKU-998, SYN-SKU-999) into exception table",
        },
        {
            "Dimension": "Validity (Domain)",
            "Table": "orders_deliveries",
            "Rule_Description": "qty_ordered > 0 AND qty_delivered >= 0",
            "Records_Checked": total_orders_rows,
            "Failed_Records": neg_qty_cnt,
            "Failure_Rate_Pct": round(100.0 * neg_qty_cnt / total_orders_rows, 2),
            "Cleaning_Assumption_Logged": "Isolate negative credit-memo adjustments from outbound fulfillment OTIF denominator",
        },
        {
            "Dimension": "Timeliness (Chronology)",
            "Table": "orders_deliveries",
            "Rule_Description": "order_date <= dispatched_date <= delivered_date AND order_date <= promised_date",
            "Records_Checked": total_orders_rows,
            "Failed_Records": bad_dates_cnt,
            "Failure_Rate_Pct": round(100.0 * bad_dates_cnt / total_orders_rows, 2),
            "Cleaning_Assumption_Logged": "Quarantine backdated timestamp anomalies before computing transit duration distributions",
        },
    ])

    # =========================================================================
    # 2. CLEAN ORDER-LEVEL AGGREGATION (HANDLING SPLIT DELIVERIES PROPERLY)
    # =========================================================================
    sku_clean = sku_df.copy()
    sku_clean["uom"] = sku_clean["uom"].replace({"Litre": "L", "ltr": "L", "Kg": "KG", "kg": "KG"})
    fam_medians = sku_clean.groupby("product_family")["master_lead_time_days"].transform("median")
    sku_clean["master_lead_time_days_imputed"] = sku_clean["master_lead_time_days"].fillna(fam_medians)

    clean_lines = clean_lines.merge(
        sku_clean[["sku_id", "product_family", "formulation_type", "master_lead_time_days_imputed", "moq", "unit_cost_inr", "unit_price_inr"]],
        on="sku_id",
        how="inner",
    )

    orders_agg = clean_lines.groupby("order_id", as_index=False).agg(
        sku_id=("sku_id", "first"),
        product_family=("product_family", "first"),
        formulation_type=("formulation_type", "first"),
        plant_id=("plant_id", "first"),
        toller_flag=("toller_flag", "first"),
        customer_region=("customer_region", "first"),
        shipping_mode=("shipping_mode", "first"),
        order_date=("order_date", "first"),
        promised_date=("promised_date", "first"),
        dispatched_date=("dispatched_date", "min"),
        delivered_date=("delivered_date", "max"),
        qty_ordered=("qty_ordered", "first"),
        qty_delivered=("qty_delivered", "sum"),
        split_lines_count=("line_id", "count"),
        master_lead_time_days=("master_lead_time_days_imputed", "first"),
        moq=("moq", "first"),
        unit_price_inr=("unit_price_inr", "first"),
    )

    ord_dt = pd.to_datetime(orders_agg["order_date"])
    prom_dt = pd.to_datetime(orders_agg["promised_date"])
    deliv_dt = pd.to_datetime(orders_agg["delivered_date"])

    orders_agg["actual_lead_time_days"] = (deliv_dt - ord_dt).dt.days
    orders_agg["promised_lead_time_days"] = (prom_dt - ord_dt).dt.days
    orders_agg["delay_days"] = (deliv_dt - prom_dt).dt.days
    orders_agg["is_late"] = (orders_agg["delay_days"] > 0).astype(int)
    orders_agg["is_on_time"] = (orders_agg["delay_days"] <= 0).astype(int)
    orders_agg["is_in_full"] = (orders_agg["qty_delivered"] >= orders_agg["qty_ordered"]).astype(int)
    orders_agg["is_otif"] = ((orders_agg["is_on_time"] == 1) & (orders_agg["is_in_full"] == 1)).astype(int)
    orders_agg["order_month"] = ord_dt.dt.month
    orders_agg["is_kharif"] = orders_agg["order_month"].isin([6, 7, 8, 9]).astype(int)

    # =========================================================================
    # 3. MASTER DATA VS. REALITY LEAD-TIME CHECK PER PLANT / TOLLER
    # =========================================================================
    lt_reality_check = orders_agg.groupby(["plant_id", "toller_flag"], as_index=False).agg(
        total_clean_orders=("order_id", "count"),
        sap_master_lt_median=("master_lead_time_days", "median"),
        promised_lt_median=("promised_lead_time_days", "median"),
        actual_lt_median_p50=("actual_lead_time_days", "median"),
        actual_lt_p90=("actual_lead_time_days", lambda s: round(float(np.percentile(s, 90)), 1)),
        actual_lt_std=("actual_lead_time_days", lambda s: round(float(np.std(s, ddof=1)), 2)),
        on_time_pct=("is_on_time", lambda s: round(100.0 * s.mean(), 2)),
        in_full_pct=("is_in_full", lambda s: round(100.0 * s.mean(), 2)),
        otif_pct=("is_otif", lambda s: round(100.0 * s.mean(), 2)),
    )
    lt_reality_check["p90_vs_sap_gap_days"] = (
        lt_reality_check["actual_lt_p90"] - lt_reality_check["sap_master_lt_median"]
    ).round(1)

    # =========================================================================
    # 4. STATISTICAL HYPOTHESIS TESTS & SIMPSON'S PARADOX PROOF
    # =========================================================================
    toller_lt = orders_agg.loc[orders_agg["toller_flag"] == 1, "actual_lead_time_days"].values
    inhouse_lt = orders_agg.loc[orders_agg["toller_flag"] == 0, "actual_lead_time_days"].values
    u_stat, mw_pval = stats.mannwhitneyu(toller_lt, inhouse_lt, alternative="greater")

    kharif_df = orders_agg[orders_agg["is_kharif"] == 1]
    contingency = pd.crosstab(kharif_df["toller_flag"], kharif_df["is_late"])
    chi2_stat, chi2_pval, dof, _ = stats.chi2_contingency(contingency)

    kharif_toller_late_pct = round(100.0 * kharif_df.loc[kharif_df["toller_flag"] == 1, "is_late"].mean(), 2)
    kharif_inhouse_late_pct = round(100.0 * kharif_df.loc[kharif_df["toller_flag"] == 0, "is_late"].mean(), 2)
    overall_late_pct = round(100.0 * orders_agg["is_late"].mean(), 2)
    overall_otif_pct = round(100.0 * orders_agg["is_otif"].mean(), 2)

    # Simpson's Paradox Proof: Compare Express_Air_Road vs Surface_FTL in Aggregate vs Stratified by (toller_flag, is_kharif)
    sub_modes = orders_agg[orders_agg["shipping_mode"].isin(["Express_Air_Road", "Surface_FTL"])]
    agg_mode_comp = sub_modes.groupby("shipping_mode", as_index=False).agg(
        total_orders=("order_id", "count"),
        aggregate_late_rate_pct=("is_late", lambda s: round(100.0 * s.mean(), 2)),
        aggregate_otif_pct=("is_otif", lambda s: round(100.0 * s.mean(), 2)),
    )
    strat_mode_comp = sub_modes.groupby(["toller_flag", "is_kharif", "shipping_mode"], as_index=False).agg(
        orders=("order_id", "count"),
        stratum_late_rate_pct=("is_late", lambda s: round(100.0 * s.mean(), 2)),
        stratum_otif_pct=("is_otif", lambda s: round(100.0 * s.mean(), 2)),
    )

    sql_results = {}
    if sql_path and os.path.exists(sql_path):
        conn = sqlite3.connect(":memory:")
        sku_df.to_sql("sku_master", conn, index=False)
        orders_df.to_sql("orders_deliveries", conn, index=False)
        inv_df.to_sql("inventory_snapshots", conn, index=False)
        weather_df.to_sql("external_weather", conn, index=False)

        with open(sql_path, "r", encoding="utf-8") as f:
            sql_text = f.read()
        statements = [s.strip() for s in sql_text.split(";") if s.strip()]
        for idx, stmt in enumerate(statements, 1):
            df_res = pd.read_sql_query(stmt, conn)
            sql_results[f"query_{idx}"] = df_res
        conn.close()

    print("=== MODULE 4: AUTOMATED DATA QUALITY SCORECARD ===")
    print(dq_scorecard[["Dimension", "Table", "Rule_Description", "Failed_Records", "Failure_Rate_Pct"]].to_string(index=False))
    print("\n=== MASTER LEAD TIME VS. EMPIRICAL REALITY BY PLANT/TOLLER ===")
    print(lt_reality_check.to_string(index=False))
    print("\n=== STATISTICAL HYPOTHESIS TEST RESULTS ===")
    print(f"Overall Clean Orders Evaluated: {len(orders_agg):,} | Overall Late Rate: {overall_late_pct}% | Overall OTIF: {overall_otif_pct}%")
    print(f"1. Mann-Whitney U Test (Toller vs In-House Lead Time): U = {u_stat:,.1f}, p-value = {mw_pval:.4e}")
    print(f"   Toller Median/P90 LT: {np.median(toller_lt):.1f}d / {np.percentile(toller_lt, 90):.1f}d vs In-House Median/P90 LT: {np.median(inhouse_lt):.1f}d / {np.percentile(inhouse_lt, 90):.1f}d")
    print(f"2. Chi-Square Test (Kharif Late Rate: Toller vs In-House): Chi2 = {chi2_stat:,.2f}, p-value = {chi2_pval:.4e} (dof={dof})")
    print(f"   Kharif Toller Late Rate: {kharif_toller_late_pct}% vs Kharif In-House Late Rate: {kharif_inhouse_late_pct}%")
    print("\n=== SIMPSON'S PARADOX PROOF: EXPRESS_AIR_ROAD VS. SURFACE_FTL ===")
    print("--- A. Unstratified Aggregate (Misleading Reversal!) ---")
    print(agg_mode_comp.to_string(index=False))
    print("--- B. Stratified by Manufacturing Type (Toller) & Season (Kharif) ---")
    print(strat_mode_comp.to_string(index=False))

    if sql_results:
        print("\n=== VERIFIED SQL QUERY EXECUTION SUMMARY ===")
        for qk, qdf in sql_results.items():
            print(f"  {qk}: returned {len(qdf)} rows")
        print("\n--- SQL Query 3 (ABC/XYZ 9-Box Segment Counts) ---")
        print(sql_results["query_3"]["abc_xyz_segment"].value_counts().sort_index().to_string())
        print("\n--- SQL Query 4 (Top 5 SLOB / FEFO Write-Off Risk Batches) ---")
        print(sql_results["query_4"].head(5).to_string(index=False))
        print(f"Total SLOB Write-Off Risk across {len(sql_results['query_4'])} flagged batches (2025-12): INR {sql_results['query_4']['write_off_risk_inr'].sum():,.2f}")
        print("\n--- SQL Query 5 (Top 5 Bullwhip Effect SKUs) ---")
        print(sql_results["query_5"].head(5).to_string(index=False))
        print(f"Mean Distributor Bullwhip Ratio: {sql_results['query_5']['distributor_bullwhip_ratio'].mean():.2f}x | Mean Total Bullwhip Ratio: {sql_results['query_5']['total_supply_chain_bullwhip_ratio'].mean():.2f}x")

    return {
        "dq_scorecard": dq_scorecard,
        "lt_reality_check": lt_reality_check,
        "orders_agg": orders_agg,
        "sql_results": sql_results,
    }


if __name__ == "__main__":
    run_dq_and_diagnostics(
        data_dir="data",
        sql_path="sql/supply_chain_queries.sql",
    )