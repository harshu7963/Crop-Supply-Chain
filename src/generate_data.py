"""
Module 2: Realistic Agrochemical Synthetic Data Generator (src/generate_data.py)
Syngenta Crop Protection (CP P&S, India) - End-to-End Supply Chain Analytics Project
Candidate Portfolio: Harshita Kadam

Generates 4 realistic ERP/SAP & external datasets with intentional production data flaws:
1. sku_master.csv (50 SKUs across Herbicides, Insecticides, Fungicides, Biologicals)
2. orders_deliveries.csv (~52,700 order/delivery lines over 36 months across Pune, Goa, Panoli, Dahej)
3. inventory_snapshots.csv (1,800 monthly SKU inventory snapshots with batch shelf-life & multi-echelon demand)
4. external_weather.csv (180 regional monthly IMD rainfall & pest pressure records)
"""

import os
import numpy as np
import pandas as pd


def generate_syngenta_supply_chain_data(output_dir: str = "data", seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # 1. GENERATE SKU MASTER (50 Agrochemical SKUs + Intentional Flaws)
    # =========================================================================
    product_families = [
        ("Herbicides", ["EC", "SL", "OD"], 0.30, (450, 1450), (720, 2350), [540, 730, 1095]),
        ("Insecticides", ["EC", "SC", "WG"], 0.34, (580, 2200), (940, 3600), [540, 730, 730]),
        ("Fungicides", ["SC", "WG", "EC"], 0.24, (510, 1850), (830, 2950), [540, 730, 1095]),
        ("Biologicals", ["SC", "SL"], 0.12, (680, 1950), (1150, 3200), [270, 365, 450]),
    ]

    brand_roots = {
        "Herbicides": ["DualGold", "Rifit", "Axial", "Fusilade", "Gramoxone", "Callisto", "Topik"],
        "Insecticides": ["Actara", "Ampligo", "Virtako", "Karate", "Alika", "Pegasus", "Curacron", "Simodis"],
        "Fungicides": ["Amistar", "RidomilGold", "Kavach", "Score", "FolioGold", "Tilt", "Revus"],
        "Biologicals": ["Isabion", "Quantis", "Megafol", "Hicane", "VictratoBio"],
    }

    liquid_uoms_clean_and_dirty = ["L", "L", "L", "Litre", "ltr"]
    solid_uoms_clean_and_dirty = ["KG", "KG", "KG", "Kg", "kg"]
    xyz_regimes = ["X_Smooth", "Y_Seasonal", "Z_Erratic"]

    sku_rows = []
    sku_counter = 1
    for fam, forms, share, cost_range, price_range, shelf_lives in product_families:
        n_skus = int(round(50 * share))
        for i in range(n_skus):
            if sku_counter > 50:
                break
            sku_id = f"SYN-SKU-{sku_counter:03d}"
            form = forms[i % len(forms)]
            brand = brand_roots[fam][i % len(brand_roots[fam])]
            pack_size = rng.choice([0.25, 0.5, 1.0, 2.5, 5.0])
            uom = rng.choice(solid_uoms_clean_and_dirty) if form == "WG" else rng.choice(liquid_uoms_clean_and_dirty)

            master_lt = float(rng.integers(12, 19))
            moq = int(rng.choice([150, 250, 400, 600, 800]))
            shelf_life = int(rng.choice(shelf_lives))
            unit_cost = round(float(rng.uniform(*cost_range)), 2)
            margin_mult = float(rng.uniform(1.45, 1.75))
            unit_price = round(unit_cost * margin_mult, 2)

            if fam == "Biologicals":
                demand_profile = rng.choice(["Y_Seasonal", "Z_Erratic"], p=[0.4, 0.6])
            else:
                demand_profile = xyz_regimes[(sku_counter - 1) % 3]

            sku_rows.append({
                "sku_id": sku_id,
                "sku_name": f"{brand}_{form}_{pack_size}u",
                "product_family": fam,
                "formulation_type": form,
                "pack_size": pack_size,
                "uom": uom,
                "master_lead_time_days": master_lt,
                "moq": moq,
                "shelf_life_days": shelf_life,
                "unit_cost_inr": unit_cost,
                "unit_price_inr": unit_price,
                "ai_synthesis_lead_time_days": int(rng.integers(90, 121)),
                "demand_profile": demand_profile,
            })
            sku_counter += 1

    sku_df = pd.DataFrame(sku_rows)

    # Inject Flaw 1: Exactly 8% (4 of 50 SKUs) missing master_lead_time_days
    missing_lt_indices = rng.choice(sku_df.index, size=4, replace=False)
    sku_df.loc[missing_lt_indices, "master_lead_time_days"] = np.nan

    # =========================================================================
    # 2. GENERATE EXTERNAL WEATHER DATASET (36 Months x 5 Customer Regions)
    # =========================================================================
    regions = [
        "West - MH/GJ",
        "North - PB/HR/UP",
        "South - AP/TG/KA",
        "Central - MP/CG",
        "East - WB/BR",
    ]
    months = pd.date_range("2023-01-01", "2025-12-01", freq="MS")

    base_monthly_rain = {
        1: 15.0, 2: 18.0, 3: 22.0, 4: 30.0, 5: 55.0,
        6: 185.0, 7: 310.0, 8: 280.0, 9: 190.0, 10: 85.0,
        11: 28.0, 12: 14.0,
    }

    weather_rows = []
    for dt in months:
        m = dt.month
        ym = dt.strftime("%Y-%m")
        for reg in regions:
            lpa = base_monthly_rain[m] * (1.15 if "East" in reg or "West" in reg else 0.92)
            year_shock = -0.16 if dt.year == 2023 else (-0.08 if dt.year == 2024 else 0.22)
            anomaly_pct = round(float(np.clip(year_shock + rng.normal(0, 0.08), -0.45, 0.55) * 100), 2)
            actual_rain = round(max(0.0, lpa * (1.0 + anomaly_pct / 100.0)), 1)
            onset_dev = int(rng.integers(-6, 12)) if m in (6, 7, 8, 9) else 0
            base_pest = 72.0 if m in (7, 8, 9) else (58.0 if m in (10, 11, 12, 1, 2) else 32.0)
            pest_index = round(float(np.clip(base_pest + 0.45 * anomaly_pct + rng.normal(0, 4.0), 10.0, 99.0)), 1)

            weather_rows.append({
                "year_month": ym,
                "customer_region": reg,
                "imd_rainfall_mm": actual_rain,
                "long_period_avg_mm": round(lpa, 1),
                "rainfall_anomaly_pct": anomaly_pct,
                "monsoon_onset_deviation_days": onset_dev,
                "pest_pressure_index": pest_index,
            })
    weather_df = pd.DataFrame(weather_rows)
    weather_lookup = weather_df.set_index(["year_month", "customer_region"]).to_dict(orient="index")
    national_rain_by_ym = weather_df.groupby("year_month")["rainfall_anomaly_pct"].mean().to_dict()
    national_pest_by_ym = weather_df.groupby("year_month")["pest_pressure_index"].mean().to_dict()

    # =========================================================================
    # 3. GENERATE ORDERS & DELIVERIES (~50,000 Base Orders + Splits & Flaws)
    # =========================================================================
    plants = [
        ("PUNE-IH-01", 0, 0.32),
        ("GOA-IH-02", 0, 0.22),
        ("PANOLI-TL-01", 1, 0.25),
        ("DAHEJ-TL-02", 1, 0.21),
    ]
    plant_ids = [p[0] for p in plants]
    plant_toller_map = {p[0]: p[1] for p in plants}
    plant_probs = [p[2] for p in plants]

    month_weights = np.array([
        0.075, 0.068, 0.052, 0.045, 0.065,
        0.115, 0.145, 0.135, 0.095, 0.068,
        0.072, 0.065
    ])
    month_weights = month_weights / month_weights.sum()

    n_base_orders = 50000
    all_months_list = list(months)
    m_probs = []
    for dt in all_months_list:
        y_mult = 0.88 if dt.year == 2023 else (1.00 if dt.year == 2024 else 1.14)
        m_probs.append(month_weights[dt.month - 1] * y_mult)
    m_probs = np.array(m_probs) / np.sum(m_probs)

    sampled_month_idx = rng.choice(len(all_months_list), size=n_base_orders, p=m_probs)
    sampled_month_starts = pd.to_datetime([all_months_list[i] for i in sampled_month_idx])
    day_offsets = rng.integers(0, 28, size=n_base_orders)
    order_dates = np.sort(sampled_month_starts + pd.to_timedelta(day_offsets, unit="D"))

    sku_pareto_weights = np.power(np.arange(1, 51, dtype=float), -1.12)
    sku_pareto_weights /= sku_pareto_weights.sum()
    sampled_skus = rng.choice(sku_df["sku_id"].values, size=n_base_orders, p=sku_pareto_weights)
    sampled_plants = rng.choice(plant_ids, size=n_base_orders, p=plant_probs)
    sampled_regions = rng.choice(regions, size=n_base_orders, p=[0.26, 0.24, 0.20, 0.18, 0.12])

    sku_master_lt_map = sku_df.set_index("sku_id")["master_lead_time_days"].fillna(15.0).to_dict()
    sku_moq_map = sku_df.set_index("sku_id")["moq"].to_dict()
    sku_profile_map = sku_df.set_index("sku_id")["demand_profile"].to_dict()

    order_rows = []
    for idx in range(n_base_orders):
        ord_id = f"ORD-{202300000 + idx + 1}"
        sku_id = sampled_skus[idx]
        plant_id = sampled_plants[idx]
        toller_flag = plant_toller_map[plant_id]
        reg = sampled_regions[idx]
        ord_dt = pd.Timestamp(order_dates[idx])
        ym = ord_dt.strftime("%Y-%m")
        m = ord_dt.month
        is_kharif = 1 if m in (6, 7, 8, 9) else 0

        w_info = weather_lookup[(ym, reg)]
        rain_anom = w_info["rainfall_anomaly_pct"]
        profile = sku_profile_map[sku_id]

        # Simpson's Paradox structure across Shipping Modes
        if toller_flag == 1 and is_kharif == 1:
            ship_mode = rng.choice(
                ["Express_Air_Road", "Surface_FTL", "Surface_PTL", "Rail_Rake"],
                p=[0.42, 0.18, 0.25, 0.15],
            )
        elif toller_flag == 1 and is_kharif == 0:
            ship_mode = rng.choice(
                ["Express_Air_Road", "Surface_FTL", "Surface_PTL", "Rail_Rake"],
                p=[0.12, 0.50, 0.24, 0.14],
            )
        elif toller_flag == 0 and is_kharif == 1:
            ship_mode = rng.choice(
                ["Express_Air_Road", "Surface_FTL", "Surface_PTL", "Rail_Rake"],
                p=[0.10, 0.62, 0.18, 0.10],
            )
        else:
            ship_mode = rng.choice(
                ["Express_Air_Road", "Surface_FTL", "Surface_PTL", "Rail_Rake"],
                p=[0.04, 0.74, 0.14, 0.08],
            )

        base_moq = sku_moq_map[sku_id]
        if profile == "X_Smooth":
            season_mult = 1.18 if is_kharif else (1.05 if m in (11, 12, 1, 2) else 0.88)
            qty_ordered = int(max(40, round(base_moq * season_mult * rng.lognormal(0.0, 0.18))))
        elif profile == "Y_Seasonal":
            season_mult = 1.75 if is_kharif else (1.22 if m in (11, 12, 1, 2) else 0.58)
            weather_mult = max(0.75, 1.0 + 0.005 * rain_anom)
            qty_ordered = int(max(30, round(base_moq * season_mult * weather_mult * rng.lognormal(0.0, 0.30))))
        else:
            season_mult = 2.65 if is_kharif else (1.15 if m in (11, 12) else 0.25)
            weather_mult = max(0.50, 1.0 + 0.009 * rain_anom)
            qty_ordered = int(max(20, round(base_moq * season_mult * weather_mult * rng.lognormal(0.0, 0.55))))

        sap_lt = sku_master_lt_map[sku_id]
        promised_days = int(max(9, round(sap_lt + rng.integers(-2, 4))))
        promised_dt = ord_dt + pd.Timedelta(days=promised_days)

        qty_ratio = qty_ordered / float(base_moq)
        mode_shift = {
            "Express_Air_Road": -1.8,
            "Surface_FTL": 0.0,
            "Surface_PTL": 2.4,
            "Rail_Rake": 3.6,
        }[ship_mode]

        if toller_flag == 1 and is_kharif == 1:
            stratum_shift = 2.8
        elif toller_flag == 1 and is_kharif == 0:
            stratum_shift = -0.8
        elif toller_flag == 0 and is_kharif == 1:
            stratum_shift = -1.6
        else:
            stratum_shift = -4.2

        rain_shift = 1.9 if (rain_anom > 12.0 and is_kharif) else (0.7 if rain_anom > 8.0 else -0.4)
        qty_shift = 2.2 if qty_ratio > 2.0 else (0.9 if qty_ratio > 1.35 else -0.7)
        tight_promise_shift = 2.1 if promised_days <= 13 else (-1.2 if promised_days >= 17 else 0.0)
        plant_specific_shift = 0.9 if plant_id == "DAHEJ-TL-02" else (0.4 if plant_id == "PANOLI-TL-01" else -0.3)

        expected_delay = stratum_shift + mode_shift + rain_shift + qty_shift + tight_promise_shift + plant_specific_shift
        actual_delay_days = int(round(expected_delay + rng.normal(0, 2.3)))
        actual_lt_days = max(3, promised_days + actual_delay_days)
        dispatch_lag = int(max(1, round(actual_lt_days * rng.uniform(0.55, 0.75))))

        dispatched_dt = ord_dt + pd.Timedelta(days=dispatch_lag)
        delivered_dt = ord_dt + pd.Timedelta(days=actual_lt_days)

        shortage_prob = 0.12 if (toller_flag == 1 and is_kharif) else (0.05 if toller_flag == 1 else 0.025)
        if rng.random() < shortage_prob:
            fill_ratio = float(rng.uniform(0.74, 0.94))
            qty_delivered = int(round(qty_ordered * fill_ratio))
        else:
            qty_delivered = qty_ordered

        if rng.random() < 0.040 and qty_delivered >= 80:
            qty_deliv_1 = int(round(qty_delivered * 0.60))
            qty_deliv_2 = qty_delivered - qty_deliv_1
            order_rows.append({
                "order_id": ord_id,
                "line_id": 1,
                "sku_id": sku_id,
                "plant_id": plant_id,
                "toller_flag": toller_flag,
                "customer_region": reg,
                "shipping_mode": ship_mode,
                "order_date": ord_dt.strftime("%Y-%m-%d"),
                "promised_date": promised_dt.strftime("%Y-%m-%d"),
                "dispatched_date": dispatched_dt.strftime("%Y-%m-%d"),
                "delivered_date": delivered_dt.strftime("%Y-%m-%d"),
                "qty_ordered": qty_ordered,
                "qty_delivered": qty_deliv_1,
            })
            delivered_dt_2 = delivered_dt + pd.Timedelta(days=int(rng.integers(1, 4)))
            order_rows.append({
                "order_id": ord_id,
                "line_id": 2,
                "sku_id": sku_id,
                "plant_id": plant_id,
                "toller_flag": toller_flag,
                "customer_region": reg,
                "shipping_mode": ship_mode,
                "order_date": ord_dt.strftime("%Y-%m-%d"),
                "promised_date": promised_dt.strftime("%Y-%m-%d"),
                "dispatched_date": (dispatched_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "delivered_date": delivered_dt_2.strftime("%Y-%m-%d"),
                "qty_ordered": qty_ordered,
                "qty_delivered": qty_deliv_2,
            })
        else:
            order_rows.append({
                "order_id": ord_id,
                "line_id": 1,
                "sku_id": sku_id,
                "plant_id": plant_id,
                "toller_flag": toller_flag,
                "customer_region": reg,
                "shipping_mode": ship_mode,
                "order_date": ord_dt.strftime("%Y-%m-%d"),
                "promised_date": promised_dt.strftime("%Y-%m-%d"),
                "dispatched_date": dispatched_dt.strftime("%Y-%m-%d"),
                "delivered_date": delivered_dt.strftime("%Y-%m-%d"),
                "qty_ordered": qty_ordered,
                "qty_delivered": qty_delivered,
            })

    orders_df = pd.DataFrame(order_rows)

    # Inject Flaws 2, 3, 4, 5
    n_bad_dates = 280
    bad_date_idx = rng.choice(orders_df.index, size=n_bad_dates, replace=False)
    half = n_bad_dates // 2
    orders_df.loc[bad_date_idx[:half], "delivered_date"] = (
        pd.to_datetime(orders_df.loc[bad_date_idx[:half], "order_date"]) - pd.Timedelta(days=5)
    ).dt.strftime("%Y-%m-%d")
    orders_df.loc[bad_date_idx[half:], "promised_date"] = (
        pd.to_datetime(orders_df.loc[bad_date_idx[half:], "order_date"]) - pd.Timedelta(days=4)
    ).dt.strftime("%Y-%m-%d")

    neg_qty_idx = rng.choice(orders_df.index.difference(bad_date_idx), size=150, replace=False)
    orders_df.loc[neg_qty_idx, "qty_ordered"] = -orders_df.loc[neg_qty_idx, "qty_ordered"].abs()
    orders_df.loc[neg_qty_idx, "qty_delivered"] = -orders_df.loc[neg_qty_idx, "qty_delivered"].abs()

    orphan_idx = rng.choice(
        orders_df.index.difference(bad_date_idx).difference(neg_qty_idx), size=180, replace=False
    )
    orders_df.loc[orphan_idx, "sku_id"] = rng.choice(["SYN-SKU-998", "SYN-SKU-999"], size=180)

    dup_sample_idx = rng.choice(orders_df[orders_df["line_id"] == 1].index, size=750, replace=False)
    dup_rows = orders_df.loc[dup_sample_idx].copy()
    orders_df = pd.concat([orders_df, dup_rows], ignore_index=True)

    # =========================================================================
    # 4. GENERATE MONTHLY INVENTORY SNAPSHOTS (36 Months x 50 SKUs = 1,800 Rows)
    # =========================================================================
    sku_shelf_map = sku_df.set_index("sku_id")["shelf_life_days"].to_dict()

    # Deterministic seasonal curves by regime so X_Smooth (CV~0.28), Y_Seasonal (CV~0.62), Z_Erratic (CV~0.98)
    seasonal_idx_x = {1: 1.05, 2: 0.98, 3: 0.72, 4: 0.68, 5: 0.85, 6: 1.25, 7: 1.42, 8: 1.38, 9: 1.15, 10: 0.85, 11: 0.88, 12: 0.79}
    seasonal_idx_y = {1: 1.10, 2: 0.92, 3: 0.42, 4: 0.35, 5: 0.65, 6: 1.65, 7: 2.15, 8: 1.98, 9: 1.35, 10: 0.48, 11: 0.52, 12: 0.43}
    seasonal_idx_z = {1: 0.95, 2: 0.65, 3: 0.00, 4: 0.00, 5: 0.30, 6: 2.10, 7: 3.10, 8: 2.75, 9: 1.40, 10: 0.25, 11: 0.30, 12: 0.20}

    ym_list = [dt.strftime("%Y-%m") for dt in months]

    inv_rows = []
    for t_i, dt in enumerate(months):
        ym = dt.strftime("%Y-%m")
        m = dt.month
        yr = dt.year
        # Pre-season (t-3) soil-moisture rainfall anomaly and pest build-up drive channel stocking at t!
        lag3_ym = ym_list[max(0, t_i - 3)]
        rain_lag3 = national_rain_by_ym[lag3_ym]
        pest_lag3 = national_pest_by_ym[lag3_ym]

        # Structural YoY trend + 3-month lead weather/pest stocking effect
        yr_trend = 0.88 if yr == 2023 else (1.00 if yr == 2024 else 1.15)
        weather_stocking_boost = 1.0 + 0.0065 * rain_lag3 + 0.0025 * (pest_lag3 - 50.0)

        for s_idx, srow in sku_df.iterrows():
            sku_id = srow["sku_id"]
            shelf_life = sku_shelf_map[sku_id]
            profile = sku_profile_map[sku_id]
            # Pareto base monthly demand scale
            base_scale = 28000.0 * np.power(s_idx + 1, -0.92)

            if profile == "X_Smooth":
                s_mult = seasonal_idx_x[m]
                noise = rng.normal(1.0, 0.045)
                dist_demand = int(max(80, round(base_scale * s_mult * yr_trend * weather_stocking_boost * noise)))
            elif profile == "Y_Seasonal":
                s_mult = seasonal_idx_y[m]
                noise = rng.normal(1.0, 0.065)
                dist_demand = int(max(40, round(base_scale * s_mult * yr_trend * weather_stocking_boost * noise)))
            else:
                s_mult = seasonal_idx_z[m]
                if s_mult == 0.0 or (m in (10, 11, 12, 2, 5) and rng.random() < 0.25):
                    dist_demand = 0
                else:
                    noise = rng.normal(1.0, 0.09)
                    dist_demand = int(max(0, round(base_scale * s_mult * yr_trend * weather_stocking_boost * noise)))

            # Multi-echelon Bullwhip progression
            retail_sales = int(max(50, round(0.60 * base_scale + 0.40 * dist_demand + rng.normal(0, 0.03 * base_scale))))
            batch_mult = rng.choice([0.0, 0.95, 2.05], p=[0.30, 0.40, 0.30])
            plant_prod_qty = int(max(0, round(dist_demand * batch_mult + rng.normal(0, 0.10 * base_scale))))

            daily_rate = max(4.0, base_scale / 30.0)
            if srow["product_family"] == "Biologicals" or (profile == "Z_Erratic" and int(sku_id[-3:]) % 4 == 0):
                on_hand = int(round(daily_rate * rng.uniform(115, 170)))
                batch_age = int(rng.integers(int(shelf_life * 0.72), int(shelf_life * 0.90)))
            else:
                on_hand = int(round(daily_rate * rng.uniform(32, 68)))
                batch_age = int(rng.integers(30, int(shelf_life * 0.42)))

            rem_shelf_life = max(10, shelf_life - batch_age)
            primary_plant = rng.choice(plant_ids, p=plant_probs)

            inv_rows.append({
                "snapshot_month": ym,
                "plant_id": primary_plant,
                "sku_id": sku_id,
                "batch_id": f"B-{ym.replace('-', '')}-{sku_id[-3:]}",
                "on_hand_qty": on_hand,
                "batch_age_days": batch_age,
                "remaining_shelf_life_days": rem_shelf_life,
                "retail_secondary_sales_qty": retail_sales,
                "distributor_order_qty": dist_demand,
                "plant_production_order_qty": plant_prod_qty,
            })

    inv_df = pd.DataFrame(inv_rows)

    sku_df.to_csv(os.path.join(output_dir, "sku_master.csv"), index=False)
    orders_df.to_csv(os.path.join(output_dir, "orders_deliveries.csv"), index=False)
    inv_df.to_csv(os.path.join(output_dir, "inventory_snapshots.csv"), index=False)
    weather_df.to_csv(os.path.join(output_dir, "external_weather.csv"), index=False)

    summary = {
        "sku_master_rows": len(sku_df),
        "orders_deliveries_rows": len(orders_df),
        "inventory_snapshots_rows": len(inv_df),
        "external_weather_rows": len(weather_df),
        "missing_lead_times_count": int(sku_df["master_lead_time_days"].isna().sum()),
        "duplicate_order_lines": int(orders_df.duplicated(subset=["order_id", "line_id"]).sum()),
    }
    print("=== SYNGENTA CP P&S SYNTHETIC DATA GENERATION COMPLETE ===")
    for k, v in summary.items():
        print(f"  {k}: {v:,}")
    return summary


if __name__ == "__main__":
    generate_syngenta_supply_chain_data(output_dir="data", seed=42)