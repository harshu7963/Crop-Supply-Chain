# Responsible AI Model Card & Governance Pack

## 1. Model Inventory & Intended Use
- **Model A (`LateDeliveryRiskClassifier`):** Class-weighted `HistGradientBoosting` / `LightGBM` model predicting `target_is_late` at the moment of order placement (`order_date`).
  - *Out-of-Time Performance (5-Fold `TimeSeriesSplit`):* **PR-AUC = 0.9040**, **ROC-AUC = 0.9182**.
  - *Planner-Tuned Operating Point ($p \ge 0.235$):* **Precision = 65.05%**, **Recall = 95.47%**, **F1 = 0.7738**.
- **Model B (`ScaleNormalizedGlobalLightGBM`):** Cross-SKU global gradient boosted regressor predicting monthly SKU demand at a 3-month planning horizon ($t+3$).
  - *Rolling-Origin Performance:* **WAPE = 9.37%** (vs. **30.03%** Seasonal-Naive, **+20.66 pts FVA**), **Bias = -2.88%**, **MASE = 0.834**.

## 2. Out-of-Scope Scenarios
- **New Product Introductions (NPIs):** SKUs with $< 15$ months of history must use hierarchical product-family analog priors rather than direct lag-12 tree splits.
- **Automated Vendor Penalties:** Model probabilities must never trigger automatic financial chargebacks against tollers (`Panoli`, `Dahej`) without human-in-the-loop verification.

## 3. Segment Fairness & Simpson's Paradox Guardrail
- Unstratified aggregate metrics penalize `Express_Air_Road` (**48.96% late** vs **25.34%** for `Surface_FTL`) because 59.2% of Express orders are emergency Kharif Toller dispatches. Vendor SLA scorecards must stratify performance by `(toller_flag, is_kharif, qty_vs_moq_ratio)`.

## 4. Drift Monitoring & Retraining Triggers
- Monitor weekly **Population Stability Index (PSI)** on `rainfall_anomaly_pct`, `qty_vs_moq_ratio`, and `days_to_promised` (Alert: $	ext{PSI} > 0.15$; Retrain: $	ext{PSI} > 0.25$).
- Retrain if rolling 90-day classifier Precision falls below **60.0%** or $t+3$ forecast WAPE exceeds **15.0%**.

## 5. Data Privacy & Regulatory Compliance (India DPDP Act 2023)
- Operates strictly on B2B supply chain entities (`plant_id`, `sku_id`, `order_id`, `batch_id`, `customer_region`). Contains zero farmer or retail dealer PII.
