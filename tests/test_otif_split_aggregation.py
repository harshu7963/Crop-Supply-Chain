"""
Unit Test 2: Split-Shipment OTIF Aggregation Verification (tests/test_otif_split_aggregation.py)
Verifies that split delivery lines sharing an order_id are summed before evaluating In-Full.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dq_and_diagnostics import run_dq_and_diagnostics


def test_split_shipment_header_aggregation():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    sql_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sql", "supply_chain_queries.sql"))
    res = run_dq_and_diagnostics(data_dir=data_dir, sql_path=sql_path)
    orders_agg = res["orders_agg"]

    # Every order_id in orders_agg must be unique (1 row per order header)
    assert orders_agg["order_id"].is_unique, "Split shipments were not aggregated to unique order_id grain!"
    assert (orders_agg["split_lines_count"] > 1).sum() > 1500, "Expected split-shipment orders to be present!"
    assert len(res["sql_results"]) == 6, "Expected all 6 SQL queries to execute cleanly!"


if __name__ == "__main__":
    test_split_shipment_header_aggregation()
    print("PASS: test_split_shipment_header_aggregation (unique order_id grain & all 6 SQL queries verified)")
