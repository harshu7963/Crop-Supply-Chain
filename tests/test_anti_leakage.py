"""
Unit Test 1: Strict Anti-Leakage Verification (tests/test_anti_leakage.py)
Asserts that zero post-order columns exist in the engineered feature dataset.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.late_delivery_model import FORBIDDEN_POST_ORDER_COLUMNS, build_leakage_free_order_dataset


def test_zero_post_order_leakage():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    df = build_leakage_free_order_dataset(data_dir=data_dir)
    leaked = [c for c in FORBIDDEN_POST_ORDER_COLUMNS if c in df.columns]
    assert len(leaked) == 0, f"Leakage columns found: {leaked}"
    assert "target_is_late" in df.columns
    assert len(df) > 45000


if __name__ == "__main__":
    test_zero_post_order_leakage()
    print("PASS: test_zero_post_order_leakage (0 post-order columns in feature dataset)")
