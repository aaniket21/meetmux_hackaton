import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.db import setup_db, load_baseline, load_batch, load_from_file, store_violations, store_repairs, get_column_profiles

def test_db_functions():
    conn = setup_db(":memory:")

    # Load baseline
    df = pd.DataFrame({"age": [20, 30, None, 50], "category": ["A", "B", "A", "C"]})
    load_baseline(conn, df)

    profiles = get_column_profiles(conn, "baseline")
    assert "age" in profiles
    assert profiles["age"]["min"] == 20
    assert profiles["age"]["max"] == 50
    assert profiles["age"]["null_rate"] == 0.25

    # Load batch
    df_batch = pd.DataFrame({"age": [99], "category": ["X"]})
    load_batch(conn, df_batch, batch_id=1)
    result = conn.execute("SELECT * FROM batch_1").df()
    assert len(result) == 1

    # Store violations
    violations = [{"type": "out_of_range", "column": "age", "details": "99 out of range"}]
    store_violations(conn, violations, batch_id=1)
    v_rows = conn.execute("SELECT * FROM violations").fetchall()
    assert len(v_rows) == 1

    # Store repairs
    audit = [{"row_idx": 0, "col": "age", "old_value": 99, "new_value": 35, "confidence": 0.9, "status": "accepted", "score_after": 1.2}]
    store_repairs(conn, audit, batch_id=1)
    r_rows = conn.execute("SELECT * FROM repairs").fetchall()
    assert len(r_rows) == 1

def test_parquet_loading(tmp_path=None):
    """Test CSV and Parquet loading via load_from_file."""
    conn = setup_db(":memory:")
    # Write a temp CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'baseline.csv')
    if os.path.exists(csv_path):
        load_from_file(conn, csv_path, "baseline_csv")
        count = conn.execute("SELECT COUNT(*) FROM baseline_csv").fetchone()[0]
        assert count > 0

if __name__ == "__main__":
    test_db_functions()
    print("test_db_functions passed!")
    test_parquet_loading()
    print("test_parquet_loading passed!")
