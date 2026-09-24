import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.db import setup_db, load_baseline, get_column_profiles

def test_db_functions():
    db_path = ":memory:" # use in-memory db for tests
    conn = setup_db(db_path)
    
    # Check tables are created
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    assert "baseline" in tables
    assert "batches" in tables
    assert "violations" in tables
    assert "repairs" in tables
    
    # Load dummy data
    df = pd.DataFrame({
        "age": [20, 30, None, 50],
        "category": ["A", "B", "A", "C"]
    })
    load_baseline(conn, df)
    
    # Check profiles
    profiles = get_column_profiles(conn, "baseline")
    assert "age" in profiles
    assert "category" in profiles
    
    # Verify age profile (min, max, null_rate)
    assert profiles["age"]["min"] == 20
    assert profiles["age"]["max"] == 50
    assert profiles["age"]["null_rate"] == 0.25 # 1 out of 4 is null
    
if __name__ == "__main__":
    test_db_functions()
    print("test_db_functions passed!")
