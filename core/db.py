import duckdb
import pandas as pd

def setup_db(db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(db_path)
    # Create empty tables for structure (types will be dynamic based on data for baseline/batches)
    # For now, just create them to satisfy the test, we'll recreate them dynamically during ingestion
    conn.execute("CREATE TABLE IF NOT EXISTS baseline (id INTEGER)")
    conn.execute("CREATE TABLE IF NOT EXISTS batches (id INTEGER)")
    conn.execute("CREATE TABLE IF NOT EXISTS violations (id INTEGER, issue VARCHAR)")
    conn.execute("CREATE TABLE IF NOT EXISTS repairs (id INTEGER, old_val VARCHAR, new_val VARCHAR, conf FLOAT, status VARCHAR)")
    return conn

def load_baseline(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    # Register the dataframe as a view, then create/replace table
    conn.register("temp_df", df)
    conn.execute("CREATE OR REPLACE TABLE baseline AS SELECT * FROM temp_df")
    conn.unregister("temp_df")

def get_column_profiles(conn: duckdb.DuckDBPyConnection, table_name: str) -> dict:
    profiles = {}
    
    # Get columns
    columns = [c[1] for c in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
    total_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    
    if total_rows == 0:
        return profiles
        
    for col in columns:
        res = conn.execute(f"""
            SELECT 
                MIN("{col}") as min_val,
                MAX("{col}") as max_val,
                COUNT(*) FILTER (WHERE "{col}" IS NULL) as null_count,
                COUNT(DISTINCT "{col}") as distinct_count
            FROM {table_name}
        """).fetchone()
        
        profiles[col] = {
            "min": res[0],
            "max": res[1],
            "null_rate": res[2] / total_rows,
            "distinct_count": res[3]
        }
        
    return profiles
