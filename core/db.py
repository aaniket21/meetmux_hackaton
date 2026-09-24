import duckdb
import pandas as pd

def setup_db(db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            batch_id INTEGER,
            violation_type VARCHAR,
            col_name VARCHAR,
            details VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS repairs (
            batch_id INTEGER,
            row_idx INTEGER,
            col_name VARCHAR,
            old_value VARCHAR,
            new_value VARCHAR,
            confidence FLOAT,
            status VARCHAR,
            score_after FLOAT
        )
    """)

    return conn

def load_baseline(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    conn.register("temp_df", df)
    conn.execute("CREATE OR REPLACE TABLE baseline AS SELECT * FROM temp_df")
    conn.unregister("temp_df")

def load_batch(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, batch_id: int = 1):
    conn.register("temp_batch", df)
    conn.execute(f"CREATE OR REPLACE TABLE batch_{batch_id} AS SELECT * FROM temp_batch")
    conn.unregister("temp_batch")

def load_from_file(conn: duckdb.DuckDBPyConnection, filepath: str, table_name: str = "baseline"):
    """Load CSV or Parquet file into a DuckDB table."""
    if filepath.endswith(".parquet"):
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{filepath}')")
    else:
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{filepath}')")

def store_violations(conn: duckdb.DuckDBPyConnection, violations: list[dict], batch_id: int = 1):
    for v in violations:
        conn.execute(
            "INSERT INTO violations VALUES (?, ?, ?, ?)",
            [batch_id, v.get("type", ""), v.get("column", ""), v.get("details", "")]
        )

def store_repairs(conn: duckdb.DuckDBPyConnection, audit_log: list[dict], batch_id: int = 1):
    for entry in audit_log:
        conn.execute(
            "INSERT INTO repairs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                batch_id,
                entry.get("row_idx", -1),
                entry.get("col", ""),
                str(entry.get("old_value", "")),
                str(entry.get("new_value", "")),
                entry.get("confidence", 0.0),
                entry.get("status", ""),
                entry.get("score_after", 0.0),
            ]
        )

def get_column_profiles(conn: duckdb.DuckDBPyConnection, table_name: str) -> dict:
    profiles = {}

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
