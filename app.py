import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Import core modules
from core.db import setup_db, load_baseline, get_column_profiles
from core.schema import infer_contract, validate_batch
from core.preprocess import Preprocessor
from core.drift import detect_drift

st.set_page_config(page_title="Self-Healing Data Engine", layout="wide")

st.title("Self-Healing Tabular Anomaly Detection Engine")

tab1, tab2, tab3, tab4 = st.tabs(["Baseline", "Monitor", "Heal", "Export"])

if "db_conn" not in st.session_state:
    st.session_state.db_conn = setup_db()

with tab1:
    st.header("1. Baseline Data & Training")
    uploaded_file = st.file_uploader("Upload clean baseline data (CSV)", type="csv", key="baseline_upload")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:")
        st.dataframe(df.head())
        
        if st.button("Train Model & Infer Schema"):
            with st.spinner("Processing..."):
                load_baseline(st.session_state.db_conn, df)
                st.session_state.contract = infer_contract(df)
                
                # Setup Preprocessor
                preprocessor = Preprocessor(st.session_state.contract)
                preprocessor.fit(df)
                st.session_state.preprocessor = preprocessor
                
                # Mocking model training for UI flow
                st.success("Training complete! Model and schema contract saved.")
                
                st.subheader("Inferred Schema Contract")
                st.json(st.session_state.contract)
                
with tab2:
    st.header("2. Monitor New Batch")
    batch_file = st.file_uploader("Upload new batch (CSV)", type="csv", key="batch_upload")
    
    if batch_file is not None and "contract" in st.session_state:
        df_batch = pd.read_csv(batch_file)
        st.write("Batch Preview:")
        st.dataframe(df_batch.head())
        
        if st.button("Run Diagnostics"):
            st.session_state.df_batch = df_batch
            
            # 1. Schema Validation
            st.subheader("Schema Violations")
            violations = validate_batch(df_batch, st.session_state.contract)
            if violations:
                st.error(f"Found {len(violations)} schema violations.")
                st.table(pd.DataFrame(violations))
            else:
                st.success("No schema violations.")
                
            # 2. Drift Detection
            st.subheader("Distribution Drift")
            # For drift we need the baseline data. Let's fetch it from duckdb
            baseline_df = st.session_state.db_conn.execute("SELECT * FROM baseline").df()
            drift_results = detect_drift(baseline_df, df_batch, st.session_state.contract)
            
            drift_data = [{"Column": col, "Test": res["test"], "p-value": res["p_value"], "Drifted": res["drifted"]} 
                          for col, res in drift_results.items()]
            if drift_data:
                st.table(pd.DataFrame(drift_data))
            else:
                st.info("No common columns to check drift.")
                
            # 3. Anomaly Scores (Mocked visualization)
            st.subheader("Anomaly Score Distribution")
            mock_scores = np.random.exponential(scale=1.0, size=len(df_batch))
            fig = px.histogram(mock_scores, nbins=50, title="Anomaly Scores")
            fig.add_vline(x=3.0, line_dash="dash", line_color="red", annotation_text="Threshold")
            st.plotly_chart(fig)
            
            st.session_state.diagnostics_run = True

with tab3:
    st.header("3. Self-Healing")
    if st.session_state.get("diagnostics_run", False):
        st.write("Identifying bad cells and imputing...")
        
        # Mocking the before/after for UI demonstration
        df_batch = st.session_state.df_batch
        
        # Fake a repair on the first row
        df_repaired = df_batch.copy()
        if len(df_repaired) > 0 and len(df_repaired.columns) > 0:
            col = df_repaired.columns[0]
            df_repaired.at[0, col] = "REPAIRED"
            
            st.subheader("Suggested Repairs")
            repair_data = {
                "Row": [0],
                "Column": [col],
                "Old Value": [df_batch.at[0, col]],
                "New Value": ["REPAIRED"],
                "Confidence": [0.95],
                "Status": ["Pending"]
            }
            repair_df = pd.DataFrame(repair_data)
            st.dataframe(repair_df)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Approve All"):
                    st.success("Repairs accepted and logged.")
                    st.session_state.df_cleaned = df_repaired
            with col2:
                if st.button("Reject All"):
                    st.warning("Repairs rejected.")
    else:
        st.info("Run diagnostics in the Monitor tab first.")

with tab4:
    st.header("4. Export")
    if "df_cleaned" in st.session_state:
        csv = st.session_state.df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Data (CSV)",
            data=csv,
            file_name='cleaned_batch.csv',
            mime='text/csv',
        )
        
        # Mock audit log
        audit_csv = "row,col,old_val,new_val,status\n0,col1,bad,good,accepted".encode('utf-8')
        st.download_button(
            label="Download Audit Log (CSV)",
            data=audit_csv,
            file_name='audit_log.csv',
            mime='text/csv',
        )
    else:
        st.info("No cleaned data available yet. Complete the Healing step first.")
