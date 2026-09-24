import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go

from core.db import setup_db, load_baseline, load_batch, store_violations, store_repairs, get_column_profiles
from core.schema import infer_contract, validate_batch
from core.preprocess import Preprocessor
from core.scarf import SCARF, SCARFDataset, train_scarf
from core.scoring import KNNAnomalyScorer, IFBaselineScorer
from core.drift import detect_drift, compute_severity
from core.heal import attribute_bad_cells, impute_cells
from core.persistence import save_pipeline, load_pipeline

st.set_page_config(page_title="Self-Healing Data Engine", layout="wide")
st.title("Self-Healing Tabular Anomaly Detection Engine")

# Session state init
if "db_conn" not in st.session_state:
    st.session_state.db_conn = setup_db()

tab1, tab2, tab3, tab4 = st.tabs(["Baseline", "Monitor", "Heal", "Export"])

# ── TAB 1: Baseline ──────────────────────────────────────────────────
with tab1:
    st.header("1. Baseline Data & Training")
    uploaded_file = st.file_uploader("Upload clean baseline data (CSV)", type=["csv", "parquet"], key="baseline_upload")

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.write(f"**Preview** ({len(df)} rows, {len(df.columns)} columns):")
        st.dataframe(df.head(10))

        if st.button("Train Model & Infer Schema"):
            with st.spinner("Training SCARF model — this may take a minute..."):
                # 1. Store baseline in DuckDB
                load_baseline(st.session_state.db_conn, df)

                # 2. Infer schema contract
                contract = infer_contract(df)
                st.session_state.contract = contract

                # 3. Fit preprocessor
                preprocessor = Preprocessor(contract)
                preprocessor.fit(df)
                st.session_state.preprocessor = preprocessor

                # 4. Transform baseline
                X_baseline = preprocessor.transform(df)
                input_dim = X_baseline.shape[1]

                # 5. Train SCARF
                dataset = SCARFDataset(X_baseline.astype(np.float32), corruption_rate=0.4)
                model = SCARF(input_dim=input_dim, emb_dim=128, head_dim=64)
                avg_loss = train_scarf(model, dataset, epochs=15, batch_size=128, lr=1e-3, seed=42)
                st.session_state.model = model

                # 6. Compute baseline embeddings
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X_baseline.astype(np.float32))
                    baseline_emb, _ = model(X_tensor)
                    baseline_emb = baseline_emb.numpy()
                st.session_state.baseline_emb = baseline_emb

                # 7. Fit kNN scorer
                scorer = KNNAnomalyScorer(k=5)
                scorer.fit(baseline_emb)
                st.session_state.scorer = scorer

                # 8. Fit Isolation Forest
                if_scorer = IFBaselineScorer()
                if_scorer.fit(X_baseline)
                st.session_state.if_scorer = if_scorer

                # 9. Compute baseline stats for attribution
                baseline_stats = {}
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        baseline_stats[col] = float(df[col].median())
                    else:
                        baseline_stats[col] = df[col].mode()[0]
                st.session_state.baseline_stats = baseline_stats

                st.success(f"Training complete! Avg contrastive loss: {avg_loss:.4f}")

            st.subheader("Inferred Schema Contract")
            st.json(contract)

            profiles = get_column_profiles(st.session_state.db_conn, "baseline")
            st.subheader("Column Profiles")
            st.table(pd.DataFrame(profiles).T)

# ── TAB 2: Monitor ───────────────────────────────────────────────────
with tab2:
    st.header("2. Monitor New Batch")
    batch_file = st.file_uploader("Upload new batch (CSV or Parquet)", type=["csv", "parquet"], key="batch_upload")

    if batch_file is not None and "contract" in st.session_state:
        if batch_file.name.endswith(".parquet"):
            df_batch = pd.read_parquet(batch_file)
        else:
            df_batch = pd.read_csv(batch_file)
        st.write(f"**Batch Preview** ({len(df_batch)} rows):")
        st.dataframe(df_batch.head(10))

        if st.button("Run Diagnostics"):
            st.session_state.df_batch = df_batch
            load_batch(st.session_state.db_conn, df_batch, batch_id=1)

            # Schema Validation
            st.subheader("Schema Violations")
            violations = validate_batch(df_batch, st.session_state.contract)
            store_violations(st.session_state.db_conn, violations, batch_id=1)
            if violations:
                st.error(f"Found {len(violations)} schema violations.")
                st.table(pd.DataFrame(violations))
            else:
                st.success("No schema violations.")

            # Drift Detection
            st.subheader("Distribution Drift")
            baseline_df = st.session_state.db_conn.execute("SELECT * FROM baseline").df()
            drift_results = detect_drift(baseline_df, df_batch, st.session_state.contract)
            severity = compute_severity(drift_results)
            st.session_state.drift_results = drift_results

            drift_data = [{"Column": col, "Test": res["test"], "p-value": f"{res['p_value']:.4f}",
                           "PSI": f"{res.get('psi', 'N/A')}", "Drifted": res["drifted"]}
                          for col, res in drift_results.items()]
            if drift_data:
                severity_color = {"green": "🟢", "amber": "🟡", "red": "🔴"}
                st.write(f"**Overall Drift Severity:** {severity_color.get(severity, '')} {severity.upper()}")
                st.table(pd.DataFrame(drift_data))
            else:
                st.info("No common columns to check drift.")

            # Real Anomaly Scoring
            st.subheader("Anomaly Scoring")
            preprocessor = st.session_state.preprocessor
            model = st.session_state.model
            scorer = st.session_state.scorer

            # Only score columns that exist in contract
            batch_cols = set(df_batch.columns)
            contract_cols = set(st.session_state.contract["columns"].keys())
            common_cols = batch_cols.intersection(contract_cols)

            if len(common_cols) == len(contract_cols):
                X_batch = preprocessor.transform(df_batch)
                model.eval()
                with torch.no_grad():
                    batch_emb, _ = model(torch.tensor(X_batch.astype(np.float32)))
                    batch_emb = batch_emb.numpy()
                st.session_state.batch_emb = batch_emb

                scores, is_anomaly = scorer.score(batch_emb)
                st.session_state.scores = scores
                st.session_state.is_anomaly = is_anomaly

                # IF comparison
                if_scores, if_anomaly = st.session_state.if_scorer.score(X_batch)

                n_flagged = int(is_anomaly.sum())
                n_if_flagged = int(if_anomaly.sum())
                st.write(f"**SCARF+kNN flagged:** {n_flagged} / {len(df_batch)} rows")
                st.write(f"**Isolation Forest flagged:** {n_if_flagged} / {len(df_batch)} rows")

                fig = px.histogram(scores, nbins=50, title="SCARF Anomaly Scores")
                fig.add_vline(x=scorer.threshold, line_dash="dash", line_color="red",
                              annotation_text=f"Threshold={scorer.threshold:.3f}")
                st.plotly_chart(fig)

                st.session_state.diagnostics_run = True
            else:
                missing = contract_cols - common_cols
                st.warning(f"Cannot score: batch is missing columns {missing}. Schema violations detected above.")
                st.session_state.diagnostics_run = False

    elif batch_file is not None:
        st.warning("Please train a baseline model first (Baseline tab).")

# ── TAB 3: Heal ──────────────────────────────────────────────────────
with tab3:
    st.header("3. Self-Healing")
    if st.session_state.get("diagnostics_run", False) and st.session_state.get("is_anomaly") is not None:
        is_anomaly = st.session_state.is_anomaly
        df_batch = st.session_state.df_batch
        scores = st.session_state.scores

        flagged_indices = np.where(is_anomaly)[0]
        st.write(f"**{len(flagged_indices)} anomalous rows** detected. Running attribution & imputation...")

        if len(flagged_indices) > 0 and st.button("Run Self-Healing"):
            preprocessor = st.session_state.preprocessor
            model = st.session_state.model
            scorer = st.session_state.scorer
            baseline_stats = st.session_state.baseline_stats
            baseline_emb = st.session_state.baseline_emb

            # Score function for attribution
            def score_fn(df_sub):
                X = preprocessor.transform(df_sub)
                model.eval()
                with torch.no_grad():
                    emb, _ = model(torch.tensor(X.astype(np.float32)))
                s, _ = scorer.score(emb.numpy())
                return s

            # kNN impute function
            from sklearn.neighbors import NearestNeighbors
            baseline_df = st.session_state.db_conn.execute("SELECT * FROM baseline").df()
            nn = NearestNeighbors(n_neighbors=5)
            nn.fit(baseline_emb)

            def knn_impute_fn(row, col):
                row_df = pd.DataFrame([row])
                try:
                    X = preprocessor.transform(row_df)
                    model.eval()
                    with torch.no_grad():
                        emb, _ = model(torch.tensor(X.astype(np.float32)))
                    _, indices = nn.kneighbors(emb.numpy(), n_neighbors=5)
                    neighbors = baseline_df.iloc[indices[0]]
                    if pd.api.types.is_numeric_dtype(baseline_df[col]):
                        return float(neighbors[col].mean())
                    else:
                        return neighbors[col].mode()[0]
                except Exception:
                    return baseline_stats.get(col, row[col])

            df_flagged = df_batch.iloc[flagged_indices].copy()

            with st.spinner("Running attribution and imputation..."):
                bad_cells = attribute_bad_cells(df_flagged, score_fn, baseline_stats)
                df_repaired, audit_log = impute_cells(df_flagged, bad_cells, knn_impute_fn, score_fn, scorer.threshold)

            # Store in DuckDB
            store_repairs(st.session_state.db_conn, audit_log, batch_id=1)
            st.session_state.audit_log = audit_log

            # Merge repaired rows back
            df_cleaned = df_batch.copy()
            for i, orig_idx in enumerate(flagged_indices):
                for cell in [c for c in bad_cells if c["row_idx"] == df_flagged.index[i]]:
                    col = cell["col"]
                    df_cleaned.iloc[orig_idx, df_cleaned.columns.get_loc(col)] = df_repaired.at[df_flagged.index[i], col]
            st.session_state.df_cleaned = df_cleaned

            # Display
            st.subheader("Suggested Repairs")
            if audit_log:
                audit_df = pd.DataFrame(audit_log)
                st.dataframe(audit_df)

                accepted = sum(1 for a in audit_log if a["status"] == "accepted")
                review = sum(1 for a in audit_log if a["status"] == "human_review")
                st.write(f"✅ **Accepted:** {accepted} | 🔍 **Needs Review:** {review}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Approve All"):
                    st.success("All repairs accepted and logged to DuckDB.")
            with col2:
                if st.button("Reject All"):
                    st.session_state.df_cleaned = df_batch
                    st.warning("Repairs rejected. Original data preserved.")
        elif len(flagged_indices) == 0:
            st.success("No anomalous rows to heal!")
    else:
        st.info("Run diagnostics in the Monitor tab first.")

# ── TAB 4: Export ─────────────────────────────────────────────────────
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

    if "audit_log" in st.session_state and st.session_state.audit_log:
        audit_df = pd.DataFrame(st.session_state.audit_log)
        audit_csv = audit_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Audit Log (CSV)",
            data=audit_csv,
            file_name='audit_log.csv',
            mime='text/csv',
        )

    # Also export from DuckDB
    try:
        repairs_df = st.session_state.db_conn.execute("SELECT * FROM repairs").df()
        if len(repairs_df) > 0:
            st.subheader("Audit Log (from DuckDB)")
            st.dataframe(repairs_df)
    except Exception:
        pass

    if "df_cleaned" not in st.session_state:
        st.info("No cleaned data available yet. Complete the Healing step first.")
