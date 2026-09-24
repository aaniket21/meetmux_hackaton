# Self-Healing Tabular Anomaly Detection Engine

A tool that learns what "normal" data looks like from a clean table, checks new incoming data against it, and automatically fixes the bad values it finds. Built with PyTorch, Streamlit, and DuckDB.

## Features
1. **Detect**: Schema violations, distribution drift, and anomalous rows via self-supervised contrastive learning (SCARF).
2. **Locate**: Identify exact cells that are wrong using score-drop attribution.
3. **Repair**: Impute cells using kNN in embedding space and verify repairs.
4. **Audit**: Maintain an audit log of all changes.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run tests**:
   ```bash
   python -m pytest tests/
   # Or run individually: python tests/test_schema.py
   ```

## Running the Application

Start the Streamlit dashboard:
```bash
python -m streamlit run app.py
```
*(Note: use `python -m streamlit run app.py` instead of `streamlit run app.py` if `streamlit` is not directly in your system PATH).*

## Demo Workflow

1. Open the **Baseline** tab. Upload a clean CSV dataset (e.g., adult income). Click "Train Model & Infer Schema".
2. Open the **Monitor** tab. Upload a new batch (you can generate one using `scripts/inject_faults.py`). Run diagnostics to see schema violations, drift heatmaps, and anomaly scores.
3. Open the **Heal** tab. Review the identified bad cells and suggested repairs. Click "Approve All" to accept.
4. Open the **Export** tab to download your cleaned data and audit log.

## Architecture

- **PyTorch**: Used for the SCARF contrastive model.
- **DuckDB**: Used for fast data ingestion and column profiling.
- **Scikit-Learn**: Used for kNN anomaly scoring, Isolation Forest baselines, and preprocessing.
- **Streamlit**: Provides the interactive UI.
