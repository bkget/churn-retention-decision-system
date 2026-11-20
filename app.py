"""Streamlit entry point: Explainable Churn & Retention ROI Decision System.

Run with:  make app   (or: streamlit run app.py)
"""

from __future__ import annotations

import streamlit as st

from src.app_views import common, explorer, overview, roi_simulator
from src.config import load_settings

st.set_page_config(
    page_title="Churn & Retention ROI Decision System",
    page_icon=None,
    layout="wide",
)

settings = load_settings()
data = common.load_artifacts()

st.title("Churn & Retention ROI Decision System")
st.caption(
    "Which customers to save, why they are at risk, and whether saving them pays for itself."
)

if data["missing"]:
    st.error(
        "Model artifacts not found: " + ", ".join(data["missing"]) + ".\n\n"
        "Run `make train` first (trains the models, computes SHAP drivers and the ROI threshold)."
    )
    st.stop()

tab_overview, tab_roi, tab_explorer = st.tabs(
    ["Executive Overview", "ROI Simulator", "At-Risk Account Explorer"]
)
with tab_overview:
    overview.render(data, settings)
with tab_roi:
    roi_state = roi_simulator.render(data, settings)
with tab_explorer:
    explorer.render(data, roi_state)
