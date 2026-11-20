"""Shared loading, formatting and colour constants for the Streamlit views."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import load_settings, resolve

# Reference categorical palette (light-mode steps), fixed slot order.
BLUE = "#2a78d6"  # series 1 / sequential hue
ORANGE = "#eb6834"  # series 2 / "pushes risk up"
MUTED = "#a9a8a2"  # de-emphasised marks
# One-hue sequential ramp (light -> dark) for magnitude.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

REQUIRED_ARTIFACTS = [
    "scored_customers.parquet",
    "shap_values.parquet",
    "metrics.json",
    "explain_summary.json",
    "global_importance.csv",
]


@st.cache_data(show_spinner="Loading model artifacts...")
def load_artifacts() -> dict:
    """Read everything `make train` produced. Returns {"missing": [...]} if it has not run."""
    settings = load_settings()
    art = resolve(settings["paths"]["artifacts_dir"])
    missing = [n for n in REQUIRED_ARTIFACTS if not (art / n).exists()]
    if missing:
        return {"missing": missing}
    return {
        "missing": [],
        "scored": pd.read_parquet(art / "scored_customers.parquet"),
        "shap": pd.read_parquet(art / "shap_values.parquet"),
        "importance": pd.read_csv(art / "global_importance.csv"),
        "metrics": json.loads((art / "metrics.json").read_text()),
        "explain": json.loads((art / "explain_summary.json").read_text()),
    }


def money(x: float, decimals: int = 0) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.{decimals}f}"


def pct(x: float, decimals: int = 1) -> str:
    return f"{x * 100:.{decimals}f}%"


def threshold_label(t: float, profitable: bool = True) -> str:
    return f"{t:.3f}" if profitable else "no campaign"
