"""Data loading, schema adaptation, cleaning and feature engineering.

Two sources are supported and normalised into one canonical snake_case schema:

* ``ibm_extended`` - IBM Telco v11.1.3+ merged release (49 columns incl. CLTV,
  Churn Reason, Satisfaction Score). Default.
* ``kaggle_21``    - the original 21-column Kaggle/IBM file (fallback).

Everything downstream (training, SHAP, ROI, app) only sees the canonical schema.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_settings, resolve

log = logging.getLogger(__name__)

TARGET = "churn"
ID_COL = "customer_id"

# --------------------------------------------------------------------------- #
# Canonical feature groups (a column is used only if the source provides it)
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "age",
    "number_of_dependents",
    "number_of_referrals",
    "avg_monthly_gb_download",
    "avg_monthly_long_distance_charges",
    "total_extra_data_charges",
    "total_long_distance_charges",
    "total_refunds",
    # engineered
    "avg_monthly_spend",
    "service_count",
    "charge_vs_tenure_gap",
]
BINARY_FEATURES = [
    "senior_citizen",
    "under_30",
    "partner",
    "married",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "streaming_music",
    "unlimited_data",
    "paperless_billing",
    "referred_a_friend",
    # engineered
    "is_new_fiber_m2m",
]
CATEGORICAL_FEATURES = ["contract", "internet_type", "payment_method", "offer", "gender"]

# Services counted in ``service_count`` - present in BOTH sources so the
# feature means the same thing regardless of data source (range 0..9).
SERVICE_COLUMNS = [
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

# --------------------------------------------------------------------------- #
# Source-specific renames
# --------------------------------------------------------------------------- #
_IBM_EXTENDED_RENAMES = {
    "tenure_in_months": "tenure",
    "monthly_charge": "monthly_charges",
    "device_protection_plan": "device_protection",
    "premium_tech_support": "tech_support",
}

_KAGGLE_RENAMES = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_type",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

_CONTRACT_MAP = {
    "month-to-month": "Month-to-Month",
    "one year": "One Year",
    "two year": "Two Year",
}
_INTERNET_TYPE_MAP = {
    "fiber optic": "Fiber Optic",
    "dsl": "DSL",
    "cable": "Cable",
    "no": "None",
    "none": "None",
}
_YES_NO_MAP = {
    "yes": 1,
    "no": 0,
    "no internet service": 0,
    "no phone service": 0,
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0,
}


def _snake(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int)
    mapped = series.astype(str).str.strip().str.lower().map(_YES_NO_MAP)
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique())[:5]
        raise ValueError(f"Unmapped binary values in '{series.name}': {bad}")
    return mapped.astype(int)


# --------------------------------------------------------------------------- #
# Loading + schema adapter
# --------------------------------------------------------------------------- #
def load_raw(settings: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read the configured raw source and return it in the canonical schema (uncleaned)."""
    settings = settings or load_settings()
    source = settings["data"]["source"]
    if source == "ibm_extended":
        frames = [pd.read_csv(resolve(p)) for p in settings["data"]["ibm_extended_files"]]
        return adapt_ibm_extended(pd.concat(frames, ignore_index=True))
    if source == "kaggle_21":
        # keep TotalCharges as raw text so the cleaning step sees the blanks
        raw = pd.read_csv(resolve(settings["data"]["kaggle_21_file"]), dtype={"TotalCharges": str})
        return adapt_kaggle_21(raw)
    raise ValueError(f"Unknown data source: {source!r}")


def adapt_ibm_extended(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=_snake).rename(columns=_IBM_EXTENDED_RENAMES)
    out["internet_type"] = out["internet_type"].fillna("None")
    out["offer"] = out["offer"].fillna("None")
    out.attrs["source"] = "ibm_extended"
    return _normalise_common(out)


def adapt_kaggle_21(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(_KAGGLE_RENAMES) - set(df.columns)
    if missing:
        raise ValueError(f"21-column file is missing expected columns: {sorted(missing)}")
    out = df.rename(columns=_KAGGLE_RENAMES)
    out["gender"] = out["gender"].str.strip()
    # Kaggle encodes "has internet" inside InternetService; split it out.
    internet_type = out["internet_type"].astype(str).str.strip().str.lower()
    out["internet_service"] = (internet_type != "no").astype(int)
    out.attrs["source"] = "kaggle_21"
    return _normalise_common(out)


def _normalise_common(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["contract"] = out["contract"].astype(str).str.strip().str.lower().map(_CONTRACT_MAP)
    out["internet_type"] = (
        out["internet_type"].astype(str).str.strip().str.lower().map(_INTERNET_TYPE_MAP)
    )
    if out["contract"].isna().any() or out["internet_type"].isna().any():
        raise ValueError("Unmapped values in contract / internet_type")
    out["payment_method"] = out["payment_method"].astype(str).str.strip()
    for col in BINARY_FEATURES:
        if col in out.columns:
            out[col] = _to_binary(out[col])
    out[TARGET] = _to_binary(out[TARGET])
    return out


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``total_charges`` to float.

    Blank values are legitimate only for brand-new customers (``tenure == 0``,
    never billed) and become 0.0. A blank for a customer with tenure > 0 is a
    data error and raises instead of being silently imputed.
    """
    out = df.copy()
    as_text = out["total_charges"].astype(str).str.strip()
    numeric = pd.to_numeric(as_text.replace({"": np.nan, "nan": np.nan}), errors="coerce")
    missing = numeric.isna()
    bad = missing & (out["tenure"] > 0)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} rows have missing total_charges with tenure > 0: "
            f"{out.loc[bad, ID_COL].head().tolist()}"
        )
    out["total_charges"] = numeric.fillna(0.0).astype(float)
    return out


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame, new_customer_months: int = 12) -> pd.DataFrame:
    out = df.copy()
    out["avg_monthly_spend"] = out["total_charges"] / out["tenure"].clip(lower=1)
    # Positive gap = paying more now than their historical average (price creep).
    out["charge_vs_tenure_gap"] = out["monthly_charges"] - out["avg_monthly_spend"]
    out["service_count"] = out[SERVICE_COLUMNS].sum(axis=1).astype(int)
    out["is_new_fiber_m2m"] = (
        (out["tenure"] <= new_customer_months)
        & (out["internet_type"] == "Fiber Optic")
        & (out["contract"] == "Month-to-Month")
    ).astype(int)
    return out


def feature_columns(
    df: pd.DataFrame, settings: dict[str, Any] | None = None
) -> dict[str, list[str]]:
    """Model input columns available in ``df``, with leakage + drop lists removed."""
    settings = settings or load_settings()
    excluded = set(settings["features"]["leakage_blocklist"]) | set(settings["features"]["drop"])
    excluded |= {TARGET, ID_COL}

    def keep(cols: list[str]) -> list[str]:
        return [c for c in cols if c in df.columns and c not in excluded]

    return {
        "numeric": keep(NUMERIC_FEATURES),
        "binary": keep(BINARY_FEATURES),
        "categorical": keep(CATEGORICAL_FEATURES),
    }


def all_feature_names(groups: dict[str, list[str]]) -> list[str]:
    return groups["numeric"] + groups["binary"] + groups["categorical"]


def prepare_dataset(settings: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load -> adapt -> clean -> engineer. Returns the full canonical frame."""
    settings = settings or load_settings()
    df = load_raw(settings)
    source = df.attrs.get("source")
    df = clean_total_charges(df)
    df = engineer_features(df, settings["features"]["new_customer_months"])
    df.attrs["source"] = source
    if df[ID_COL].duplicated().any():
        raise ValueError("Duplicate customer IDs after loading")
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    df = prepare_dataset(settings)
    out_path = resolve(settings["data"]["processed_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    groups = feature_columns(df, settings)
    log.info(
        "source=%s rows=%d churn_rate=%.3f features=%d -> %s",
        df.attrs.get("source"),
        len(df),
        df[TARGET].mean(),
        len(all_feature_names(groups)),
        out_path,
    )


if __name__ == "__main__":
    main()
