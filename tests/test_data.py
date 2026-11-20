"""Data cleaning, schema adapter, feature engineering and leakage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.config import resolve
from src.data import (
    SERVICE_COLUMNS,
    TARGET,
    adapt_ibm_extended,
    adapt_kaggle_21,
    all_feature_names,
    clean_total_charges,
    engineer_features,
    feature_columns,
    prepare_dataset,
)


# --------------------------------------------------------------------------- #
# TotalCharges cleaning
# --------------------------------------------------------------------------- #
def test_blank_total_charges_with_zero_tenure_becomes_zero(kaggle_raw):
    df = clean_total_charges(adapt_kaggle_21(kaggle_raw))
    assert df["total_charges"].dtype == float
    assert df.loc[df["tenure"] == 0, "total_charges"].eq(0.0).all()
    assert df["total_charges"].notna().all()


def test_numeric_strings_are_parsed(kaggle_raw):
    df = clean_total_charges(adapt_kaggle_21(kaggle_raw))
    assert df.loc[df["customer_id"] == "A2", "total_charges"].item() == pytest.approx(470.25)
    assert df.loc[df["customer_id"] == "A4", "total_charges"].item() == pytest.approx(1800.0)


def test_blank_total_charges_with_positive_tenure_raises(kaggle_raw):
    kaggle_raw.loc[1, "TotalCharges"] = " "  # tenure 5 -> must not be silently imputed
    with pytest.raises(ValueError, match="tenure > 0"):
        clean_total_charges(adapt_kaggle_21(kaggle_raw))


# --------------------------------------------------------------------------- #
# Schema adapter
# --------------------------------------------------------------------------- #
def test_kaggle_adapter_normalises_values(kaggle_raw):
    df = adapt_kaggle_21(kaggle_raw)
    assert set(df["contract"]) <= {"Month-to-Month", "One Year", "Two Year"}
    assert df.loc[df["customer_id"] == "A4", "internet_type"].item() == "None"
    assert df.loc[df["customer_id"] == "A4", "internet_service"].item() == 0
    # "No internet service" / "No phone service" -> 0
    assert df.loc[df["customer_id"] == "A4", "online_security"].item() == 0
    assert df.loc[df["customer_id"] == "A3", "multiple_lines"].item() == 0
    assert df[TARGET].tolist() == [0, 1, 0, 0, 1]


def test_ibm_adapter_renames_and_fills(ibm_raw):
    df = adapt_ibm_extended(ibm_raw)
    for col in ("tenure", "monthly_charges", "tech_support", "device_protection", "cltv"):
        assert col in df.columns
    assert df.loc[df["customer_id"] == "B3", "internet_type"].item() == "None"
    assert df.loc[df["customer_id"] == "B2", "offer"].item() == "None"


def test_kaggle_adapter_rejects_missing_columns(kaggle_raw):
    with pytest.raises(ValueError, match="missing expected columns"):
        adapt_kaggle_21(kaggle_raw.drop(columns=["Contract"]))


def test_unmapped_binary_value_raises(kaggle_raw):
    kaggle_raw.loc[0, "Partner"] = "Maybe"
    with pytest.raises(ValueError, match="Unmapped binary"):
        adapt_kaggle_21(kaggle_raw)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
@pytest.fixture
def engineered(kaggle_raw):
    return engineer_features(clean_total_charges(adapt_kaggle_21(kaggle_raw)))


def test_avg_monthly_spend_handles_zero_tenure(engineered):
    row = engineered.set_index("customer_id")
    assert row.loc["A1", "avg_monthly_spend"] == 0.0  # tenure 0, no division error
    assert row.loc["A2", "avg_monthly_spend"] == pytest.approx(470.25 / 5)
    assert np.isfinite(engineered["avg_monthly_spend"]).all()


def test_charge_vs_tenure_gap(engineered):
    row = engineered.set_index("customer_id").loc["A2"]
    assert row["charge_vs_tenure_gap"] == pytest.approx(95.5 - 470.25 / 5)


def test_service_count_bounds_and_values(engineered):
    assert engineered["service_count"].between(0, len(SERVICE_COLUMNS)).all()
    row = engineered.set_index("customer_id")
    # A4: phone + multiple lines only, no internet
    assert row.loc["A4", "service_count"] == 2
    # A3: no phone; DSL + security + backup + protection + support
    assert row.loc["A3", "service_count"] == 5


def test_is_new_fiber_m2m_truth_table(engineered):
    flag = engineered.set_index("customer_id")["is_new_fiber_m2m"]
    assert flag["A1"] == 1  # tenure 0, fiber, m2m
    assert flag["A2"] == 1  # tenure 5, fiber, m2m
    assert flag["A5"] == 1  # tenure 3, fiber, m2m
    assert flag["A3"] == 0  # DSL, one year
    assert flag["A4"] == 0  # no internet, two year


def test_is_new_fiber_m2m_respects_tenure_cutoff(kaggle_raw):
    kaggle_raw.loc[1, "tenure"] = 13
    kaggle_raw.loc[1, "TotalCharges"] = "1200"
    df = engineer_features(clean_total_charges(adapt_kaggle_21(kaggle_raw)), new_customer_months=12)
    assert df.set_index("customer_id").loc["A2", "is_new_fiber_m2m"] == 0


# --------------------------------------------------------------------------- #
# Leakage guards
# --------------------------------------------------------------------------- #
def test_feature_columns_exclude_leakage_ids_and_target(ibm_raw, settings):
    df = engineer_features(clean_total_charges(adapt_ibm_extended(ibm_raw)))
    features = set(all_feature_names(feature_columns(df, settings)))
    blocked = set(settings["features"]["leakage_blocklist"]) | set(settings["features"]["drop"])
    assert not features & blocked
    assert TARGET not in features and "customer_id" not in features
    # the leakage columns ARE present in the data - they are excluded, not missing
    assert {"satisfaction_score", "churn_score", "cltv", "churn_reason"} <= set(df.columns)


def test_blocklist_covers_known_label_proxies(settings):
    required = {
        "churn_category",
        "churn_reason",
        "churn_score",
        "customer_status",
        "satisfaction_score",
        "offer",
    }
    assert required <= set(settings["features"]["leakage_blocklist"])


@pytest.mark.skipif(
    not resolve("data/raw/ibm_extended/train.csv").exists(), reason="raw data not downloaded"
)
def test_no_model_feature_is_a_near_perfect_predictor(settings):
    """A single column with AUC > 0.90 almost always means label leakage."""
    df = prepare_dataset(settings)
    for col in all_feature_names(feature_columns(df, settings)):
        x = df[col]
        if not pd.api.types.is_numeric_dtype(x):
            x = x.map(df.groupby(col)[TARGET].mean())
        auc = roc_auc_score(df[TARGET], x)
        assert max(auc, 1 - auc) < 0.90, f"{col} looks like leakage (AUC {auc:.3f})"


@pytest.mark.skipif(
    not resolve("data/raw/ibm_extended/train.csv").exists(), reason="raw data not downloaded"
)
def test_real_ibm_dataset_loads_clean(settings):
    df = prepare_dataset(settings)
    assert len(df) == 7043
    assert df["customer_id"].is_unique
    assert df[TARGET].isin([0, 1]).all()
    assert df["total_charges"].notna().all()


@pytest.mark.skipif(
    not resolve("data/raw/Telco-Customer-Churn.csv").exists(), reason="21-col file not present"
)
def test_real_kaggle_file_has_only_zero_tenure_blanks(settings):
    raw = pd.read_csv(resolve(settings["data"]["kaggle_21_file"]), dtype={"TotalCharges": str})
    blanks = raw["TotalCharges"].str.strip() == ""
    assert blanks.sum() == 11
    assert (raw.loc[blanks, "tenure"] == 0).all()
    df = clean_total_charges(adapt_kaggle_21(raw))
    assert df["total_charges"].notna().all()
