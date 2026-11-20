"""SHAP aggregation, driver selection / text, agreement table and additivity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explain import (
    DEMOGRAPHIC_FEATURES,
    aggregate_shap,
    build_driver_table,
    compute_shap,
    driver_agreement,
    driver_text,
    original_feature_map,
    top_risk_drivers,
)
from src.train import make_xgb


# --------------------------------------------------------------------------- #
# One-hot aggregation
# --------------------------------------------------------------------------- #
def test_one_hot_columns_map_back_to_original_feature():
    names = [
        "tenure",
        "contract_Month-to-Month",
        "contract_Two Year",
        "payment_method_Mailed Check",
    ]
    m = original_feature_map(names, ["contract", "payment_method"])
    assert m == {
        "tenure": "tenure",
        "contract_Month-to-Month": "contract",
        "contract_Two Year": "contract",
        "payment_method_Mailed Check": "payment_method",
    }


def test_aggregate_shap_sums_one_hot_columns():
    names = ["tenure", "contract_A", "contract_B"]
    values = np.array([[0.5, 0.2, -0.1], [-0.3, 0.0, 0.4]])
    agg = aggregate_shap(values, names, ["contract"])
    assert list(agg.columns) == ["contract", "tenure"]
    assert agg["contract"].tolist() == pytest.approx([0.1, 0.4])
    assert agg["tenure"].tolist() == pytest.approx([0.5, -0.3])


# --------------------------------------------------------------------------- #
# Driver selection
# --------------------------------------------------------------------------- #
def test_top_drivers_are_positive_only_and_ordered():
    s = pd.Series({"contract": 0.9, "tenure": 0.4, "monthly_charges": 0.1, "tech_support": -0.7})
    assert top_risk_drivers(s, 3) == ["contract", "tenure", "monthly_charges"]


def test_top_drivers_fewer_than_three_when_few_positive():
    s = pd.Series({"contract": 0.9, "tenure": -0.4, "monthly_charges": -0.1})
    assert top_risk_drivers(s, 3) == ["contract"]


def test_demographic_features_never_shown_as_drivers():
    s = pd.Series({f: 5.0 for f in DEMOGRAPHIC_FEATURES} | {"contract": 0.1, "tenure": 0.05})
    assert top_risk_drivers(s, 3) == ["contract", "tenure"]


def test_driver_table_gives_three_drivers_with_text():
    shap_df = pd.DataFrame(
        {
            "contract": [0.9, 0.1],
            "tenure": [0.5, 0.2],
            "monthly_charges": [0.3, 0.05],
            "tech_support": [0.2, -0.5],
        },
        index=[10, 11],
    )
    feats = pd.DataFrame(
        {
            "contract": ["Month-to-Month", "Two Year"],
            "tenure": [3, 40],
            "monthly_charges": [95.0, 30.0],
            "tech_support": [0, 1],
        },
        index=[10, 11],
    )
    t = build_driver_table(shap_df, feats)
    assert t.loc[10, "driver_1_text"] == "Month-to-month contract (no lock-in)"
    assert t.loc[10, "driver_2_text"] == "Brand-new customer (3 months of tenure)"
    assert t.loc[10, "driver_3_text"] == "High monthly bill ($95)"
    assert [t.loc[11, f"driver_{k}_feature"] for k in (1, 2, 3)] == [
        "tenure",
        "contract",
        "monthly_charges",
    ]  # tech_support is negative, so exactly three positive drivers remain
    assert t.loc[11, "driver_1_shap"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# Plain-English text
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("feature", "value", "expected"),
    [
        ("tenure", 1, "Brand-new customer (1 month of tenure)"),
        ("tenure", 8, "First-year customer (8 months of tenure)"),
        ("contract", "One Year", "One-year contract, not yet on a 2-year commitment"),
        ("internet_type", "Fiber Optic", "Fiber Optic internet service"),
        ("tech_support", 0, "No premium tech support"),
        ("is_new_fiber_m2m", 1, "New month-to-month fiber customer"),
        ("service_count", 2, "Only 2 of 9 services"),
        ("charge_vs_tenure_gap", 12.4, "Now paying $12/month more than their historical average"),
        ("charge_vs_tenure_gap", -9.6, "Now paying $10/month less than their historical average"),
        ("number_of_referrals", 0, "Has made no referrals"),
        ("number_of_referrals", 1, "Has made few referrals (1)"),
    ],
)
def test_driver_text(feature, value, expected):
    assert driver_text(feature, pd.Series({feature: value})) == expected


def test_unknown_feature_gets_generic_text():
    assert driver_text("brand_new_col", pd.Series({"brand_new_col": 7})) == "Brand new col: 7"


# --------------------------------------------------------------------------- #
# Driver vs stated reason
# --------------------------------------------------------------------------- #
def test_driver_agreement_hit_and_baseline_rates():
    churners = pd.DataFrame(
        {
            "churn_category": ["Price", "Price", "Competitor", "Competitor"],
            "driver_1_feature": ["monthly_charges", "contract", "contract", "tenure"],
            "driver_2_feature": ["tenure", "tenure", "paperless_billing", "payment_method"],
            "driver_3_feature": [None, None, None, None],
        }
    )
    a = driver_agreement(churners, n=3).set_index("stated_category")
    # Price: 1 of 2 price churners has a price feature; 0 of 2 others do
    assert a.loc["Price", "hit_rate"] == pytest.approx(0.5)
    assert a.loc["Price", "baseline_rate"] == pytest.approx(0.0)
    # Competitor: both have contract/tenure; the price churners' baseline is 1.0 vs 2/2
    assert a.loc["Competitor", "hit_rate"] == pytest.approx(1.0)
    assert a.loc["Competitor", "baseline_rate"] == pytest.approx(1.0)
    assert a.loc["Competitor", "lift"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# SHAP additivity on a real (tiny) XGBoost pipeline
# --------------------------------------------------------------------------- #
def test_shap_is_additive_and_aggregated_to_original_features():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n).astype(float),
            "monthly_charges": rng.uniform(20, 110, n),
            "tech_support": rng.integers(0, 2, n),
            "contract": rng.choice(["Month-to-Month", "One Year", "Two Year"], n),
        }
    )
    y = ((X["tenure"] < 12) & (X["contract"] == "Month-to-Month")).astype(int)
    groups = {
        "numeric": ["tenure", "monthly_charges"],
        "binary": ["tech_support"],
        "categorical": ["contract"],
    }
    model = make_xgb(groups, {"n_estimators": 25, "max_depth": 3}, seed=1).fit(X, y)

    shap_df, base, margin = compute_shap(model, X, groups["categorical"])

    assert set(shap_df.columns) == set(X.columns)  # one column per ORIGINAL feature
    assert not any(c.startswith("contract_") for c in shap_df.columns)
    np.testing.assert_allclose(base + shap_df.sum(axis=1).to_numpy(), margin, atol=1e-3)
    # the engineered signal must dominate the importance ranking
    top2 = shap_df.abs().mean().sort_values(ascending=False).index[:2]
    assert set(top2) == {"tenure", "contract"}
