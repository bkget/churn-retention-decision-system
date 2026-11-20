"""ROI maths, threshold optimiser and LTV tests (all against hand-computed values)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.roi import (
    NO_CAMPAIGN_THRESHOLD,
    LtvSettings,
    RoiParams,
    compute_ltv,
    cost_benefit_matrix,
    ev_rule_mask,
    evaluate_policy,
    optimal_threshold,
    params_from_settings,
    profit_curve,
)

# Five customers:        p      churned   LTV
P = np.array([0.90, 0.60, 0.30, 0.10, 0.05])
Y = np.array([1, 1, 0, 1, 0])
LTV = np.array([1000.0, 500.0, 800.0, 200.0, 100.0])
PARAMS = RoiParams(retention_offer_cost=50.0, offer_acceptance_rate=0.4)
GRID = np.array([0.05, 0.10, 0.30, 0.50, 0.90, NO_CAMPAIGN_THRESHOLD])


# --------------------------------------------------------------------------- #
# evaluate_policy
# --------------------------------------------------------------------------- #
def test_realised_economics_at_threshold_050():
    r = evaluate_policy(P >= 0.5, Y, LTV, PARAMS)
    assert r["targeted"] == 2
    assert r["churners_reached"] == 2
    assert r["revenue_saved"] == pytest.approx(0.4 * (1000 + 500))  # 600
    assert r["campaign_cost"] == pytest.approx(100.0)
    assert r["net_roi"] == pytest.approx(500.0)
    assert r["roi_pct"] == pytest.approx(500.0)


def test_expected_economics_use_probabilities():
    r = evaluate_policy(P >= 0.5, P, LTV, PARAMS)
    assert r["revenue_saved"] == pytest.approx(0.4 * (0.9 * 1000 + 0.6 * 500))  # 480
    assert r["net_roi"] == pytest.approx(380.0)
    assert r["churners_reached"] == pytest.approx(1.5)


def test_false_positives_cost_money_but_save_nothing():
    # customer 3 (p=0.30) did not churn: targeting adds cost, no revenue
    with_fp = evaluate_policy(P >= 0.3, Y, LTV, PARAMS)
    without = evaluate_policy(P >= 0.5, Y, LTV, PARAMS)
    assert with_fp["revenue_saved"] == pytest.approx(without["revenue_saved"])
    assert with_fp["campaign_cost"] - without["campaign_cost"] == pytest.approx(50.0)


def test_cost_only_when_accepted_flag():
    params = RoiParams(50.0, 0.4, cost_applies_to_all_targeted=False)
    r = evaluate_policy(P >= 0.5, Y, LTV, params)
    assert r["campaign_cost"] == pytest.approx(2 * 0.4 * 50.0)  # 40, not 100
    assert r["net_roi"] == pytest.approx(600 - 40)


def test_nobody_targeted_is_zero_with_nan_roi():
    r = evaluate_policy(np.zeros(5, dtype=bool), Y, LTV, PARAMS)
    assert (r["targeted"], r["revenue_saved"], r["campaign_cost"], r["net_roi"]) == (0, 0, 0, 0)
    assert np.isnan(r["roi_pct"])


def test_threshold_above_one_targets_nobody():
    curve = profit_curve(P, P, LTV, PARAMS, thresholds=np.array([NO_CAMPAIGN_THRESHOLD]))
    assert curve.loc[0, "targeted"] == 0
    assert curve.loc[0, "net_roi"] == 0


# --------------------------------------------------------------------------- #
# Threshold optimiser
# --------------------------------------------------------------------------- #
def test_curve_matches_hand_calculation():
    curve = profit_curve(P, P, LTV, PARAMS, GRID).set_index("threshold")
    expected_net = {0.05: 336.0, 0.10: 384.0, 0.30: 426.0, 0.50: 380.0, 0.90: 310.0}
    for t, net in expected_net.items():
        assert curve.loc[t, "net_roi"] == pytest.approx(net), t


def test_optimal_threshold_picks_max_expected_profit():
    best = optimal_threshold(profit_curve(P, P, LTV, PARAMS, GRID))
    assert best["threshold"] == pytest.approx(0.30)
    assert best["net_roi"] == pytest.approx(426.0)
    assert best["targeted"] == 3
    assert best["profitable"] is True


def test_optimal_threshold_beats_fixed_050():
    curve = profit_curve(P, P, LTV, PARAMS, GRID).set_index("threshold")
    best = optimal_threshold(curve.reset_index())
    assert best["net_roi"] > curve.loc[0.50, "net_roi"]


def test_zero_acceptance_means_no_campaign():
    params = RoiParams(50.0, 0.0)
    best = optimal_threshold(profit_curve(P, P, LTV, params, GRID))
    assert best["profitable"] is False
    assert best["targeted"] == 0 and best["net_roi"] == 0
    assert best["threshold"] > 1.0


def test_offer_costlier_than_any_ltv_means_no_campaign():
    params = RoiParams(10_000.0, 1.0)
    assert optimal_threshold(profit_curve(P, P, LTV, params, GRID))["profitable"] is False


def test_zero_cost_targets_every_customer_with_positive_risk():
    params = RoiParams(0.0, 0.4)
    best = optimal_threshold(profit_curve(P, P, LTV, params, GRID))
    assert best["threshold"] == pytest.approx(0.05)
    assert best["targeted"] == 5


def test_tie_goes_to_higher_threshold():
    p = np.array([0.8, 0.8])
    curve = profit_curve(p, p, np.array([1000.0, 1000.0]), PARAMS, np.array([0.2, 0.5, 0.8]))
    assert optimal_threshold(curve)["threshold"] == pytest.approx(0.8)


def test_higher_offer_cost_never_lowers_the_optimal_threshold():
    t = [
        optimal_threshold(profit_curve(P, P, LTV, RoiParams(c, 0.4), GRID))["threshold"]
        for c in (10.0, 50.0, 150.0)
    ]
    assert t == sorted(t)


# --------------------------------------------------------------------------- #
# EV rule + cost-benefit matrix
# --------------------------------------------------------------------------- #
def test_ev_rule_matches_hand_calculation():
    # p * 0.4 * LTV: 360, 120, 96, 8, 2 vs cost 50 -> first three
    assert ev_rule_mask(P, LTV, PARAMS).tolist() == [True, True, True, False, False]


def test_cost_benefit_matrix_rows_and_values():
    m = cost_benefit_matrix(P, Y, LTV, PARAMS, optimal_t=0.3).set_index("strategy")
    assert list(m.index) == [
        "No campaign",
        "Target everyone",
        "Fixed threshold 0.50",
        "Profit-optimal threshold 0.300",
        "Per-customer EV rule (upper bound)",
    ]
    assert m.loc["No campaign", "net_roi"] == 0
    everyone = m.loc["Target everyone"]
    assert everyone["revenue_saved"] == pytest.approx(0.4 * (1000 + 500 + 200))  # 680
    assert everyone["campaign_cost"] == pytest.approx(250.0)
    assert everyone["net_roi"] == pytest.approx(430.0)
    assert m.loc["Fixed threshold 0.50", "net_roi"] == pytest.approx(500.0)
    opt = m.loc["Profit-optimal threshold 0.300"]
    assert (opt["targeted"], opt["net_roi"]) == (3, pytest.approx(450.0))


def test_net_equals_revenue_minus_cost_everywhere():
    curve = profit_curve(P, Y, LTV, PARAMS)
    assert (curve["net_roi"] - (curve["revenue_saved"] - curve["campaign_cost"])).abs().max() < 1e-9


# --------------------------------------------------------------------------- #
# LTV + parameters
# --------------------------------------------------------------------------- #
def test_formula_ltv_uses_contract_months():
    df = pd.DataFrame(
        {
            "monthly_charges": [100.0, 50.0, 20.0],
            "contract": ["Month-to-Month", "One Year", "Two Year"],
        }
    )
    ltv = compute_ltv(df, LtvSettings(source="formula"))
    assert ltv.tolist() == [100 * 12, 50 * 18, 20 * 30]


def test_ltv_margin_scales_value():
    df = pd.DataFrame({"monthly_charges": [100.0], "contract": ["Month-to-Month"]})
    assert compute_ltv(df, LtvSettings(margin=0.6)).item() == pytest.approx(720.0)


def test_cltv_source_uses_column_and_requires_it():
    df = pd.DataFrame({"monthly_charges": [1.0], "contract": ["One Year"], "cltv": [4321]})
    assert compute_ltv(df, LtvSettings(source="cltv")).item() == 4321
    with pytest.raises(ValueError, match="cltv"):
        compute_ltv(df.drop(columns="cltv"), LtvSettings(source="cltv"))


def test_unknown_contract_raises():
    df = pd.DataFrame({"monthly_charges": [10.0], "contract": ["Five Year"]})
    with pytest.raises(ValueError, match="Five Year"):
        compute_ltv(df, LtvSettings())


@pytest.mark.parametrize("kwargs", [{"offer_acceptance_rate": 1.5}, {"retention_offer_cost": -1}])
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        RoiParams(**kwargs)


def test_settings_load_into_params(settings):
    params, ltv = params_from_settings(settings)
    assert params.cost_applies_to_all_targeted is True  # conservative default
    assert ltv.source == "formula"
    assert set(ltv.months_by_contract) == {"Month-to-Month", "One Year", "Two Year"}
