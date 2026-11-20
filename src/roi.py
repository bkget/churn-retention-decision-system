"""Retention-campaign ROI maths and profit-maximising threshold.

Everything here is a pure, vectorised function of arrays - no I/O - so the
Streamlit sliders can call it on every interaction and the tests can check it
against hand-computed examples.  ``main()`` is the only part that touches disk.

Economics of targeting customer *i* with a retention offer
----------------------------------------------------------
    revenue saved_i = P(churn_i) * acceptance_rate * LTV_i      (expected)
                    = 1[churned_i] * acceptance_rate * LTV_i    (realised)
    campaign cost_i = retention_offer_cost                      (conservative default:
                      paid for every targeted customer, false positives included)
                    = acceptance_rate * retention_offer_cost    (if the offer is only
                      paid when accepted)
    net_i           = revenue saved_i - campaign cost_i

LTV is the revenue the customer would still generate if retained.  Multiply by
``ltv_margin`` (< 1) to turn revenue into contribution margin.

Usage:  python -m src.roi        (after train + explain)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src import cli_ui
from src.config import load_settings, resolve
from src.data import TARGET

log = logging.getLogger(__name__)

DEFAULT_MONTHS_BY_CONTRACT = {"Month-to-Month": 12, "One Year": 18, "Two Year": 30}
# Strictly above any probability, so "target nobody" is representable as a threshold.
NO_CAMPAIGN_THRESHOLD = float(np.nextafter(1.0, 2.0))


@dataclass(frozen=True)
class RoiParams:
    retention_offer_cost: float = 50.0
    offer_acceptance_rate: float = 0.30
    cost_applies_to_all_targeted: bool = True

    def __post_init__(self) -> None:
        if self.retention_offer_cost < 0:
            raise ValueError("retention_offer_cost must be >= 0")
        if not 0.0 <= self.offer_acceptance_rate <= 1.0:
            raise ValueError("offer_acceptance_rate must be in [0, 1]")


@dataclass(frozen=True)
class LtvSettings:
    source: str = "formula"  # "formula" | "cltv"
    months_by_contract: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MONTHS_BY_CONTRACT)
    )
    margin: float = 1.0


# --------------------------------------------------------------------------- #
# LTV
# --------------------------------------------------------------------------- #
def compute_ltv(df: pd.DataFrame, ltv: LtvSettings | None = None) -> pd.Series:
    """Customer lifetime value: revenue still to come if the customer is retained."""
    ltv = ltv or LtvSettings()
    if ltv.source == "cltv":
        if "cltv" not in df.columns:
            raise ValueError("ltv source 'cltv' requires a 'cltv' column (use 'formula')")
        value = df["cltv"].astype(float)
    elif ltv.source == "formula":
        months = df["contract"].map(ltv.months_by_contract)
        if months.isna().any():
            bad = sorted(df.loc[months.isna(), "contract"].astype(str).unique())
            raise ValueError(f"No expected_remaining_months for contract(s): {bad}")
        value = df["monthly_charges"].astype(float) * months.astype(float)
    else:
        raise ValueError(f"Unknown ltv source: {ltv.source!r}")
    return (value * ltv.margin).rename("ltv")


# --------------------------------------------------------------------------- #
# Campaign economics
# --------------------------------------------------------------------------- #
def _cost_per_targeted(params: RoiParams) -> float:
    if params.cost_applies_to_all_targeted:
        return params.retention_offer_cost
    return params.offer_acceptance_rate * params.retention_offer_cost


def evaluate_policy(
    target: np.ndarray,
    churn: np.ndarray,
    ltv: np.ndarray,
    params: RoiParams,
) -> dict[str, float]:
    """Economics of targeting the customers where ``target`` is True.

    ``churn`` is P(churn) for an *expected* evaluation, or the 0/1 outcome for a
    *realised* one - the formulas are identical.
    """
    target = np.asarray(target, dtype=bool)
    churn = np.asarray(churn, dtype=float)
    ltv = np.asarray(ltv, dtype=float)
    n = int(target.sum())
    reached = float(churn[target].sum())
    revenue = float(params.offer_acceptance_rate * (churn[target] * ltv[target]).sum())
    cost = n * _cost_per_targeted(params)
    net = revenue - cost
    return {
        "targeted": n,
        "churners_reached": reached,
        "revenue_saved": revenue,
        "campaign_cost": cost,
        "net_roi": net,
        "roi_pct": float(net / cost * 100.0) if cost > 0 else float("nan"),
    }


def profit_curve(
    proba: np.ndarray,
    churn: np.ndarray,
    ltv: np.ndarray,
    params: RoiParams,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Campaign economics at each probability threshold (target iff p >= t)."""
    proba = np.asarray(proba, dtype=float)
    if thresholds is None:
        thresholds = np.append(np.linspace(0.0, 1.0, 201), NO_CAMPAIGN_THRESHOLD)
    rows = [
        {"threshold": float(t), **evaluate_policy(proba >= t, churn, ltv, params)}
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def optimal_threshold(curve: pd.DataFrame) -> dict[str, float]:
    """Profit-maximising row of a profit curve.

    Ties go to the HIGHER threshold (the smaller campaign). If no threshold has a
    positive net, the answer is "run no campaign": threshold above 1, zero targeted.
    """
    best_net = curve["net_roi"].max()
    if best_net <= 0:
        row = curve.loc[curve["threshold"] == NO_CAMPAIGN_THRESHOLD].iloc[0]
        return {**row.to_dict(), "profitable": False}
    tied = curve[np.isclose(curve["net_roi"], best_net, rtol=0, atol=1e-9)]
    row = tied.loc[tied["threshold"].idxmax()]
    return {**row.to_dict(), "profitable": True}


def ev_rule_mask(proba: np.ndarray, ltv: np.ndarray, params: RoiParams) -> np.ndarray:
    """Per-customer rule: target iff expected net > 0, i.e. p * acceptance * LTV > cost.

    This is the true optimum when LTV varies across customers (a pure probability
    threshold cannot express it) and is reported as an upper bound.
    """
    proba, ltv = np.asarray(proba, float), np.asarray(ltv, float)
    gain = params.offer_acceptance_rate * proba * ltv
    return gain > _cost_per_targeted(params)


def cost_benefit_matrix(
    proba: np.ndarray,
    churn: np.ndarray,
    ltv: np.ndarray,
    params: RoiParams,
    optimal_t: float,
    fixed_t: float = 0.50,
) -> pd.DataFrame:
    """Compare targeting strategies: revenue saved vs. campaign cost vs. net ROI."""
    proba = np.asarray(proba, dtype=float)
    strategies = {
        "No campaign": np.zeros(len(proba), dtype=bool),
        "Target everyone": np.ones(len(proba), dtype=bool),
        f"Fixed threshold {fixed_t:.2f}": proba >= fixed_t,
        f"Profit-optimal threshold {optimal_t:.3f}": proba >= optimal_t,
        "Per-customer EV rule (upper bound)": ev_rule_mask(proba, ltv, params),
    }
    rows = [
        {"strategy": name, **evaluate_policy(mask, churn, ltv, params)}
        for name, mask in strategies.items()
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Config + CLI
# --------------------------------------------------------------------------- #
def params_from_settings(settings: dict | None = None) -> tuple[RoiParams, LtvSettings]:
    cfg = (settings or load_settings())["roi"]
    return (
        RoiParams(
            retention_offer_cost=cfg["retention_offer_cost"],
            offer_acceptance_rate=cfg["offer_acceptance_rate"],
            cost_applies_to_all_targeted=cfg["cost_applies_to_all_targeted"],
        ),
        LtvSettings(
            source=cfg["ltv_source"],
            months_by_contract=cfg["expected_remaining_months"],
            margin=cfg.get("ltv_margin", 1.0),
        ),
    )


def analyse(scored: pd.DataFrame, params: RoiParams, ltv_cfg: LtvSettings) -> dict:
    """Pick the threshold on validation (expected profit), report realised profit on test."""
    ltv = compute_ltv(scored, ltv_cfg)
    val = scored["split"] == "validation"
    test = scored["split"] == "test"

    val_curve = profit_curve(
        scored.loc[val, "p_churn"].to_numpy(),
        scored.loc[val, "p_churn"].to_numpy(),
        ltv[val].to_numpy(),
        params,
    )
    best = optimal_threshold(val_curve)
    t_star = float(best["threshold"])

    test_args = (
        scored.loc[test, "p_churn"].to_numpy(),
        scored.loc[test, TARGET].to_numpy(),
        ltv[test].to_numpy(),
        params,
    )
    realised = cost_benefit_matrix(*test_args, optimal_t=t_star)
    expected = cost_benefit_matrix(
        scored.loc[test, "p_churn"].to_numpy(),
        scored.loc[test, "p_churn"].to_numpy(),
        ltv[test].to_numpy(),
        params,
        optimal_t=t_star,
    )
    test_curve_realised = profit_curve(*test_args)
    oracle = optimal_threshold(test_curve_realised)  # hindsight, for context only

    return {
        "params": asdict(params),
        "ltv": asdict(ltv_cfg),
        "mean_ltv": float(ltv.mean()),
        "threshold_selected_on": "validation (expected profit, calibrated probabilities)",
        "optimal_threshold": t_star,
        "campaign_is_profitable": bool(best["profitable"]),
        "validation_curve": val_curve,
        "test_curve_realised": test_curve_realised,
        "test_realised_matrix": realised,
        "test_expected_matrix": expected,
        "test_hindsight_best_threshold": float(oracle["threshold"]),
    }


def save_profit_figure(result: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from src.train import _GRID, _INK, _INK_2, _SERIES, _SURFACE, _style_axes

    t_star = result["optimal_threshold"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=_SURFACE)
    _style_axes(ax)
    grid = result["validation_curve"]
    real = result["test_curve_realised"]
    grid = grid[grid["threshold"] <= 1.0]
    real = real[real["threshold"] <= 1.0]
    ax.plot(
        grid["threshold"],
        grid["net_roi"],
        color=_SERIES["xgb_cal"],
        linewidth=2,
        label="Expected net (validation, calibrated p)",
    )
    ax.plot(
        real["threshold"],
        real["net_roi"],
        color=_SERIES["logreg"],
        linewidth=2,
        label="Realised net (held-out test)",
    )
    ax.axhline(0, color=_GRID, linewidth=1.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    ax.axvline(0.5, color=_INK_2, linewidth=1, linestyle="--")
    ax.text(0.505, ax.get_ylim()[1] * 0.92, "fixed 0.50", color=_INK_2, fontsize=8)
    if result["campaign_is_profitable"]:
        ax.axvline(t_star, color=_INK, linewidth=1, linestyle=":")
        ax.text(
            t_star + 0.005, ax.get_ylim()[1] * 0.80, f"optimal {t_star:.3f}", color=_INK, fontsize=8
        )
    ax.set_xlabel("Probability threshold (target customers with p >= threshold)")
    ax.set_ylabel("Net campaign profit (1,409 customers)")
    ax.set_title("Profit vs. targeting threshold", color=_INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=_INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _roi_comparison(realised: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy: expected (calibrated p) vs realised (actual churn) economics."""
    return pd.DataFrame(
        {
            "strategy": realised["strategy"],
            "targeted": realised["targeted"],
            "revenue saved": realised["revenue_saved"],
            "campaign cost": realised["campaign_cost"],
            "net (expected)": expected["net_roi"],
            "net (realised)": realised["net_roi"],
            "ROI % (realised)": realised["roi_pct"],
        }
    )


def main() -> None:
    args = cli_ui.base_parser(__doc__).parse_args()
    cli_ui.setup(args.quiet)
    cli_ui.banner("ROI", "profit-maximising threshold and cost-benefit matrix")
    settings = load_settings()
    params, ltv_cfg = params_from_settings(settings)
    scored = pd.read_parquet(resolve(settings["paths"]["scored_file"]))
    with cli_ui.spinner("Optimising threshold on validation, scoring on test"):
        result = analyse(scored, params, ltv_cfg)

    art = resolve(settings["paths"]["artifacts_dir"])
    figure_path = resolve(settings["paths"]["figures_dir"]) / "profit_curve.png"
    result["validation_curve"].to_csv(art / "roi_curve_validation.csv", index=False)
    result["test_curve_realised"].to_csv(art / "roi_curve_test.csv", index=False)
    result["test_realised_matrix"].to_csv(art / "roi_matrix_test_realised.csv", index=False)
    summary = {k: v for k, v in result.items() if not isinstance(v, pd.DataFrame)}
    summary["test_realised_matrix"] = result["test_realised_matrix"].to_dict(orient="records")
    summary["test_expected_matrix"] = result["test_expected_matrix"].to_dict(orient="records")
    (art / "roi_summary.json").write_text(json.dumps(summary, indent=2))
    save_profit_figure(result, figure_path)

    cli_ui.df_table(
        _roi_comparison(result["test_realised_matrix"], result["test_expected_matrix"]),
        "Cost-benefit matrix: expected vs realised (held-out test set)",
        {
            "revenue saved": cli_ui.money,
            "campaign cost": cli_ui.money,
            "net (expected)": cli_ui.money,
            "net (realised)": cli_ui.money,
            "ROI % (realised)": ".0f",
        },
        caption=(
            f"mean LTV ${result['mean_ltv']:,.0f} | optimal threshold "
            f"{result['optimal_threshold']:.3f} | profitable={result['campaign_is_profitable']}"
        ),
    )
    cli_ui.summary_panel(
        "ROI analysis complete",
        [
            art / "roi_summary.json",
            art / "roi_curve_validation.csv",
            art / "roi_curve_test.csv",
            art / "roi_matrix_test_realised.csv",
            figure_path,
        ],
    )


if __name__ == "__main__":
    main()
