"""SHAP explanations: global importance, per-customer drivers, plain-English text.

SHAP values are computed on the tuned (uncalibrated) XGBoost with ``TreeExplainer``
and are in log-odds. The calibrator is a monotonic sigmoid on top of the same
margin, so driver ranking is identical for the calibrated probability.

One-hot columns are summed back to their original feature, so a driver reads
"Contract", never "contract_Month-to-Month".

Usage:  python -m src.explain        (after ``python -m src.train``)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from src import cli_ui
from src.config import load_settings, resolve
from src.data import ID_COL, TARGET

log = logging.getLogger(__name__)

N_DRIVERS = 3

# Kept in the model, but never shown to a Customer Success rep as the "reason"
# for a retention offer: demographics / household composition are neither
# actionable nor appropriate to act on.
DEMOGRAPHIC_FEATURES = {
    "gender",
    "age",
    "senior_citizen",
    "under_30",
    "partner",
    "married",
    "dependents",
    "number_of_dependents",
}


# --------------------------------------------------------------------------- #
# SHAP computation
# --------------------------------------------------------------------------- #
def original_feature_map(transformed_names: list[str], categorical: list[str]) -> dict[str, str]:
    """Map each transformed (post one-hot) column name to its original feature."""
    mapping = {}
    for name in transformed_names:
        origin = name
        for cat in categorical:
            if name.startswith(f"{cat}_"):
                origin = cat
                break
        mapping[name] = origin
    return mapping


def aggregate_shap(
    shap_values: np.ndarray, transformed_names: list[str], categorical: list[str]
) -> pd.DataFrame:
    """Sum SHAP values of one-hot columns back into their original feature."""
    mapping = original_feature_map(transformed_names, categorical)
    frame = pd.DataFrame(shap_values, columns=transformed_names)
    return frame.T.groupby(mapping).sum().T


def compute_shap(
    pipeline, X: pd.DataFrame, categorical: list[str]
) -> tuple[pd.DataFrame, float, np.ndarray]:
    """Return (SHAP by original feature, base value, raw log-odds margin)."""
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    X_t = prep.transform(X)
    names = list(prep.get_feature_names_out())
    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(X_t)
    base = float(np.ravel(explainer.expected_value)[0])
    margin = clf.predict(X_t, output_margin=True)
    agg = aggregate_shap(values, names, categorical)
    agg.index = X.index
    return agg, base, margin


# --------------------------------------------------------------------------- #
# Plain-English driver text
# --------------------------------------------------------------------------- #
def _money(x: float) -> str:
    return f"${x:,.0f}"


def _months(v: int) -> str:
    return f"{v} month" if v == 1 else f"{v} months"


def _tenure(v: float, _row: pd.Series) -> str:
    v = int(v)
    if v <= 6:
        return f"Brand-new customer ({_months(v)} of tenure)"
    if v <= 12:
        return f"First-year customer ({_months(v)} of tenure)"
    if v <= 24:
        return f"Still early in the relationship ({_months(v)} of tenure)"
    return f"{_months(v)} of tenure"


def _monthly_charges(v: float, _row: pd.Series) -> str:
    return f"High monthly bill ({_money(v)})" if v >= 80 else f"Monthly bill of {_money(v)}"


def _total_charges(v: float, _row: pd.Series) -> str:
    return (
        f"Low lifetime spend so far ({_money(v)})" if v < 1000 else f"Lifetime spend of {_money(v)}"
    )


def _gap(v: float, _row: pd.Series) -> str:
    if v > 1:
        return f"Now paying {_money(v)}/month more than their historical average"
    if v < -1:
        return f"Now paying {_money(-v)}/month less than their historical average"
    return "Current bill in line with their historical average"


def _service_count(v: float, _row: pd.Series) -> str:
    return f"Only {int(v)} of 9 services" if v <= 3 else f"{int(v)} of 9 services"


def _no_yes(no: str, yes: str):
    return lambda v, _row: yes if int(v) == 1 else no


_FEATURE_TEXT = {
    "contract": lambda v, _r: {
        "Month-to-Month": "Month-to-month contract (no lock-in)",
        "One Year": "One-year contract, not yet on a 2-year commitment",
        "Two Year": "Two-year contract",
    }.get(str(v), f"{v} contract"),
    "internet_type": lambda v, _r: (
        "No internet service" if v == "None" else f"{v} internet service"
    ),
    "payment_method": lambda v, _r: f"Pays by {str(v).lower()}",
    "tenure": _tenure,
    "monthly_charges": _monthly_charges,
    "total_charges": _total_charges,
    "avg_monthly_spend": lambda v, _r: f"Average spend of {_money(v)}/month",
    "charge_vs_tenure_gap": _gap,
    "service_count": _service_count,
    "is_new_fiber_m2m": _no_yes(
        "Not a new month-to-month fiber customer", "New month-to-month fiber customer"
    ),
    "online_security": _no_yes("No online security add-on", "Has online security"),
    "online_backup": _no_yes("No online backup add-on", "Has online backup"),
    "device_protection": _no_yes("No device protection plan", "Has device protection"),
    "tech_support": _no_yes("No premium tech support", "Has premium tech support"),
    "streaming_tv": _no_yes("Does not stream TV", "Streams TV"),
    "streaming_movies": _no_yes("Does not stream movies", "Streams movies"),
    "streaming_music": _no_yes("Does not stream music", "Streams music"),
    "unlimited_data": _no_yes("No unlimited-data plan", "On an unlimited-data plan"),
    "paperless_billing": _no_yes("Receives paper bills", "Paperless billing"),
    "phone_service": _no_yes("No phone service", "Has phone service"),
    "multiple_lines": _no_yes("Single phone line", "Multiple phone lines"),
    "internet_service": _no_yes("No internet service", "Has internet service"),
    "partner": _no_yes("No partner on the account", "Has a partner on the account"),
    "married": _no_yes("Not married", "Married"),
    "dependents": _no_yes("No dependents", "Has dependents"),
    "number_of_dependents": lambda v, _r: f"{int(v)} dependents",
    "referred_a_friend": _no_yes("Has not referred a friend", "Has referred a friend"),
    "number_of_referrals": lambda v, _r: (
        "Has made no referrals"
        if int(v) == 0
        else f"Has made few referrals ({int(v)})"
        if int(v) <= 2
        else f"Has made {int(v)} referrals"
    ),
    "avg_monthly_gb_download": lambda v, _r: f"Downloads {v:.0f} GB/month on average",
    "avg_monthly_long_distance_charges": lambda v, _r: (
        f"Long-distance charges average {_money(v)}/month"
    ),
    "total_long_distance_charges": lambda v, _r: f"{_money(v)} in long-distance charges to date",
    "total_extra_data_charges": lambda v, _r: (
        "Has paid extra data charges" if v > 0 else "No extra data charges"
    ),
    "total_refunds": lambda v, _r: (
        f"Received {_money(v)} in refunds" if v > 0 else "No refunds received"
    ),
    "gender": lambda v, _r: f"Gender: {v}",
    "age": lambda v, _r: f"Age {int(v)}",
    "senior_citizen": _no_yes("Not a senior citizen", "Senior citizen"),
    "under_30": _no_yes("Age 30 or over", "Under 30"),
}


def driver_text(feature: str, row: pd.Series) -> str:
    """Plain-English description of a customer's state on ``feature``."""
    fn = _FEATURE_TEXT.get(feature)
    value = row[feature]
    if fn is None:
        return f"{feature.replace('_', ' ').capitalize()}: {value}"
    return fn(value, row)


def top_risk_drivers(
    shap_row: pd.Series, n: int = N_DRIVERS, exclude: set[str] | None = None
) -> list[str]:
    """Features that push churn risk UP the most (positive SHAP), strongest first."""
    exclude = DEMOGRAPHIC_FEATURES if exclude is None else exclude
    candidates = shap_row.drop(labels=[f for f in exclude if f in shap_row.index])
    positive = candidates[candidates > 0].sort_values(ascending=False)
    return positive.index[:n].tolist()


def build_driver_table(
    shap_df: pd.DataFrame, features_df: pd.DataFrame, n: int = N_DRIVERS
) -> pd.DataFrame:
    """One row per customer: top-n driver feature names, SHAP values and text."""
    rows = []
    for idx in shap_df.index:
        s = shap_df.loc[idx]
        feats = top_risk_drivers(s, n)
        rec: dict[str, Any] = {"_idx": idx}
        for k in range(n):
            if k < len(feats):
                f = feats[k]
                rec[f"driver_{k + 1}_feature"] = f
                rec[f"driver_{k + 1}_shap"] = float(s[f])
                rec[f"driver_{k + 1}_text"] = driver_text(f, features_df.loc[idx])
            else:
                rec[f"driver_{k + 1}_feature"] = None
                rec[f"driver_{k + 1}_shap"] = np.nan
                rec[f"driver_{k + 1}_text"] = None
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("_idx")
    out.index.name = None
    return out


# --------------------------------------------------------------------------- #
# Driver vs. stated churn reason
# --------------------------------------------------------------------------- #
# Which model features a customer's *stated* reason would plausibly touch.
# This mapping is a judgement call; it is documented in the README.
STATED_CATEGORY_FEATURES: dict[str, set[str]] = {
    "Price": {
        "monthly_charges",
        "avg_monthly_spend",
        "charge_vs_tenure_gap",
        "total_extra_data_charges",
        "total_long_distance_charges",
        "avg_monthly_long_distance_charges",
    },
    "Competitor": {
        "contract",
        "is_new_fiber_m2m",
        "internet_type",
        "tenure",
        "avg_monthly_gb_download",
        "unlimited_data",
    },
    "Attitude": {"tech_support", "online_security", "online_backup", "device_protection"},
    "Dissatisfaction": {
        "tech_support",
        "online_security",
        "online_backup",
        "device_protection",
        "internet_type",
        "service_count",
    },
}


def driver_agreement(churners: pd.DataFrame, n: int = N_DRIVERS) -> pd.DataFrame:
    """Does the model's top-n driver set hit the features the customer's stated reason points to?

    ``hit_rate``      share of churners with that stated category whose top-n drivers include
                      at least one related feature.
    ``baseline_rate`` same hit test, but among churners with a *different* stated category.
                      If hit_rate is not above it, the model is not reflecting the stated reason.
    """
    feat_cols = [f"driver_{k + 1}_feature" for k in range(n)]
    drivers = churners[feat_cols].apply(lambda r: set(r.dropna()), axis=1)
    rows = []
    for category, related in STATED_CATEGORY_FEATURES.items():
        in_cat = churners["churn_category"] == category
        hit = drivers.map(lambda d, rel=related: bool(d & rel))
        rows.append(
            {
                "stated_category": category,
                "churners": int(in_cat.sum()),
                "related_features": ", ".join(sorted(related)),
                "hit_rate": float(hit[in_cat].mean()),
                "baseline_rate": float(hit[~in_cat].mean()),
                "lift": float(hit[in_cat].mean() / max(hit[~in_cat].mean(), 1e-9)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    settings = load_settings()
    models_dir = resolve(settings["paths"]["models_dir"])
    cli_ui.banner("Explain", "global + per-customer SHAP, plain-English drivers")
    with cli_ui.spinner("Loading model and scored customers"):
        pipeline = joblib.load(models_dir / "xgb.joblib")
        spec = joblib.load(models_dir / "feature_spec.joblib")
        features, categorical = spec["features"], spec["groups"]["categorical"]
        scored_path = resolve(settings["paths"]["scored_file"])
        scored = pd.read_parquet(scored_path)

    with cli_ui.spinner("Computing SHAP values (TreeExplainer)"):
        shap_df, base, margin = compute_shap(pipeline, scored[features], categorical)

    # Additivity check: base + sum(SHAP) must reproduce the model's log-odds.
    err = float(np.abs(base + shap_df.sum(axis=1).to_numpy() - margin).max())
    if err > 1e-3:
        raise RuntimeError(f"SHAP additivity violated (max error {err:.2e})")

    with cli_ui.spinner("Building top-3 plain-English drivers"):
        drivers = build_driver_table(shap_df, scored[features])
    drop = [c for c in scored.columns if c.startswith("driver_")]
    scored = scored.drop(columns=drop).join(drivers)
    scored.to_parquet(scored_path, index=False)

    out_shap = shap_df.copy()
    out_shap.insert(0, ID_COL, scored[ID_COL].to_numpy())
    out_shap.to_parquet(resolve(settings["paths"]["artifacts_dir"]) / "shap_values.parquet")

    importance = (
        shap_df.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").reset_index()
    )
    importance.columns = ["feature", "mean_abs_shap"]
    importance["share"] = importance["mean_abs_shap"] / importance["mean_abs_shap"].sum()

    churners = scored[scored[TARGET] == 1]
    agreement = driver_agreement(churners)

    high = scored[scored["p_churn"] >= 0.5]
    summary = {
        "shap_base_value_logodds": base,
        "max_additivity_error": err,
        "global_importance": importance.head(15).to_dict(orient="records"),
        "high_risk_threshold": 0.5,
        "high_risk_customers": int(len(high)),
        "high_risk_with_3_drivers": int(high["driver_3_feature"].notna().sum()),
        "customers_with_3_drivers": int(scored["driver_3_feature"].notna().sum()),
        "demographic_features_hidden_from_text": sorted(DEMOGRAPHIC_FEATURES),
        "demographic_share_of_total_shap": float(
            importance.loc[importance["feature"].isin(DEMOGRAPHIC_FEATURES), "share"].sum()
        ),
        "driver_vs_stated_reason": agreement.to_dict(orient="records"),
    }
    art = resolve(settings["paths"]["artifacts_dir"])
    (art / "explain_summary.json").write_text(json.dumps(summary, indent=2))
    importance.to_csv(art / "global_importance.csv", index=False)
    agreement.to_csv(art / "driver_vs_stated_reason.csv", index=False)

    top = importance.head(12).copy()
    top["rep_facing"] = np.where(top["feature"].isin(DEMOGRAPHIC_FEATURES), "no (hidden)", "yes")
    cli_ui.df_table(
        top,
        "Top SHAP drivers (global)",
        {"mean_abs_shap": ".3f", "share": ".1%"},
        caption="mean |SHAP| in log-odds; hidden = used by the model, never shown to reps",
    )
    cli_ui.df_table(
        agreement.drop(columns="related_features"), "Model drivers vs stated churn reason"
    )
    log.info(
        "high-risk with 3 drivers: %d/%d; additivity error %.1e",
        summary["high_risk_with_3_drivers"],
        summary["high_risk_customers"],
        err,
    )
    cli_ui.summary_panel(
        "Explanations complete",
        [
            scored_path,
            art / "shap_values.parquet",
            art / "explain_summary.json",
            art / "global_importance.csv",
            art / "driver_vs_stated_reason.csv",
        ],
    )
    return summary


def main() -> None:
    args = cli_ui.base_parser(__doc__).parse_args()
    cli_ui.setup(args.quiet)
    run()


if __name__ == "__main__":
    main()
