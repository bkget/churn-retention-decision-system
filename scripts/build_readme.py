"""Render README.md from docs/README.template.md using the artifacts `make train` produced.

Every number in the README comes from artifacts/metrics.json, explain_summary.json,
roi_summary.json or a recomputation over scored_customers.parquet - none is typed by hand.

Usage:  python scripts/build_readme.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_settings, resolve  # noqa: E402
from src.retention import template_offer  # noqa: E402
from src.roi import RoiParams, analyse, compute_ltv, params_from_settings  # noqa: E402


def money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def kmoney(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x) / 1000:,.1f}k"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def model_table(metrics: dict) -> str:
    rows = []
    for key in ("logreg", "xgb_raw", "xgb_cal"):
        m = metrics["models"][key]
        a, b = m["test_at_0.50"], m["test_at_val_f1_threshold"]
        name = m["label"]
        if key == "xgb_cal":
            name = f"**{name}**"
        rows.append(
            [
                name,
                f"{a['pr_auc']:.3f}",
                f"{a['roc_auc']:.3f}",
                f"{a['brier']:.3f}",
                f"{a['recall']:.3f}",
                f"{a['f1']:.3f}",
                f"{b['recall']:.3f}",
                f"{b['f1']:.3f}",
            ]
        )
    return md_table(
        ["Model", "PR-AUC", "ROC-AUC", "Brier", "Recall @0.50", "F1 @0.50",
         "Recall @tuned thr", "F1 @tuned thr"],
        rows,
    )  # fmt: skip


def roi_table(matrix: list[dict]) -> str:
    rows = []
    for r in matrix:
        name = r["strategy"]
        if name.startswith("Profit-optimal"):
            name = f"**{name}**"
        roi = "–" if r["roi_pct"] != r["roi_pct"] else f"{r['roi_pct']:,.0f}%"
        rows.append(
            [
                name,
                f"{int(r['targeted']):,}",
                f"{r['churners_reached']:,.0f}",
                money(r["revenue_saved"]),
                money(r["campaign_cost"]),
                money(r["net_roi"]),
                roi,
            ]
        )
    return md_table(
        ["Strategy", "Targeted", "Churners reached", "Revenue saved", "Campaign cost",
         "Net ROI", "ROI %"],
        rows,
    )  # fmt: skip


def sensitivity_table(scored: pd.DataFrame, ltv_cfg, costs, accs) -> str:
    rows = []
    for cost in costs:
        row = [f"${cost}"]
        for acc in accs:
            r = analyse(scored, RoiParams(float(cost), acc, True), ltv_cfg)
            m = r["test_realised_matrix"].set_index("strategy")
            fixed = m.loc["Fixed threshold 0.50", "net_roi"]
            if not r["campaign_is_profitable"]:
                row.append(f"no campaign (fixed 0.50: {kmoney(fixed)})")
            else:
                opt = m[m.index.str.startswith("Profit-optimal")].iloc[0]["net_roi"]
                row.append(
                    f"t={r['optimal_threshold']:.2f}: {kmoney(opt)} (fixed 0.50: {kmoney(fixed)})"
                )
        rows.append(row)
    return md_table(["Offer cost \\ acceptance", *[f"{a:.0%}" for a in accs]], rows)


def agreement_table(explain: dict) -> str:
    rows = [
        [
            r["stated_category"],
            f"{r['churners']:,}",
            f"{r['hit_rate']:.2f}",
            f"{r['baseline_rate']:.2f}",
            f"{r['lift']:.2f}",
        ]
        for r in explain["driver_vs_stated_reason"]
    ]
    return md_table(
        ["Stated reason", "Churners", "Hit rate", "Baseline (other churners)", "Lift"], rows
    )


def importance_table(explain: dict, n: int = 6) -> str:
    rows = [
        [f"`{r['feature']}`", f"{r['mean_abs_shap']:.3f}", f"{r['share']:.1%}"]
        for r in explain["global_importance"][:n]
    ]
    return md_table(["Feature", "Mean abs SHAP", "Share of total"], rows)


def example_block(scored: pd.DataFrame, params: RoiParams, ltv_cfg) -> str:
    active = scored[scored["churn"] == 0].sort_values("p_churn", ascending=False)
    row = active.iloc[0]
    ltv = float(compute_ltv(active.iloc[[0]], ltv_cfg).iloc[0])
    drivers = [
        {"feature": row[f"driver_{k}_feature"], "text": row[f"driver_{k}_text"]}
        for k in (1, 2, 3)
        if isinstance(row[f"driver_{k}_feature"], str)
    ]
    offer = template_offer(
        row.to_dict(), drivers, params.retention_offer_cost, {"p_churn": row["p_churn"], "ltv": ltv}
    )
    lines = [
        f"Highest-risk current subscriber (`{row['customer_id']}`, "
        f"{row['p_churn']:.0%} churn probability, LTV {money(ltv)}):",
        "",
        *[f"{i}. {d['text']}" for i, d in enumerate(drivers, 1)],
        "",
        f"Template-engine offer within a {money(params.retention_offer_cost)} budget "
        f"(est. cost {money(offer.total_cost)}): **{offer.headline}**",
        "",
        f"> {offer.customer_message}",
    ]
    return "\n".join(lines)


def build() -> str:
    settings = load_settings()
    art = resolve(settings["paths"]["artifacts_dir"])
    metrics = json.loads((art / "metrics.json").read_text())
    explain = json.loads((art / "explain_summary.json").read_text())
    roi = json.loads((art / "roi_summary.json").read_text())
    scored = pd.read_parquet(art / "scored_customers.parquet")
    params, ltv_cfg = params_from_settings(settings)

    lr = metrics["models"]["logreg"]["test_at_0.50"]
    xg = metrics["models"]["xgb_cal"]["test_at_0.50"]
    xg_thr = metrics["models"]["xgb_cal"]["test_at_val_f1_threshold"]
    lr_thr = metrics["models"]["logreg"]["test_at_val_f1_threshold"]
    boot = metrics["xgb_cal_minus_logreg_test_bootstrap"]
    pr, rc = boot["pr_auc"], boot["roc_auc"]

    def ci_text(b: dict) -> str:
        return f"{b['mean_diff']:+.3f} (95% CI {b['ci_low']:+.3f} to {b['ci_high']:+.3f})"

    pr_sig = pr["ci_low"] > 0
    roc_sig = rc["ci_low"] > 0
    significance = (
        "The PR-AUC gain is not distinguishable from noise"
        if not pr_sig
        else "The PR-AUC gain is statistically distinguishable from zero"
    )
    significance += (
        ", and the ROC-AUC gain is small but its interval excludes zero."
        if roc_sig
        else ", and neither is the ROC-AUC gain."
    )
    if not pr_sig and not roc_sig:
        significance = "Neither the PR-AUC nor the ROC-AUC gain is distinguishable from noise."

    realised = {r["strategy"]: r for r in roi["test_realised_matrix"]}
    expected = {r["strategy"]: r for r in roi["test_expected_matrix"]}
    opt_key = next(k for k in realised if k.startswith("Profit-optimal"))
    fixed_key = "Fixed threshold 0.50"
    everyone_key = "Target everyone"
    opt, fixed, every = realised[opt_key], realised[fixed_key], realised[everyone_key]
    opt_exp = expected[opt_key]
    uplift = opt["net_roi"] - fixed["net_roi"]

    # segment concentration
    seg = scored.groupby(["contract", "internet_type"]).churn.agg(["mean", "size"])
    worst = seg[seg["size"] >= 100]["mean"].idxmax()
    worst_rate, worst_n = seg.loc[worst, "mean"], int(seg.loc[worst, "size"])
    two_year = scored[scored["contract"] == "Two Year"]["churn"].mean()

    corr = float(np.corrcoef(scored["tenure"], scored["number_of_referrals"])[0, 1])
    imp = pd.read_csv(art / "global_importance.csv")
    ref_rank = int(imp.index[imp["feature"] == "number_of_referrals"][0]) + 1
    ref_share = float(imp.loc[imp["feature"] == "number_of_referrals", "share"].iloc[0])

    rows = metrics["rows"]
    values = {
        "n_customers": f"{rows['total']:,}",
        "churn_rate": f"{metrics['churn_rate']['overall']:.1%}",
        "n_features": str(metrics["n_features"]),
        "n_train": f"{rows['train']:,}",
        "n_val": f"{rows['validation']:,}",
        "n_test": f"{rows['test']:,}",
        "worst_segment": f"{worst[0]} + {worst[1]}",
        "worst_rate": f"{worst_rate:.0%}",
        "worst_n": f"{worst_n:,}",
        "two_year_rate": f"{two_year:.1%}",
        "model_table": model_table(metrics),
        "xgb_pr": f"{xg['pr_auc']:.3f}",
        "lr_pr": f"{lr['pr_auc']:.3f}",
        "xgb_roc": f"{xg['roc_auc']:.3f}",
        "lr_roc": f"{lr['roc_auc']:.3f}",
        "xgb_brier": f"{xg['brier']:.3f}",
        "lr_brier": f"{lr['brier']:.3f}",
        "xgb_recall_thr": f"{xg_thr['recall']:.2f}",
        "lr_recall_thr": f"{lr_thr['recall']:.2f}",
        "xgb_recall_050": f"{xg['recall']:.2f}",
        "lr_recall_050": f"{lr['recall']:.2f}",
        "pr_diff": ci_text(pr),
        "roc_diff": ci_text(rc),
        "significance": significance,
        "trials": str(metrics["optuna"]["trials"]),
        "cal_method": metrics["calibration_method"],
        "lift10": f"{xg['top_decile_lift']:.1f}",
        "importance_table": importance_table(explain),
        "demo_share": f"{explain['demographic_share_of_total_shap']:.0%}",
        "demo_list": ", ".join(f"`{f}`" for f in explain["demographic_features_hidden_from_text"]),
        "high_risk_n": f"{explain['high_risk_customers']:,}",
        "agreement_table": agreement_table(explain),
        "example_block": example_block(scored, params, ltv_cfg),
        "offer_cost": money(params.retention_offer_cost),
        "acceptance": f"{params.offer_acceptance_rate:.0%}",
        "mean_ltv": money(roi["mean_ltv"]),
        "months": ", ".join(f"{k} {v}" for k, v in roi["ltv"]["months_by_contract"].items()),
        "opt_t": f"{roi['optimal_threshold']:.3f}",
        "roi_table": roi_table(roi["test_realised_matrix"]),
        "opt_net": money(opt["net_roi"]),
        "fixed_net": money(fixed["net_roi"]),
        "every_net": money(every["net_roi"]),
        "uplift": money(uplift),
        "uplift_pct": f"{uplift / fixed['net_roi']:.0%}",
        "opt_targeted": f"{int(opt['targeted']):,}",
        "fixed_targeted": f"{int(fixed['targeted']):,}",
        "pred_net": money(opt_exp["net_roi"]),
        "sensitivity_table": sensitivity_table(
            scored, ltv_cfg, costs=[25, 50, 100, 200], accs=[0.1, 0.3, 0.5]
        ),
        "ref_corr": f"{corr:.2f}",
        "ref_rank": str(ref_rank),
        "ref_share": f"{ref_share:.0%}",
        "xgb_version": metrics["versions"]["xgboost"],
        "sklearn_version": metrics["versions"]["sklearn"],
        "generated_at": metrics["generated_at"][:10],
    }

    text = (ROOT / "docs" / "README.template.md").read_text()
    missing = set(re.findall(r"\{\{(\w+)\}\}", text)) - set(values)
    if missing:
        raise KeyError(f"template placeholders without a value: {sorted(missing)}")
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", v)
    return text


def main() -> None:
    (ROOT / "README.md").write_text(build())
    print("README.md written")


if __name__ == "__main__":
    main()
