"""Executive Overview tab: churn KPIs, segment risk heatmap, global SHAP chart."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_views.common import BLUE, BLUE_RAMP, MUTED, money, pct
from src.explain import DEMOGRAPHIC_FEATURES
from src.roi import compute_ltv, params_from_settings

CONTRACT_ORDER = ["Month-to-Month", "One Year", "Two Year"]
INTERNET_ORDER = ["Fiber Optic", "DSL", "Cable", "None"]


def segment_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Contract x internet type: customers, actual churn rate, mean predicted risk."""
    grouped = (
        scored.groupby(["contract", "internet_type"], observed=True)
        .agg(customers=("churn", "size"), actual=("churn", "mean"), predicted=("p_churn", "mean"))
        .reset_index()
    )
    return grouped


def heatmap(seg: pd.DataFrame, measure: str) -> go.Figure:
    rows = [c for c in CONTRACT_ORDER if c in set(seg["contract"])]
    cols = [c for c in INTERNET_ORDER if c in set(seg["internet_type"])]
    z, text, hover = [], [], []
    for contract in rows:
        z_row, t_row, h_row = [], [], []
        for net in cols:
            cell = seg[(seg["contract"] == contract) & (seg["internet_type"] == net)]
            if cell.empty:
                z_row.append(None)
                t_row.append("")
                h_row.append("No customers")
                continue
            r = cell.iloc[0]
            v = float(r[measure])
            z_row.append(v)
            t_row.append(f"{v * 100:.0f}%  (n={int(r['customers']):,})")
            h_row.append(
                f"<b>{contract} / {net}</b><br>Customers: {int(r['customers']):,}"
                f"<br>Actual churn: {r['actual'] * 100:.1f}%"
                f"<br>Mean predicted risk: {r['predicted'] * 100:.1f}%"
            )
        z.append(z_row)
        text.append(t_row)
        hover.append(h_row)
    scale = [[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=cols,
            y=rows,
            text=text,
            texttemplate="%{text}",
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            colorscale=scale,
            zmin=0,
            zmax=max(0.01, max(v for r in z for v in r if v is not None)),
            xgap=2,
            ygap=2,
            colorbar={"title": "", "tickformat": ".0%", "thickness": 12},
        )
    )
    fig.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"title": "Internet service", "side": "bottom"},
        yaxis={"autorange": "reversed", "title": ""},
    )
    return fig


def shap_bar(importance: pd.DataFrame, top_n: int = 12) -> go.Figure:
    top = importance.head(top_n).iloc[::-1]
    shown = ~top["feature"].isin(DEMOGRAPHIC_FEATURES)
    fig = go.Figure()
    for mask, name, color in (
        (shown, "Shown to reps as drivers", BLUE),
        (~shown, "In model, hidden from rep-facing drivers", MUTED),
    ):
        sub = top[mask]
        if sub.empty:
            continue
        fig.add_bar(
            x=sub["mean_abs_shap"],
            y=sub["feature"],
            orientation="h",
            name=name,
            marker_color=color,
            hovertemplate="<b>%{y}</b><br>mean |SHAP|: %{x:.3f}<extra></extra>",
        )
    fig.update_layout(
        height=440,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        xaxis_title="Mean |SHAP| (log-odds impact on churn)",
        yaxis_title="",
        barmode="overlay",
        bargap=0.35,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


def render(data: dict, settings: dict) -> None:
    scored, metrics = data["scored"], data["metrics"]
    params, ltv_cfg = params_from_settings(settings)
    active = scored[scored["churn"] == 0]
    ltv_active = compute_ltv(active, ltv_cfg)
    revenue_at_risk = float((active["p_churn"] * ltv_active).sum())
    high_risk_active = int((active["p_churn"] >= 0.5).sum())
    final = metrics["models"]["xgb_cal"]["test_at_0.50"]

    st.subheader("Where is churn coming from, and how much is at stake?")
    k = st.columns(5)
    k[0].metric("Customers", f"{len(scored):,}")
    k[1].metric("Churn rate", pct(scored["churn"].mean()), help="Share who left in the last month.")
    k[2].metric(
        "Active, high-risk (p ≥ 0.50)",
        f"{high_risk_active:,}",
        help=f"{pct(high_risk_active / max(len(active), 1))} of the "
        f"{len(active):,} active customers.",
    )
    k[3].metric(
        "Revenue at risk",
        money(revenue_at_risk),
        help="Active customers: sum of churn probability x LTV "
        f"(LTV = monthly charges x remaining months, source: {ltv_cfg.source}).",
    )
    k[4].metric(
        "Model PR-AUC (test)",
        f"{final['pr_auc']:.3f}",
        help=f"ROC-AUC {final['roc_auc']:.3f}; baseline logistic regression PR-AUC "
        f"{metrics['models']['logreg']['test_at_0.50']['pr_auc']:.3f}.",
    )

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Segment risk: contract × internet service**")
        measure = st.radio(
            "Colour by",
            ["actual", "predicted"],
            format_func=lambda m: "Actual churn rate" if m == "actual" else "Mean predicted risk",
            horizontal=True,
            label_visibility="collapsed",
            key="heatmap_measure",
        )
        st.plotly_chart(heatmap(segment_table(scored), measure))
        st.caption(
            "Month-to-month contracts, especially on fiber, are where churn concentrates. "
            "Cell label = rate and customer count; hover for both actual and predicted."
        )
    with right:
        st.markdown("**What drives churn overall (global SHAP)**")
        st.plotly_chart(shap_bar(data["importance"]))
        st.caption(
            "Grey bars are features the model uses but that are never shown to a rep as the "
            "reason for an offer (actionability / responsible-AI choice)."
        )

    with st.expander("Model scorecard (held-out test set)"):
        rows = []
        for key in ("logreg", "xgb_cal"):
            m = metrics["models"][key]
            a, b = m["test_at_0.50"], m["test_at_val_f1_threshold"]
            rows.append(
                {
                    "Model": m["label"],
                    "PR-AUC": a["pr_auc"],
                    "ROC-AUC": a["roc_auc"],
                    "Brier": a["brier"],
                    "Recall @0.50": a["recall"],
                    "F1 @0.50": a["f1"],
                    "Recall @ tuned thr": b["recall"],
                    "F1 @ tuned thr": b["f1"],
                }
            )
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.3f}"))
        st.caption(
            "Brier (calibration) is where the tuned model clearly wins; PR-AUC / ROC-AUC gaps "
            "are within bootstrap noise. See README."
        )
