"""ROI Simulator tab: sliders -> profit-optimal threshold, targets, net dollars saved."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_views.common import BLUE, MUTED, ORANGE, money, pct, threshold_label
from src.roi import (
    LtvSettings,
    RoiParams,
    compute_ltv,
    cost_benefit_matrix,
    evaluate_policy,
    optimal_threshold,
    params_from_settings,
    profit_curve,
)

SCOPE_OOS = "Out-of-sample customers (validation + test)"
SCOPE_ALL = "Entire book (includes training rows)"


@dataclass(frozen=True)
class RoiState:
    """What the other tabs need from the simulator."""

    params: RoiParams
    ltv_cfg: LtvSettings
    threshold: float
    profitable: bool


def _matrix_display(matrix: pd.DataFrame) -> pd.DataFrame:
    out = matrix.rename(
        columns={
            "strategy": "Strategy",
            "targeted": "Customers targeted",
            "churners_reached": "Churners reached",
            "revenue_saved": "Revenue saved",
            "campaign_cost": "Campaign cost",
            "net_roi": "Net ROI ($)",
            "roi_pct": "ROI %",
        }
    )
    return out.set_index("Strategy")


def _style(matrix: pd.DataFrame):
    return _matrix_display(matrix).style.format(
        {
            "Customers targeted": "{:,.0f}",
            "Churners reached": "{:,.0f}",
            "Revenue saved": "${:,.0f}",
            "Campaign cost": "${:,.0f}",
            "Net ROI ($)": "${:,.0f}",
            "ROI %": "{:,.0f}%",
        },
        na_rep="–",
    )


def profit_chart(curve: pd.DataFrame, best: dict, profitable: bool) -> go.Figure:
    plot = curve[curve["threshold"] <= 1.0]
    fig = go.Figure()
    fig.add_scatter(
        x=plot["threshold"],
        y=plot["net_roi"],
        mode="lines",
        name="Expected net profit",
        line={"color": BLUE, "width": 2},
        hovertemplate="threshold %{x:.2f}<br>net %{y:$,.0f}<extra></extra>",
    )
    fig.add_vline(x=0.5, line={"color": MUTED, "dash": "dash", "width": 1})
    fig.add_annotation(
        x=0.5,
        y=1,
        yref="paper",
        text="fixed 0.50",
        showarrow=False,
        xanchor="left",
        font={"size": 11},
    )
    if profitable:
        fig.add_scatter(
            x=[best["threshold"]],
            y=[best["net_roi"]],
            mode="markers",
            name="Profit-optimal threshold",
            marker={"color": ORANGE, "size": 11, "line": {"color": "white", "width": 2}},
            hovertemplate="optimal %{x:.3f}<br>net %{y:$,.0f}<extra></extra>",
        )
    fig.add_hline(y=0, line={"color": MUTED, "width": 1})
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title="Probability threshold (target customers with p ≥ threshold)",
        yaxis_title="Net campaign profit",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return fig


def render(data: dict, settings: dict) -> RoiState:
    scored = data["scored"]
    cfg_params, cfg_ltv = params_from_settings(settings)

    st.subheader("What should the retention campaign target, and what does it return?")
    c1, c2 = st.columns(2)
    offer_cost = c1.slider(
        "Retention offer cost per targeted customer ($)",
        0,
        300,
        int(cfg_params.retention_offer_cost),
        step=5,
        key="offer_cost",
    )
    acceptance = c2.slider(
        "Offer acceptance rate (success rate)",
        0,
        100,
        int(round(cfg_params.offer_acceptance_rate * 100)),
        step=1,
        format="%d%%",
        key="acceptance",
        help="Share of targeted would-be churners who accept the offer and stay.",
    )

    with st.expander("Advanced assumptions (LTV, cost model, population)"):
        a1, a2 = st.columns(2)
        ltv_source = a1.radio(
            "Customer LTV",
            ["formula", "cltv"],
            index=0 if cfg_ltv.source == "formula" else 1,
            format_func=lambda s: (
                "MonthlyCharges × expected remaining months"
                if s == "formula"
                else "IBM CLTV column (predicted lifetime value)"
            ),
            key="ltv_source",
        )
        margin = a2.slider(
            "LTV margin (1.0 = revenue, <1.0 = contribution margin)",
            0.1,
            1.0,
            float(cfg_ltv.margin),
            step=0.05,
            key="ltv_margin",
        )
        cols = st.columns(3)
        months = {}
        for col, contract in zip(cols, ["Month-to-Month", "One Year", "Two Year"], strict=True):
            months[contract] = col.number_input(
                f"Remaining months: {contract}",
                min_value=1,
                max_value=120,
                value=int(cfg_ltv.months_by_contract[contract]),
                key=f"months_{contract}",
                disabled=ltv_source == "cltv",
            )
        cost_all = st.checkbox(
            "Pay the offer cost for every targeted customer, including false positives "
            "(conservative)",
            value=cfg_params.cost_applies_to_all_targeted,
            key="cost_all",
        )
        scope = st.radio(
            "Customers to simulate on",
            [SCOPE_OOS, SCOPE_ALL],
            key="scope",
            help="Training rows were seen by the model, so their probabilities are optimistic.",
        )

    params = RoiParams(float(offer_cost), acceptance / 100.0, cost_all)
    ltv_cfg = LtvSettings(source=ltv_source, months_by_contract=months, margin=margin)

    pop = scored if scope == SCOPE_ALL else scored[scored["split"].isin(["validation", "test"])]
    ltv = compute_ltv(pop, ltv_cfg).to_numpy()
    p = pop["p_churn"].to_numpy()

    curve = profit_curve(p, p, ltv, params)
    best = optimal_threshold(curve)
    profitable = bool(best["profitable"])
    fixed = evaluate_policy(p >= 0.5, p, ltv, params)

    if not profitable:
        st.warning(
            "No threshold earns a positive return under these assumptions: the offer costs "
            "more than the expected revenue it saves. The optimal decision is to not run a "
            "campaign."
        )

    m = st.columns(5)
    m[0].metric("Optimal threshold", threshold_label(best["threshold"], profitable))
    m[1].metric(
        "Customers to target",
        f"{int(best['targeted']):,}",
        help=f"{pct(best['targeted'] / max(len(pop), 1))} of {len(pop):,} customers simulated.",
    )
    m[2].metric("Revenue saved", money(best["revenue_saved"]))
    m[3].metric("Campaign cost", money(best["campaign_cost"]))
    m[4].metric(
        "Net $ saved",
        money(best["net_roi"]),
        delta=f"{money(best['net_roi'] - fixed['net_roi'])} vs fixed 0.50",
        help="Expected, using calibrated churn probabilities.",
    )

    st.plotly_chart(profit_chart(curve, best, profitable))

    st.markdown("**Cost-benefit matrix**")
    tab_exp, tab_real = st.tabs(["Expected (calibrated probabilities)", "Realised (held-out test)"])
    with tab_exp:
        st.dataframe(_style(cost_benefit_matrix(p, p, ltv, params, best["threshold"])))
        st.caption(
            "Revenue saved = acceptance × Σ P(churn) × LTV over targeted customers. "
            f"Campaign cost = {'every' if cost_all else 'accepting'} targeted customer "
            "× offer cost."
        )
    with tab_real:
        test = scored[scored["split"] == "test"]
        ltv_t = compute_ltv(test, ltv_cfg).to_numpy()
        st.dataframe(
            _style(
                cost_benefit_matrix(
                    test["p_churn"].to_numpy(),
                    test["churn"].to_numpy(),
                    ltv_t,
                    params,
                    best["threshold"],
                )
            )
        )
        st.caption(
            f"Same threshold applied to the {len(test):,} held-out test customers using their "
            "actual outcomes (acceptance applied in expectation)."
        )

    return RoiState(params, ltv_cfg, float(best["threshold"]), profitable)
