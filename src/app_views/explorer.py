"""At-Risk Account Explorer tab: filter subscribers, SHAP waterfall, AI retention offer."""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_views.common import BLUE, MUTED, ORANGE, money, pct
from src.app_views.roi_simulator import RoiState
from src.explain import DEMOGRAPHIC_FEATURES, driver_text
from src.retention import generate_offer, get_provider
from src.roi import compute_ltv

MAX_TABLE_ROWS = 500
WATERFALL_FEATURES = 7
PROVIDER_CHOICES = {
    "auto": "Auto (LLM_PROVIDER in .env)",
    "template": "Template only (no LLM)",
    "gemini": "Gemini",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
}


def _drivers(row: pd.Series) -> list[dict]:
    out = []
    for k in (1, 2, 3):
        feature = row.get(f"driver_{k}_feature")
        if isinstance(feature, str):
            out.append(
                {
                    "feature": feature,
                    "text": row[f"driver_{k}_text"],
                    "shap": float(row[f"driver_{k}_shap"]),
                }
            )
    return out


def waterfall(row: pd.Series, shap_row: pd.Series, base: float) -> go.Figure:
    """SHAP waterfall in log-odds: base rate -> this customer's score."""
    visible = shap_row.drop(labels=[f for f in DEMOGRAPHIC_FEATURES if f in shap_row.index])
    order = visible.abs().sort_values(ascending=False).index[:WATERFALL_FEATURES]
    labels = [driver_text(f, row) for f in order]
    values = [float(shap_row[f]) for f in order]
    other = float(shap_row.sum() - sum(values))
    labels.append("All other features (incl. demographics)")
    values.append(other)

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["absolute", *["relative"] * len(values), "total"],
            y=["Average customer", *labels, "This customer"],
            x=[base, *values, 0],
            text=["", *[f"{v:+.2f}" for v in values], f"{base + sum(values):+.2f}"],
            textposition="outside",
            increasing={"marker": {"color": ORANGE}},
            decreasing={"marker": {"color": BLUE}},
            totals={"marker": {"color": MUTED}},
            connector={"line": {"color": MUTED, "width": 1}},
            hovertemplate="%{y}<br>%{text} log-odds<extra></extra>",
        )
    )
    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 30, "t": 10, "b": 10},
        xaxis_title="Model log-odds of churn  (orange raises risk, blue lowers it)",
        yaxis={"autorange": "reversed"},
        showlegend=False,
    )
    return fig


def _filtered(scored: pd.DataFrame, roi: RoiState) -> tuple[pd.DataFrame, float]:
    st.markdown("**Filters**")
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1, 1.4])
    contracts = sorted(scored["contract"].unique())
    nets = sorted(scored["internet_type"].unique())
    sel_contract = f1.multiselect("Contract", contracts, default=contracts, key="f_contract")
    sel_net = f2.multiselect("Internet service", nets, default=nets, key="f_net")
    min_ltv = f3.number_input("Min LTV ($)", 0, 100_000, 0, step=100, key="f_ltv")
    include_churned = f4.checkbox(
        "Include customers who already churned",
        value=False,
        key="f_churned",
        help="Off = only current subscribers (the accounts a rep can still act on).",
    )

    g1, g2 = st.columns([1.4, 2])
    use_opt = g1.checkbox(
        f"Only campaign targets (p ≥ {roi.threshold:.2f})"
        if roi.profitable
        else "Only campaign targets (none profitable)",
        value=roi.profitable,
        key="f_use_opt",
        disabled=not roi.profitable,
    )
    min_p = g2.slider(
        "Minimum churn probability",
        0.0,
        1.0,
        0.5,
        step=0.01,
        key="f_minp",
        disabled=use_opt,
    )
    threshold = roi.threshold if use_opt else min_p

    df = scored[
        scored["contract"].isin(sel_contract)
        & scored["internet_type"].isin(sel_net)
        & (scored["ltv"] >= min_ltv)
        & (scored["p_churn"] >= threshold)
    ]
    if not include_churned:
        df = df[df["churn"] == 0]
    return df.sort_values("p_churn", ascending=False), threshold


def _table(df: pd.DataFrame, include_churned: bool) -> None:
    cols = ["customer_id", "p_churn", "ltv", "contract", "internet_type", "tenure",
            "monthly_charges", "driver_1_text", "driver_2_text", "driver_3_text"]  # fmt: skip
    if include_churned:
        cols.insert(2, "churn")
    show = (
        df[cols]
        .head(MAX_TABLE_ROWS)
        .rename(
            columns={
                "customer_id": "Customer",
                "p_churn": "Churn probability",
                "ltv": "LTV ($)",
                "contract": "Contract",
                "internet_type": "Internet",
                "tenure": "Tenure (mo)",
                "monthly_charges": "Monthly ($)",
                "driver_1_text": "Driver 1",
                "driver_2_text": "Driver 2",
                "driver_3_text": "Driver 3",
                "churn": "Already churned",
            }
        )
    )
    st.dataframe(
        show,
        hide_index=True,
        column_config={
            "Churn probability": st.column_config.ProgressColumn(
                "Churn probability", min_value=0.0, max_value=1.0, format="percent"
            ),
            "LTV ($)": st.column_config.NumberColumn(format="$%.0f"),
            "Monthly ($)": st.column_config.NumberColumn(format="$%.2f"),
        },
    )


def _generator(row: pd.Series, drivers: list[dict], ltv: float, roi: RoiState) -> None:
    st.markdown("**AI Retention Action Generator**")
    g1, g2 = st.columns([1, 2])
    choice = g1.selectbox(
        "Engine",
        list(PROVIDER_CHOICES),
        format_func=PROVIDER_CHOICES.get,
        key="llm_choice",
    )
    provider = None if choice == "template" else get_provider(None if choice == "auto" else choice)
    if choice not in ("template", "auto") and provider is None:
        g2.info(f"No API key found for {PROVIDER_CHOICES[choice]} - using the template engine.")
    elif choice == "auto" and provider is None:
        g2.info(
            "No LLM configured (set LLM_PROVIDER and its API key in .env) - "
            "using the deterministic template engine."
        )
    elif provider is not None:
        g2.caption(
            f"Model: {provider.model}. Prices are computed in code; "
            "the model only chooses actions and words the message."
        )

    budget = roi.params.retention_offer_cost
    key = (row["customer_id"], budget, choice)
    if st.button("Generate retention offer", key="gen_offer", type="primary"):
        context = {"p_churn": float(row["p_churn"]), "ltv": ltv}
        with st.spinner("Drafting offer..."):
            st.session_state["offer"] = (
                key,
                generate_offer(row.to_dict(), drivers, budget, provider, context),
            )

    stored = st.session_state.get("offer")
    if not stored or stored[0] != key:
        st.caption(
            f"Drafts an offer from this customer's top drivers within a ${budget:,.0f} budget."
        )
        return
    offer = stored[1]

    st.markdown(f"**{offer.badge}**")
    if offer.fallback_reason:
        st.warning(f"LLM output not used - fell back to template. Reason: {offer.fallback_reason}")
    st.markdown(f"### {offer.headline}")
    st.text_area(
        "Draft customer message (editable)",
        value=offer.customer_message,
        height=140,
        key=f"msg_{abs(hash(offer.customer_message))}",
    )
    cost_ok = "within" if offer.within_budget else "OVER"
    st.markdown(
        f"Estimated offer cost **{money(offer.total_cost, 2)}**, {cost_ok} the "
        f"{money(offer.budget)} budget."
    )
    st.markdown("**Rep notes**")
    st.text(offer.rep_notes)


def render(data: dict, roi: RoiState) -> None:
    scored = data["scored"].copy()
    scored["ltv"] = compute_ltv(scored, roi.ltv_cfg)

    st.subheader("Which accounts should a rep call, and what should they offer?")
    filtered, threshold = _filtered(scored, roi)
    include_churned = bool(st.session_state.get("f_churned", False))

    n_active = int((scored["churn"] == 0).sum())
    st.caption(
        f"{len(filtered):,} accounts match (showing the top {min(len(filtered), MAX_TABLE_ROWS)} "
        f"by risk). Current subscribers in the book: {n_active:,}."
    )
    if filtered.empty:
        st.info("No accounts match these filters.")
        return
    _table(filtered, include_churned)

    st.divider()
    top = filtered.head(MAX_TABLE_ROWS)
    labels = {
        r.customer_id: f"{r.customer_id}  -  {pct(r.p_churn, 0)} risk  -  {r.contract}"
        for r in top.itertuples()
    }
    cid = st.selectbox(
        "Inspect an account",
        list(labels),
        format_func=labels.get,
        key="selected_customer",
    )
    row = top[top["customer_id"] == cid].iloc[0]
    shap_row = (
        data["shap"].loc[data["shap"]["customer_id"] == cid].drop(columns="customer_id").iloc[0]
    )
    base = float(data["explain"]["shap_base_value_logodds"])
    drivers = _drivers(row)
    ltv = float(row["ltv"])
    expected_net = roi.params.offer_acceptance_rate * float(row["p_churn"]) * ltv - (
        roi.params.retention_offer_cost
        if roi.params.cost_applies_to_all_targeted
        else roi.params.offer_acceptance_rate * roi.params.retention_offer_cost
    )

    s = st.columns(4)
    s[0].metric("Churn probability", pct(row["p_churn"]))
    s[1].metric("LTV at stake", money(ltv))
    s[2].metric("Expected net of offering", money(expected_net))
    s[3].metric("Contract / tenure", f"{row['contract']} / {int(row['tenure'])} mo")

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("**Why this account is flagged (SHAP waterfall)**")
        st.plotly_chart(waterfall(row, shap_row, base))
    with right:
        st.markdown("**Top churn drivers**")
        for i, d in enumerate(drivers, 1):
            st.markdown(
                f"{i}. {d['text']}  \n<small>impact {d['shap']:+.2f} log-odds</small>",
                unsafe_allow_html=True,
            )
        if not drivers:
            st.caption("No risk-raising drivers for this account.")
    st.divider()
    _generator(row, drivers, ltv, roi)
    _ = os  # env is read inside get_provider
