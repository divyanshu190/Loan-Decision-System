# =============================================================================
#  app.py  —  Streamlit UI for Loan Default Prediction
#  Run: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np

from utils.predict    import predict
from utils.explain    import explain_prediction
from utils.preprocess import preprocess_input

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSight · Credit Risk Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Decision badge colors */
.badge-approve  { background:#d1fae5; color:#065f46; padding:6px 18px; border-radius:999px; font-weight:600; font-size:1.05rem; }
.badge-reject   { background:#fee2e2; color:#991b1b; padding:6px 18px; border-radius:999px; font-weight:600; font-size:1.05rem; }
.badge-review   { background:#fef3c7; color:#92400e; padding:6px 18px; border-radius:999px; font-weight:600; font-size:1.05rem; }
.badge-error    { background:#f3f4f6; color:#374151; padding:6px 18px; border-radius:999px; font-weight:600; font-size:1.05rem; }

.risk-chip      { display:inline-block; background:#f1f5f9; border:1px solid #e2e8f0; border-radius:6px;
                  padding:4px 10px; margin:3px 2px; font-size:0.82rem; color:#475569; }
.warn-chip      { display:inline-block; background:#fffbeb; border:1px solid #fde68a; border-radius:6px;
                  padding:4px 10px; margin:3px 2px; font-size:0.82rem; color:#92400e; }

.metric-card    { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
                  padding:16px 20px; text-align:center; }
.metric-label   { font-size:0.75rem; text-transform:uppercase; letter-spacing:.08em; color:#94a3b8; }
.metric-value   { font-family:'DM Serif Display',serif; font-size:2rem; color:#0f172a; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar — applicant inputs ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 LoanSight")
    st.caption("Credit Risk Engine · Give Me Some Credit dataset")
    st.divider()

    st.markdown("### Applicant Profile")

    age = st.slider("Age", min_value=18, max_value=100, value=35, step=1)

    monthly_income = st.number_input(
        "Monthly Income ($)", min_value=0, max_value=500_000, value=5000, step=500
    )

    revolving_util = st.slider(
        "Revolving Credit Utilization", min_value=0.0, max_value=1.0, value=0.25, step=0.01,
        format="%.2f"
    )

    debt_ratio = st.number_input(
        "Debt Ratio", min_value=0.0, max_value=500.0, value=0.30, step=0.01,
        format="%.2f"
    )

    st.markdown("##### Late Payments")
    late_30_59 = st.number_input("30–59 Days Late", min_value=0, max_value=20, value=0, step=1)
    late_60_89 = st.number_input("60–89 Days Late", min_value=0, max_value=20, value=0, step=1)
    late_90    = st.number_input("90+ Days Late",   min_value=0, max_value=20, value=0, step=1)

    st.markdown("##### Credit Lines")
    open_credit  = st.number_input("Open Credit Lines & Loans", min_value=0, max_value=50, value=5, step=1)
    real_estate  = st.number_input("Real Estate Loans",          min_value=0, max_value=20, value=1, step=1)
    dependents   = st.number_input("Number of Dependents",       min_value=0, max_value=20, value=1, step=1)

    st.divider()
    predict_btn = st.button("⚡ Predict", use_container_width=True, type="primary")


# ── Assemble input dict ────────────────────────────────────────────────────────
input_data = {
    "RevolvingUtilizationOfUnsecuredLines": revolving_util,
    "age":                                  age,
    "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
    "DebtRatio":                            debt_ratio,
    "MonthlyIncome":                        monthly_income if monthly_income > 0 else None,
    "NumberOfOpenCreditLinesAndLoans":      open_credit,
    "NumberOfTimes90DaysLate":              late_90,
    "NumberRealEstateLoansOrLines":         real_estate,
    "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
    "NumberOfDependents":                   dependents,
}


# ── Main panel ────────────────────────────────────────────────────────────────
st.title("Credit Risk Engine")
st.caption("Loan default prediction · Adjust the sidebar sliders and click Predict")

if not predict_btn:
    st.info("👈 Fill in the applicant profile on the left and click **⚡ Predict** to score.")
    st.stop()

# ── Run prediction ────────────────────────────────────────────────────────────
with st.spinner("Scoring applicant…"):
    preprocessed = preprocess_input(input_data)
    result       = predict(preprocessed)

# ── Error state ───────────────────────────────────────────────────────────────
if result.get("error"):
    st.error(f"**Prediction failed:** {result['error']}")
    st.stop()

decision     = result.get("decision", "ERROR")
probability  = result.get("probability", 0.0)
risk_score   = result.get("risk_score", 0)
confidence   = result.get("confidence", "—")
risk_factors = result.get("risk_factors", [])
warnings     = result.get("warnings", [])

# ── Top KPI row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

badge_class = {
    "APPROVE": "badge-approve",
    "REJECT":  "badge-reject",
    "REVIEW":  "badge-review",
}.get(decision, "badge-error")

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Decision</div>
        <div style="margin-top:8px">
            <span class="{badge_class}">{decision}</span>
        </div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Default Probability</div>
        <div class="metric-value">{probability:.1%}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Risk Score</div>
        <div class="metric-value">{risk_score}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Confidence</div>
        <div class="metric-value" style="font-size:1.5rem">{confidence}</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Probability gauge ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### Risk Score")

    bar_color = "#ef4444" if probability > 0.5 else ("#f59e0b" if probability > 0.25 else "#10b981")
    st.progress(min(probability, 1.0))
    st.caption(f"Default probability: **{probability:.2%}**  ·  Decision threshold: **0.25**")

    st.markdown("### Risk Factors")
    if risk_factors:
        chips = "".join(f'<span class="risk-chip">⚠ {f}</span>' for f in risk_factors)
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.success("No significant risk factors detected.")

    if warnings:
        st.markdown("##### Data Warnings")
        warn_chips = "".join(f'<span class="warn-chip">ℹ {w}</span>' for w in warnings)
        st.markdown(warn_chips, unsafe_allow_html=True)

with col_right:
    st.markdown("### Feature Importance (Explainability)")
    with st.spinner("Generating explanation…"):
        explanation = explain_prediction(preprocessed)

    if explanation:
        exp_df = pd.DataFrame(explanation, columns=["Feature", "Impact"])
        exp_df = exp_df.sort_values("Impact", key=abs, ascending=True)

        import plotly.express as px
        fig = px.bar(
            exp_df, x="Impact", y="Feature", orientation="h",
            color="Impact",
            color_continuous_scale=["#10b981", "#f8fafc", "#ef4444"],
            color_continuous_midpoint=0,
            template="simple_white",
            height=340,
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_showscale=False,
            xaxis_title="Impact on Default Probability",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Positive = increases default risk · Negative = reduces default risk")
    else:
        st.info("Explainability not available. See `utils/explain.py` to plug in SHAP.")

st.divider()

# ── What-If Analysis ──────────────────────────────────────────────────────────
st.markdown("### What-If Analysis")
st.caption("Adjust a single feature to see how the risk score changes.")

whatif_col, result_col = st.columns([1, 1])

with whatif_col:
    whatif_feature = st.selectbox(
        "Feature to vary",
        options=[
            "RevolvingUtilizationOfUnsecuredLines",
            "MonthlyIncome",
            "DebtRatio",
            "NumberOfTimes90DaysLate",
            "age",
        ]
    )

    ranges = {
        "RevolvingUtilizationOfUnsecuredLines": (0.0, 1.0, 0.05),
        "MonthlyIncome":                        (0, 20000, 500),
        "DebtRatio":                            (0.0, 2.0, 0.05),
        "NumberOfTimes90DaysLate":              (0, 15, 1),
        "age":                                  (18, 80, 1),
    }
    lo, hi, step = ranges[whatif_feature]
    n_steps = 20

    sweep_values = np.linspace(lo, hi, n_steps)
    sweep_probs  = []

    for v in sweep_values:
        trial_data = dict(input_data)
        trial_data[whatif_feature] = float(v)
        trial_pre  = preprocess_input(trial_data)
        trial_res  = predict(trial_pre)
        sweep_probs.append(trial_res.get("probability", 0.0))

with result_col:
    sweep_df = pd.DataFrame({
        whatif_feature: sweep_values,
        "Default Probability": sweep_probs,
    })

    import plotly.graph_objects as go
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sweep_df[whatif_feature],
        y=sweep_df["Default Probability"],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=5),
    ))
    fig2.add_hline(
        y=0.25, line_dash="dot", line_color="#f59e0b",
        annotation_text="Threshold (0.25)", annotation_position="bottom right"
    )
    fig2.update_layout(
        xaxis_title=whatif_feature,
        yaxis_title="Default Probability",
        yaxis=dict(range=[0, 1]),
        template="simple_white",
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)
