"""Interactive dashboard for the Customer Churn Prediction project."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.predict import load_artifact, predict_one


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "churn_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
DATA_PATH = ROOT / "data" / "telco_customer_churn.csv"
RISK_LOW = 0.35
RISK_HIGH = 0.65

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#172033; --muted:#718096; --blue:#4263eb; --teal:#12b8a6; }
    html, body, [class*="css"] { font-family:Inter,ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif; }
    .stApp { background:radial-gradient(circle at 8% 0%,rgba(66,99,235,.07),transparent 25%),#f4f7fb; }
    .block-container { max-width:1480px; padding-top:1.65rem; padding-bottom:2.6rem; }
    [data-testid="stSidebar"] { background:#111a2d; border-right:1px solid rgba(255,255,255,.08); }
    [data-testid="stSidebar"] * { color:#e7edf8; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#aebbd1; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.48rem .3rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p { color:#dce5f5; font-weight:600; }
    [data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.12); }
    [data-testid="stMetric"] { background:transparent; }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; color:var(--ink); }
    h1 { font-size:2rem !important; }
    h2 { font-size:1.28rem !important; }
    [data-testid="stCaptionContainer"] p { color:#76839a; }
    [data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {
        border-radius:10px; font-weight:700; min-height:2.75rem; transition:all .18s ease;
    }
    [data-testid="stFormSubmitButton"] button[kind="primary"], [data-testid="stButton"] button[kind="primary"] {
        background:#4263eb; border:1px solid #4263eb; color:#fff;
        box-shadow:0 8px 18px rgba(66,99,235,.18);
    }
    [data-testid="stFormSubmitButton"] button[kind="primary"]:hover { background:#3153d9; border-color:#3153d9; }
    [data-testid="stDataFrame"] { border:1px solid #e6ebf2; border-radius:12px; overflow:hidden; }
    [data-testid="stFileUploader"] section { background:#fff; border:1px dashed #c5cede; border-radius:14px; }
    [data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-color:#e7ecf3; border-radius:16px; padding:.35rem .55rem; box-shadow:0 5px 18px rgba(31,49,81,.035); }
    .brand { display:flex; align-items:center; gap:11px; padding:.4rem .1rem 1.2rem; }
    .brand-mark { width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center;
        background:linear-gradient(135deg,#6d83ff,#4263eb 56%,#12b8a6); color:white; font:800 20px Manrope,sans-serif;
        box-shadow:0 8px 18px rgba(66,99,235,.3); }
    .brand-name { color:#fff; font:800 17px Manrope,sans-serif; letter-spacing:-.04em; }
    .brand-sub { color:#8290a8; font-size:10px; letter-spacing:.14em; text-transform:uppercase; margin-top:1px; }
    .sidebar-status { display:flex; align-items:center; gap:8px; border:1px solid rgba(255,255,255,.1); border-radius:11px;
        background:rgba(255,255,255,.045); padding:10px 12px; color:#e7edf8; font-size:12px; font-weight:600; }
    .status-dot { width:8px; height:8px; border-radius:99px; background:#34d399; box-shadow:0 0 0 4px rgba(52,211,153,.12); }
    .sidebar-note { color:#8391a8; font-size:11px; line-height:1.55; margin-top:12px; }
    .hero { position:relative; overflow:hidden; border-radius:20px; padding:26px 30px; margin-bottom:20px;
        background:linear-gradient(112deg,#142342 0%,#20366b 58%,#28547a 100%); color:white;
        box-shadow:0 16px 34px rgba(26,47,89,.15); }
    .hero:after { content:''; position:absolute; width:330px; height:330px; border-radius:50%; right:-65px; top:-205px;
        background:radial-gradient(circle,rgba(78,211,196,.34),rgba(78,211,196,0) 68%); }
    .hero-kicker { text-transform:uppercase; letter-spacing:.16em; color:#a8c4ff; font-size:10px; font-weight:700; margin-bottom:7px; }
    .hero-title { font:800 27px Manrope,sans-serif; letter-spacing:-.05em; color:#fff; margin:0 0 7px; }
    .hero-copy { color:#c4d0e4; font-size:13px; max-width:760px; line-height:1.55; margin:0; }
    .hero-chip { display:inline-flex; align-items:center; gap:8px; border:1px solid rgba(255,255,255,.18);
        background:rgba(255,255,255,.08); color:#e3edff; border-radius:99px; padding:7px 11px; font-size:11px; font-weight:600; margin-top:15px; }
    .card { background:#fff; border:1px solid #e7ecf3; border-radius:16px; padding:17px 19px; box-shadow:0 5px 18px rgba(31,49,81,.035); }
    .kpi-label { color:#77849a; font-size:11px; font-weight:700; letter-spacing:.055em; text-transform:uppercase; }
    .kpi-value { color:#1c2941; font:800 25px Manrope,sans-serif; letter-spacing:-.04em; margin:7px 0 3px; }
    .kpi-note { color:#8994a7; font-size:11px; }
    .kpi-accent { width:27px; height:3px; border-radius:6px; margin-bottom:13px; }
    .section-head { display:flex; justify-content:space-between; align-items:flex-end; margin:22px 0 11px; }
    .section-title { color:#1c2941; font:800 16px Manrope,sans-serif; letter-spacing:-.03em; }
    .section-note { color:#8290a3; font-size:11px; }
    .white-panel { background:#fff; border:1px solid #e7ecf3; border-radius:16px; padding:18px 20px; height:100%; }
    .panel-title { color:#26334a; font:700 14px Manrope,sans-serif; margin:0 0 3px; }
    .panel-copy { color:#8490a2; font-size:11px; margin-bottom:14px; }
    .score-panel { background:linear-gradient(155deg,#16284d,#243a69); color:white; border-radius:18px; padding:23px; min-height:260px;
        box-shadow:0 15px 30px rgba(25,46,90,.16); }
    .score-kicker { text-transform:uppercase; letter-spacing:.14em; font-size:10px; color:#b9c8e4; font-weight:700; }
    .score-number { font:800 46px Manrope,sans-serif; letter-spacing:-.06em; color:#fff; margin:10px 0 2px; }
    .score-label { color:#d4def0; font-size:13px; }
    .score-band { display:inline-flex; align-items:center; gap:7px; padding:7px 10px; border-radius:99px; margin-top:14px; font-size:11px; font-weight:700; }
    .score-low { background:rgba(52,211,153,.15); color:#83f0c4; }
    .score-medium { background:rgba(250,204,21,.16); color:#ffe27a; }
    .score-high { background:rgba(251,113,133,.16); color:#ffabb8; }
    .signal { border:1px solid #e8edf4; border-radius:11px; background:#fff; padding:11px 12px; margin:7px 0; }
    .signal-name { color:#33415a; font-size:12px; font-weight:700; }
    .signal-value { color:#8692a5; font-size:10px; margin-top:2px; }
    .signal-up { color:#c2414e; font-size:11px; font-weight:700; }
    .signal-down { color:#11866f; font-size:11px; font-weight:700; }
    .matrix { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px; }
    .matrix-cell { border-radius:12px; padding:15px; background:#f3f6fb; }
    .matrix-cell.positive { background:#e9f8f3; }
    .matrix-cell.alert { background:#fff1f1; }
    .matrix-num { font:800 22px Manrope,sans-serif; color:#26354d; }
    .matrix-label { color:#79869b; font-size:10px; margin-top:3px; }
    .pill { display:inline-block; border:1px solid #e3e9f2; background:#f8faff; color:#586983; border-radius:99px;
        padding:5px 9px; font-size:10px; font-weight:700; }
    .notice { background:#edf3ff; border-left:3px solid #4263eb; border-radius:0 10px 10px 0; color:#52627b; padding:11px 14px; font-size:11px; line-height:1.55; }
    .footer { border-top:1px solid #e6ebf2; color:#8a95a7; font-size:10px; padding-top:13px; margin-top:30px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model() -> dict[str, Any]:
    return load_artifact(MODEL_PATH)


@st.cache_data
def get_metrics() -> dict[str, Any]:
    import json

    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def hero(kicker: str, title: str, copy: str, chip: str = "") -> None:
    chip_html = f'<div class="hero-chip">◈&nbsp; {chip}</div>' if chip else ""
    st.markdown(
        f'<div class="hero"><div class="hero-kicker">{kicker}</div>'
        f'<div class="hero-title">{title}</div><p class="hero-copy">{copy}</p>{chip_html}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, note: str = "") -> None:
    note_html = f'<div class="section-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="section-head"><div class="section-title">{title}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, note: str, accent: str) -> None:
    st.markdown(
        f'<div class="card"><div class="kpi-accent" style="background:{accent}"></div>'
        f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def risk_band(score: float) -> tuple[str, str]:
    if score >= RISK_HIGH:
        return "High", "score-high"
    if score >= RISK_LOW:
        return "Medium", "score-medium"
    return "Low", "score-low"


def local_signals(customer: dict[str, Any], model_data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Explain logistic-regression score contributions; these are associations, not causes."""
    pipeline = model_data["model"]
    classifier = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocess")
    if classifier is None or preprocessor is None or not hasattr(classifier, "coef_"):
        return [], []

    row = pd.DataFrame([customer]).reindex(columns=model_data["feature_names"])
    transformed = preprocessor.transform(row)
    values = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]
    contributions = values * classifier.coef_[0]
    encoded_names = preprocessor.get_feature_names_out()
    original_features = sorted(model_data["feature_names"], key=len, reverse=True)
    signals: list[dict[str, Any]] = []
    for encoded, contribution in zip(encoded_names, contributions):
        if abs(float(contribution)) < 0.025:
            continue
        clean = encoded.split("__", 1)[-1]
        for feature in original_features:
            if clean.startswith(feature + "_"):
                clean = f"{feature}: {clean[len(feature) + 1:]}"
                break
        clean = clean.replace("_", " ")
        signals.append({"signal": clean, "contribution": float(contribution)})
    toward = sorted((s for s in signals if s["contribution"] > 0), key=lambda s: s["contribution"], reverse=True)[:4]
    away = sorted((s for s in signals if s["contribution"] < 0), key=lambda s: s["contribution"])[:4]
    return toward, away


def render_signals(customer: dict[str, Any], model_data: dict[str, Any]) -> None:
    toward, away = local_signals(customer, model_data)
    if not toward and not away:
        st.markdown(
            '<div class="notice">Local feature signals are available when the selected estimator exposes coefficients.</div>',
            unsafe_allow_html=True,
        )
        return
    signal_cols = st.columns(2)
    with signal_cols[0]:
        st.markdown('<div class="panel-title">Signals toward churn</div>', unsafe_allow_html=True)
        if toward:
            for item in toward:
                st.markdown(
                    f'<div class="signal"><span class="signal-name">{item["signal"]}</span>'
                    f'<span style="float:right" class="signal-up">+{item["contribution"]:.2f}</span>'
                    '<div class="signal-value">Model contribution</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No strong positive signal in this profile.")
    with signal_cols[1]:
        st.markdown('<div class="panel-title">Signals away from churn</div>', unsafe_allow_html=True)
        if away:
            for item in away:
                st.markdown(
                    f'<div class="signal"><span class="signal-name">{item["signal"]}</span>'
                    f'<span style="float:right" class="signal-down">{item["contribution"]:.2f}</span>'
                    '<div class="signal-value">Model contribution</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No strong negative signal in this profile.")
    st.caption("Signals describe how this fitted model arrived at its score; they are not causal explanations.")


if not MODEL_PATH.exists() or not METRICS_PATH.exists() or not DATA_PATH.exists():
    st.error("A project file is missing. Restore the model, reports, and data folders, then reload the app.")
    st.stop()

try:
    model_data = get_model()
    metrics = get_metrics()
    data = get_dataset()
except Exception as exc:
    st.error(f"The dashboard could not load its project files: {exc}")
    st.stop()

test = metrics["holdout_test"]
class_counts = metrics["class_counts"]
churn_rate = class_counts["churn"] / metrics["dataset_rows"]

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">◈</div><div><div class="brand-name">ChurnIQ</div>'
        '<div class="brand-sub">Customer intelligence</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div class='sidebar-status'><span class='status-dot'></span> Model active</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    page = st.radio(
        "Workspace",
        ["Overview", "Customer scoring", "Batch scoring", "Model insights"],
        label_visibility="collapsed",
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"**{model_data['model_name']}**<br><span style='font-size:11px'>Trained project model · threshold {model_data.get('threshold', .5):.2f}</span>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sidebar-note'>Academic demonstration using historical telecom data. Scores should support analysis, not automated customer decisions.</div>",
        unsafe_allow_html=True,
    )


if page == "Overview":
    hero(
        "CUSTOMER RETENTION WORKSPACE",
        "Understand churn before it happens.",
        "Explore the customer portfolio, review model performance, and score individual or uploaded customer profiles in one place.",
        f"{metrics['dataset_rows']:,} customer records  ·  {metrics['feature_count']} model features  ·  {model_data['model_name']}",
    )
    cards = st.columns(4, gap="medium")
    with cards[0]:
        kpi("Holdout ROC-AUC", f"{test['roc_auc']:.3f}", "Ranking quality on unseen customers", "#4263eb")
    with cards[1]:
        kpi("Churn recall", f"{test['recall']:.1%}", "Share of churners identified", "#13b8a6")
    with cards[2]:
        kpi("Holdout F1", f"{test['f1']:.3f}", "Balance of precision and recall", "#f2a93b")
    with cards[3]:
        kpi("Observed churn", f"{churn_rate:.1%}", f"{class_counts['churn']:,} of {metrics['dataset_rows']:,} records", "#9b72cf")

    section("Portfolio snapshot", "The supplied dataset is imbalanced; recall and precision matter alongside accuracy.")
    left, right = st.columns([1.1, 1], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Historical class distribution</div><div class="panel-copy">Customer outcomes in the training dataset</div>', unsafe_allow_html=True)
            class_frame = pd.DataFrame({"Customers": [class_counts["no_churn"], class_counts["churn"]]}, index=["No churn", "Churn"])
            st.bar_chart(class_frame, color="#4263eb", height=220)
    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Model selection</div><div class="panel-copy">Five-fold stratified cross-validation on the training split</div>', unsafe_allow_html=True)
            cv_frame = pd.DataFrame(metrics["cross_validation"]).sort_values("roc_auc_mean", ascending=False)
            cv_chart = cv_frame.set_index("model")[["roc_auc_mean", "accuracy_mean"]].rename(columns={"roc_auc_mean": "ROC-AUC", "accuracy_mean": "Accuracy"})
            st.bar_chart(cv_chart, color=["#4263eb", "#a9b8d2"], height=220)
            st.markdown(f"<span class='pill'>Selected by ROC-AUC: {model_data['model_name']}</span>", unsafe_allow_html=True)

    section("What to do next")
    action_cols = st.columns(3, gap="medium")
    actions = [
        ("01", "Score one customer", "Enter a service profile and inspect its estimated churn score."),
        ("02", "Score a portfolio", "Upload a CSV to generate a downloadable batch of churn scores."),
        ("03", "Review the model", "Inspect the holdout confusion matrix, cross-validation results, and limitations."),
    ]
    for col, (number, title, copy) in zip(action_cols, actions):
        with col:
            st.markdown(
                f'<div class="card"><div class="kpi-accent" style="background:#4263eb"></div>'
                f'<div class="kpi-label">WORKFLOW {number}</div><div class="panel-title" style="margin:8px 0">{title}</div>'
                f'<div class="kpi-note">{copy}</div></div>',
                unsafe_allow_html=True,
            )


elif page == "Customer scoring":
    hero(
        "INDIVIDUAL ASSESSMENT",
        "Score a customer profile.",
        "Complete the service and billing profile to estimate churn risk. The score is generated locally by the saved project pipeline.",
        f"Decision threshold {model_data.get('threshold', .5):.0%}  ·  Risk bands are for demonstration",
    )
    form_col, result_col = st.columns([1.18, .82], gap="large")
    with form_col, st.container(border=True):
        st.markdown('<div class="panel-title">Customer profile</div><div class="panel-copy">Use the customer\'s current account information</div>', unsafe_allow_html=True)
        with st.form("customer_profile"):
            row1, row2, row3 = st.columns(3)
            with row1:
                st.markdown("**Account**")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior citizen", ["No", "Yes"])
                partner = st.selectbox("Has a partner", ["Yes", "No"])
                dependents = st.selectbox("Has dependents", ["Yes", "No"])
                tenure = st.slider("Tenure (months)", 0, 72, 12)
            with row2:
                st.markdown("**Services**")
                phone = st.selectbox("Phone service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple lines", ["No phone service", "No", "Yes"])
                internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
                no_internet = "No internet service"
                online_security = st.selectbox("Online security", ["No", "Yes", no_internet])
                online_backup = st.selectbox("Online backup", ["No", "Yes", no_internet])
                device_protection = st.selectbox("Device protection", ["No", "Yes", no_internet])
                tech_support = st.selectbox("Tech support", ["No", "Yes", no_internet])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", no_internet])
                streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", no_internet])
            with row3:
                st.markdown("**Billing**")
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless billing", ["Yes", "No"])
                payment = st.selectbox(
                    "Payment method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                )
                monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=1.0)
                total = st.number_input("Total charges ($)", min_value=0.0, max_value=100000.0, value=840.0, step=10.0)
            submitted = st.form_submit_button("✦  Estimate churn risk", type="primary", use_container_width=True)

    customer = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    if submitted:
        label, score = predict_one(customer, model_data)
        st.session_state["last_prediction"] = {"customer": customer, "label": label, "score": score}
    result = st.session_state.get("last_prediction")
    with result_col:
        if result:
            score = float(result["score"])
            label = result["label"]
            band, band_class = risk_band(score)
            st.markdown(
                f'<div class="score-panel"><div class="score-kicker">MODEL ASSESSMENT</div>'
                f'<div class="score-number">{score:.1%}</div><div class="score-label">Estimated churn score</div>'
                f'<div class="score-band {band_class}">●&nbsp; {band} risk</div>'
                f'<div style="margin-top:21px;border-top:1px solid rgba(255,255,255,.14);padding-top:14px;color:#bfcee7;font-size:11px">'
                f'Predicted class <span style="float:right;color:#fff;font-weight:700">{label}</span></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('<div class="panel-title">Profile signals</div><div class="panel-copy">Largest model contributions for this profile</div>', unsafe_allow_html=True)
                render_signals(result["customer"], model_data)
        else:
            st.markdown(
                '<div class="score-panel"><div class="score-kicker">YOUR RESULT</div>'
                '<div class="score-number">—</div><div class="score-label">Complete the profile and run an assessment</div>'
                '<div class="score-band score-medium">◉&nbsp; Ready to score</div>'
                '<div style="margin-top:21px;border-top:1px solid rgba(255,255,255,.14);padding-top:14px;color:#bfcee7;font-size:11px">'
                'Your profile is processed locally in this project app.</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="notice">The score is not a calibrated individual probability. Use it as an academic model output, not as a decision about a real customer.</div>', unsafe_allow_html=True)


elif page == "Batch scoring":
    hero(
        "PORTFOLIO SCORING",
        "Score a customer list in one pass.",
        "Upload a CSV containing the project input columns. The app scores the rows locally and lets you download the results.",
        "CSV processing stays in this local dashboard session",
    )
    feature_names = model_data["feature_names"]
    template = data.reindex(columns=feature_names).head(5).to_csv(index=False).encode("utf-8")
    template_col, upload_col = st.columns([.85, 1.15], gap="large")
    with template_col, st.container(border=True):
        st.markdown('<div class="panel-title">Start with the template</div><div class="panel-copy">It contains the model input columns in the expected format.</div>', unsafe_allow_html=True)
        st.download_button("Download CSV template", template, file_name="customer_churn_template.csv", mime="text/csv", use_container_width=True)
        st.markdown(f"<br><span class='pill'>{len(feature_names)} required input columns</span>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-note' style='color:#7f8ba0'>The customerID and Churn columns are not needed for scoring. Extra columns are ignored.</div>", unsafe_allow_html=True)
    with upload_col, st.container(border=True):
        st.markdown('<div class="panel-title">Upload customer profiles</div><div class="panel-copy">Accepted format: CSV. No model retraining occurs here.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose a customer CSV", type=["csv"], label_visibility="collapsed")
        st.markdown("<div class='sidebar-note' style='color:#7f8ba0'>Rows are processed in memory for this session. The app does not save the uploaded file.</div>", unsafe_allow_html=True)

    if uploaded is not None:
        try:
            batch = pd.read_csv(BytesIO(uploaded.getvalue()))
            batch = batch.drop(columns=["customerID", "Churn"], errors="ignore")
            missing = [name for name in feature_names if name not in batch.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            elif batch.empty:
                st.warning("The uploaded CSV has no customer rows.")
            else:
                batch = batch.reindex(columns=feature_names)
                scores = model_data["model"].predict_proba(batch)[:, 1]
                threshold = float(model_data.get("threshold", 0.5))
                predictions = (scores >= threshold).astype(int)
                result = pd.read_csv(BytesIO(uploaded.getvalue()))
                result["Churn score"] = scores
                result["Predicted churn"] = np.where(predictions == 1, "Yes", "No")
                result["Risk band"] = [risk_band(float(score))[0] for score in scores]
                high_count = int((result["Risk band"] == "High").sum())
                churn_count = int((result["Predicted churn"] == "Yes").sum())

                section("Scoring summary", f"Decision threshold {threshold:.0%}")
                m1, m2, m3, m4 = st.columns(4, gap="medium")
                with m1:
                    kpi("Customers scored", f"{len(result):,}", "Rows in this upload", "#4263eb")
                with m2:
                    kpi("Predicted churn", f"{churn_count:,}", f"{churn_count / len(result):.1%} of uploaded rows", "#f2a93b")
                with m3:
                    kpi("High risk band", f"{high_count:,}", "Score at or above 65%", "#e05b66")
                with m4:
                    kpi("Average score", f"{scores.mean():.1%}", "Uncalibrated model score", "#13b8a6")
                section("Scored customer list")
                st.dataframe(result, use_container_width=True, hide_index=True, height=360)
                st.download_button(
                    "Download scored CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="customer_churn_scores.csv",
                    mime="text/csv",
                    type="primary",
                )
        except Exception as exc:
            st.error(f"Could not read or score this CSV: {exc}")


elif page == "Model insights":
    hero(
        "MODEL CARD",
        "Performance, clearly explained.",
        "Review the model selection, test-set trade-offs, and the boundaries of what this academic demonstration can tell you.",
        f"Selected model: {model_data['model_name']}  ·  Random seed {model_data.get('random_state', 42)}",
    )
    cv = pd.DataFrame(metrics["cross_validation"]).sort_values("roc_auc_mean", ascending=False)
    section("Cross-validation leaderboard", "Five folds · training split only · higher is better")
    shown = cv[["model", "accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean", "roc_auc_std"]].copy()
    shown.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "ROC-AUC SD"]
    st.dataframe(shown.style.format({c: "{:.3f}" for c in shown.columns if c != "Model"}), use_container_width=True, hide_index=True)

    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown('<div class="white-panel"><div class="panel-title">Holdout confusion matrix</div><div class="panel-copy">Actual rows × predicted columns · threshold 0.50</div>', unsafe_allow_html=True)
        tn, fp = test["confusion_matrix"][0]
        fn, tp = test["confusion_matrix"][1]
        st.markdown(
            f'<div class="matrix"><div class="matrix-cell positive"><div class="matrix-num">{tn:,}</div><div class="matrix-label">Correctly identified · no churn</div></div>'
            f'<div class="matrix-cell alert"><div class="matrix-num">{fp:,}</div><div class="matrix-label">Flagged · no churn</div></div>'
            f'<div class="matrix-cell alert"><div class="matrix-num">{fn:,}</div><div class="matrix-label">Missed · churn</div></div>'
            f'<div class="matrix-cell positive"><div class="matrix-num">{tp:,}</div><div class="matrix-label">Correctly identified · churn</div></div></div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="white-panel"><div class="panel-title">Holdout metrics</div><div class="panel-copy">Unseen stratified test data</div>', unsafe_allow_html=True)
        metric_df = pd.DataFrame(
            {"Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
             "Score": [test[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]}
        )
        st.dataframe(metric_df.style.format({"Score": "{:.3f}"}), use_container_width=True, hide_index=True)
        st.markdown(f"<span class='pill'>Threshold {test['threshold']:.0%}</span> &nbsp; <span class='pill'>Test rows {metrics['test_rows']:,}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    section("Interpretation and limitations")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="white-panel"><div class="panel-title">How to read this result</div><div class="panel-copy">The selected model optimizes ranking quality, not a business outcome.</div><div class="notice">Recall of 78.3% means the model identified about 78 of every 100 churners in this holdout sample. Precision of 50.4% means about half of its churn flags were correct in this sample.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="white-panel"><div class="panel-title">Use with care</div><div class="panel-copy">This is a classroom project, not a production decision system.</div><div class="notice">The data is historical, scores are uncalibrated, and results are from one holdout split. Review privacy and fairness, validate on newer data, and set thresholds from real intervention costs before any operational use.</div></div>', unsafe_allow_html=True)


st.markdown(
    f'<div class="footer">ChurnIQ · Academic project dashboard &nbsp; | &nbsp; {model_data["model_name"]} '
    f'&nbsp; | &nbsp; Model artifact: {model_data.get("scikit_learn_version", "scikit-learn")}</div>',
    unsafe_allow_html=True,
)
