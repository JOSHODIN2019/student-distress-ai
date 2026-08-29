import streamlit as st
import joblib, os, pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="StudentAI — Distress Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── ChatGPT-style CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.25rem 1.5rem 2rem; max-width: 100%; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f9f9f9;
    border-right: 1px solid #e5e5e5;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
[data-testid="stSidebar"] .stRadio > div { gap: 6px; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 13px; font-weight: 500; color: #0d0d0d;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 10px;
    padding: 10px 12px 8px;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 11px; color: #6b6b6b; font-weight: 500;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1rem; font-weight: 700; color: #0d0d0d;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    height: 40px;
    font-family: 'Inter', sans-serif;
    transition: all .15s;
}
.stButton > button[kind="primary"] {
    background: #10a37f;
    border-color: #10a37f;
    color: white;
}
.stButton > button[kind="primary"]:hover {
    background: #0d8f6e;
    border-color: #0d8f6e;
}
.stButton > button[kind="secondary"] {
    border-color: #d1d5db;
    color: #374151;
    background: white;
}
.stButton > button[kind="secondary"]:hover { background: #f7f7f8; }

/* Form labels */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label {
    font-size: 12px; font-weight: 600; color: #0d0d0d;
}

/* Inputs & selects */
div[data-baseweb="select"] > div {
    border-color: #e5e5e5 !important;
    border-radius: 8px !important;
    background: white !important;
    font-size: 13px !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #10a37f !important;
    box-shadow: 0 0 0 3px rgba(16,163,127,.2) !important;
}
input[type="number"] {
    border-radius: 8px !important;
    border-color: #e5e5e5 !important;
    font-size: 13px !important;
}
input[type="number"]:focus {
    border-color: #10a37f !important;
    box-shadow: 0 0 0 3px rgba(16,163,127,.2) !important;
}

/* Slider accent */
[data-testid="stSlider"] [data-testid="stSlider"] div[role="slider"] { background: #10a37f; }

/* Panel boxes */
.panel-box {
    background: white;
    border: 1px solid #e5e5e5;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 0;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.panel-title {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .07em; color: #acacbe; margin-bottom: 14px;
}
.dot-live {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: #10a37f; margin-right: 6px; vertical-align: middle;
}

/* Score ring result */
.score-wrap { text-align: center; padding: 12px 0 8px; }
.badge-hi {
    display: inline-block;
    background: #fef2f2; color: #ef4444; border: 1px solid #fecaca;
    border-radius: 20px; padding: 4px 14px; font-size: 12px; font-weight: 700;
}
.badge-lo {
    display: inline-block;
    background: #e6f4f1; color: #10a37f; border: 1px solid #a7d9cc;
    border-radius: 20px; padding: 4px 14px; font-size: 12px; font-weight: 700;
}
.bar-track { background: #e5e5e5; border-radius: 4px; height: 6px; overflow: hidden; margin-top: 3px; }
.bar-fill-d  { background: #ef4444; border-radius: 4px; height: 6px; }
.bar-fill-ok { background: #10a37f; border-radius: 4px; height: 6px; }
.rec-row {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 5px 0; border-bottom: 1px solid #f7f7f8;
    font-size: 12px; color: #6b6b6b; line-height: 1.5;
}
.rec-row:last-child { border-bottom: none; }
.rec-pip-d  { width: 5px; height: 5px; border-radius: 50%; background: #ef4444; margin-top: 5px; flex-shrink: 0; }
.rec-pip-ok { width: 5px; height: 5px; border-radius: 50%; background: #10a37f; margin-top: 5px; flex-shrink: 0; }
.sb-row {
    display: flex; justify-content: space-between;
    font-size: 12px; padding: 4px 0; border-bottom: 1px solid rgba(0,0,0,.04);
}
.sb-row:last-child { border-bottom: none; }
.sb-key { color: #6b6b6b; }
.sb-val { font-weight: 600; color: #0d0d0d; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    lr  = joblib.load(os.path.join(BASE, "logistic_regression_model.pkl"))
    rf  = joblib.load(os.path.join(BASE, "random_forest_model.pkl"))
    sc  = joblib.load(os.path.join(BASE, "scaler.pkl"))
    enc = joblib.load(os.path.join(BASE, "label_encoders.pkl"))
    return lr, rf, sc, enc

lr_model, rf_model, scaler, encoders = load_models()

CATEGORICAL = ["Gender", "Sleep Duration", "Dietary Habits"]
NUMERICAL   = ["Age", "Academic Pressure", "CGPA", "Study Satisfaction", "Work/Study Hours", "Financial Stress"]
FEATURES    = ["Gender", "Age", "Academic Pressure", "CGPA", "Study Satisfaction",
               "Sleep Duration", "Dietary Habits", "Work/Study Hours", "Financial Stress"]

METRICS = {
    "lr": dict(name="Logistic Regression", acc="79.98%", prec="81.29%", rec="85.49%", f1="83.34%"),
    "rf": dict(name="Random Forest",       acc="79.01%", prec="81.47%", rec="83.04%", f1="82.25%"),
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 StudentAI")
    st.caption("Student Distress Detection System")
    st.divider()

    model_choice = st.radio("**AI Model**", ["Logistic Regression", "Random Forest"])
    mdl = "lr" if model_choice == "Logistic Regression" else "rf"
    m   = METRICS[mdl]

    st.divider()
    st.markdown("**Model Performance**")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy",  m["acc"])
    c2.metric("Precision", m["prec"])
    c1, c2 = st.columns(2)
    c1.metric("Recall",   m["rec"])
    c2.metric("F1-Score", m["f1"])

    st.divider()
    st.markdown("**Dataset**")
    st.markdown("""
<div>
<div class="sb-row"><span class="sb-key">Source</span><span class="sb-val">Kaggle</span></div>
<div class="sb-row"><span class="sb-key">Total Records</span><span class="sb-val">27,895</span></div>
<div class="sb-row"><span class="sb-key">Training Set</span><span class="sb-val">22,316 (80%)</span></div>
<div class="sb-row"><span class="sb-key">Test Set</span><span class="sb-val">5,579 (20%)</span></div>
<div class="sb-row"><span class="sb-key">Features</span><span class="sb-val">9</span></div>
<div class="sb-row"><span class="sb-key">Classes</span><span class="sb-val">2</span></div>
</div>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## Student Distress Analyzer")
st.caption(f"Predict psychological distress risk · Active model: **{m['name']}**")
st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([13, 9], gap="large")

# ── Left: Input ───────────────────────────────────────────────────────────────
with col_l:
    st.markdown('<div class="panel-title"><span class="dot-live"></span>Student Profile Input</div>', unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    gender = r1.selectbox("Gender", ["Male", "Female"])
    age    = r2.number_input("Age", min_value=15, max_value=45, value=21, step=1)

    r3, r4 = st.columns(2)
    academic_pressure = r3.slider("Academic Pressure", 1, 5, 3)
    cgpa              = r4.number_input("CGPA (0.0–5.0)", min_value=0.0, max_value=5.0, value=3.50, step=0.01, format="%.2f")

    r5, r6 = st.columns(2)
    study_satisfaction = r5.slider("Study Satisfaction", 1, 5, 3)
    sleep_duration     = r6.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Others"])

    r7, r8 = st.columns(2)
    dietary_habits   = r7.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy", "Others"])
    work_study_hours = r8.number_input("Work/Study Hours per Day", min_value=0.0, max_value=16.0, value=6.0, step=0.5, format="%.1f")

    financial_stress = st.slider("Financial Stress", 1, 5, 3)

    st.markdown("")
    ba, bb = st.columns([1, 4])
    reset_clicked   = ba.button("Reset",  use_container_width=True)
    analyze_clicked = bb.button(f"Analyze · {'LR' if mdl == 'lr' else 'RF'} →", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if reset_clicked:
    st.rerun()

result = None
if analyze_clicked:
    row = {
        "Gender":             gender,
        "Age":                float(age),
        "Academic Pressure":  int(academic_pressure),
        "CGPA":               float(cgpa),
        "Study Satisfaction": int(study_satisfaction),
        "Sleep Duration":     sleep_duration,
        "Dietary Habits":     dietary_habits,
        "Work/Study Hours":   float(work_study_hours),
        "Financial Stress":   int(financial_stress),
    }
    df = pd.DataFrame([row])
    for col in CATEGORICAL:
        df[col] = encoders[col].transform(df[col])
    df = df[FEATURES]
    df[NUMERICAL] = scaler.transform(df[NUMERICAL])

    model  = lr_model if mdl == "lr" else rf_model
    label  = int(model.predict(df)[0])
    probas = model.predict_proba(df)[0]
    conf   = float(probas[label]) * 100

    result = {
        "label":               label,
        "label_text":          "Distressed" if label == 1 else "Not Distressed",
        "confidence":          round(conf, 2),
        "prob_distressed":     round(float(probas[1]) * 100, 2),
        "prob_not_distressed": round(float(probas[0]) * 100, 2),
        "model":               m["name"],
        "inputs":              row,
    }

# ── Right: Result ─────────────────────────────────────────────────────────────
with col_r:
    st.markdown('<div class="panel-title">Prediction Result</div>', unsafe_allow_html=True)

    if result is None:
        st.markdown("""
<div style="text-align:center;padding:48px 20px;background:#f9f9f9;border:1px dashed #e5e5e5;border-radius:12px;">
    <div style="font-size:2.4rem;margin-bottom:10px;">🧠</div>
    <div style="font-size:13px;color:#6b6b6b;line-height:1.6;">Fill in the student profile and click<br><strong>Analyze</strong> to see the prediction.</div>
</div>""", unsafe_allow_html=True)
    else:
        is_d       = result["label"] == 1
        ring_color = "#ef4444" if is_d else "#10a37f"
        badge_cls  = "badge-hi" if is_d else "badge-lo"
        badge_txt  = "⚠ High Risk" if is_d else "✓ Low Risk"
        conf       = result["confidence"]
        pd_val     = result["prob_distressed"]
        pnd_val    = result["prob_not_distressed"]

        # SVG score ring
        r_val  = 44
        circ   = 2 * 3.14159265 * r_val
        offset = circ * (1 - conf / 100)
        cx = cy = 54

        st.markdown(f"""
<div class="score-wrap">
    <svg width="108" height="108" viewBox="0 0 108 108">
        <circle cx="{cx}" cy="{cy}" r="{r_val}" fill="none" stroke="#e5e5e5" stroke-width="8"/>
        <circle cx="{cx}" cy="{cy}" r="{r_val}" fill="none" stroke="{ring_color}" stroke-width="8"
                stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}"
                stroke-linecap="round" transform="rotate(-90 {cx} {cy})"/>
        <text x="{cx}" y="{cy - 5}" text-anchor="middle" font-size="22" font-weight="800"
              fill="{ring_color}" font-family="Inter,sans-serif">{conf:.0f}%</text>
        <text x="{cx}" y="{cy + 13}" text-anchor="middle" font-size="10" fill="#acacbe"
              font-family="Inter,sans-serif">Confidence</text>
    </svg>
    <div style="margin:6px 0;"><span class="{badge_cls}">{badge_txt}</span></div>
    <div style="font-size:19px;font-weight:800;color:{'#ef4444' if is_d else '#10a37f'};margin:4px 0 2px;">{result['label_text']}</div>
    <div style="font-size:12px;color:#acacbe;">via {result['model']}</div>
</div>
""", unsafe_allow_html=True)

        # Probability bars
        st.markdown(f"""
<div style="margin:4px 0 12px;">
    <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#acacbe;margin-bottom:10px;">Probability Breakdown</div>
    <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">
        <span style="color:#ef4444;font-weight:500;">Distressed</span>
        <span style="font-weight:700;color:#0d0d0d;">{pd_val:.1f}%</span>
    </div>
    <div class="bar-track"><div class="bar-fill-d" style="width:{pd_val:.1f}%;"></div></div>
    <div style="display:flex;justify-content:space-between;font-size:12px;margin:8px 0 3px;">
        <span style="color:#10a37f;font-weight:500;">Not Distressed</span>
        <span style="font-weight:700;color:#0d0d0d;">{pnd_val:.1f}%</span>
    </div>
    <div class="bar-track"><div class="bar-fill-ok" style="width:{pnd_val:.1f}%;"></div></div>
</div>
""", unsafe_allow_html=True)

        st.divider()

        # Recommendations
        rec_css   = "rec-pip-d" if is_d else "rec-pip-ok"
        recs_list = ([
            "Seek counseling or a mental health professional as soon as possible",
            "Speak to your academic advisor about reducing workload this semester",
            "Prioritise 7–8 hours of sleep — recovery depends on it",
            "Explore campus financial aid, bursaries, or scholarship programs",
            "Practice daily mindfulness, journaling, or breathing exercises",
            "Connect with peer support groups or student counselling services",
        ] if is_d else [
            "Your current routines appear balanced — maintain them consistently",
            "Continue healthy sleep habits for sustained academic performance",
            "Check in with your academic advisor each semester proactively",
            "Monitor stress levels and seek support early before issues escalate",
            "Consider supporting classmates who may be struggling",
        ])

        title = "Intervention Recommendations" if is_d else "Wellness Tips"
        rows_html = "".join(
            f'<div class="rec-row"><div class="{rec_css}"></div><span>{r}</span></div>'
            for r in recs_list
        )
        st.markdown(f"""
<div>
    <div style="font-size:12px;font-weight:700;color:#0d0d0d;margin-bottom:8px;">{title}</div>
    {rows_html}
</div>
""", unsafe_allow_html=True)
