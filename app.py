import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# CONFIG
st.set_page_config(
    page_title="PlacementIQ",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0e1a; color: #e8eaf0; }
[data-testid="stSidebar"] { background: #0f1525 !important; border-right: 1px solid #1e2a45; }
[data-testid="stSidebar"] * { color: #c8cfe0 !important; }
.brand-header {
    font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #4f8ef7 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin-bottom: 0.2rem;
}
.brand-sub { font-size: 0.85rem; color: #5a6a8a; letter-spacing: 2px; text-transform: uppercase; font-weight: 500; margin-bottom: 2rem; }
.section-title {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;
    color: #e8eaf0; padding-bottom: 0.5rem; border-bottom: 1px solid #1e2a45; margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e2d4a; border-radius: 16px; padding: 1.4rem 1.6rem; text-align: center;
}
.metric-label { font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase; color: #5a6a8a; font-weight: 500; margin-bottom: 0.4rem; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: #e8eaf0; line-height: 1; }
.metric-value.green  { color: #34d399; }
.metric-value.blue   { color: #60a5fa; }
.metric-value.purple { color: #a78bfa; }
.metric-value.orange { color: #fb923c; }
.result-placed {
    background: linear-gradient(135deg, #064e3b, #065f46); border: 1px solid #34d399;
    border-radius: 16px; padding: 1.6rem 2rem; text-align: center; margin: 1rem 0;
}
.result-not-placed {
    background: linear-gradient(135deg, #450a0a, #7f1d1d); border: 1px solid #f87171;
    border-radius: 16px; padding: 1.6rem 2rem; text-align: center; margin: 1rem 0;
}
.result-title { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; margin-bottom: 0.3rem; }
.result-subtitle { font-size: 0.9rem; color: #94a3b8; }
.stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #a78bfa) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
    font-size: 0.95rem !important; padding: 0.6rem 2rem !important; width: 100%;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODELS
@st.cache_resource
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    clf_path  = os.path.join(model_dir, "best_classifier.pkl")
    reg_path  = os.path.join(model_dir, "best_regressor.pkl")

    clf, reg = None, None
    if os.path.exists(clf_path):
        with open(clf_path, "rb") as f:
            clf = pickle.load(f)
    if os.path.exists(reg_path):
        with open(reg_path, "rb") as f:
            reg = pickle.load(f)
    return clf, reg

clf_model, reg_model = load_models()

# FEATURE ENGINEERING
FEATURE_COLS = [
    "ssc_percentage", "hsc_percentage", "degree_percentage", "cgpa",
    "entrance_exam_score", "technical_skill_score", "soft_skill_score",
    "internship_count", "live_projects", "work_experience_months",
    "certifications", "attendance_percentage", "backlogs",
    "academic_avg", "overall_skill", "experience_score",
    "high_cgpa", "clean_record", "gender_enc", "extra_enc",
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["academic_avg"]     = (df["ssc_percentage"] + df["hsc_percentage"] + df["degree_percentage"]) / 3
    df["overall_skill"]    = (df["technical_skill_score"] + df["soft_skill_score"]) / 2
    df["experience_score"] = (
        df["internship_count"]       * 10 +
        df["work_experience_months"] * 0.5 +
        df["live_projects"]          * 5 +
        df["certifications"]         * 3
    )
    df["high_cgpa"]    = (df["cgpa"] >= 8.0).astype(int)
    df["clean_record"] = (df["backlogs"] == 0).astype(int)
    df["gender_enc"]   = df["gender"].map({"Female": 0, "Male": 1})
    df["extra_enc"]    = df["extracurricular_activities"].map({"No": 0, "Yes": 1})
    return df

def predict_single(row: dict) -> dict:
    df = pd.DataFrame([row])
    df = engineer_features(df)
    X  = df[FEATURE_COLS]

    prob   = float(clf_model.predict_proba(X)[0][1])
    label  = int(clf_model.predict(X)[0])
    salary = None
    if label == 1 and reg_model:
        salary = float(reg_model.predict(X)[0])

    return {"placed": label, "placed_prob": prob, "salary": salary}
    
# CHART
def gauge_chart(prob: float) -> go.Figure:
    pct   = prob * 100
    color = "#34d399" if pct >= 50 else "#f87171"
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = pct,
        number = {"suffix": "%", "font": {"size": 40, "color": "#e8eaf0", "family": "Syne"}},
        gauge  = {
            "axis"       : {"range": [0, 100], "tickcolor": "#5a6a8a", "tickfont": {"color": "#5a6a8a"}},
            "bar"        : {"color": color, "thickness": 0.3},
            "bgcolor"    : "#111827",
            "bordercolor": "#1e2d4a",
            "steps": [
                {"range": [0,  40],  "color": "#1a0a0a"},
                {"range": [40, 60],  "color": "#1a1a0a"},
                {"range": [60, 100], "color": "#0a1a0a"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.75, "value": pct},
        },
        title = {"text": "Placement Probability", "font": {"color": "#5a6a8a", "size": 13}},
    ))
    fig.update_layout(
        height=260, margin=dict(t=40, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e8eaf0"
    )
    return fig

# SIDEBAR
with st.sidebar:
    st.markdown('<div class="brand-header">PlacementIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">ML Prediction Dashboard</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model Status**")
    clf_ok = clf_model is not None
    reg_ok = reg_model is not None
    st.markdown(f"{'🟢' if clf_ok else '🔴'} Classifier `{'Loaded' if clf_ok else 'Not Found'}`")
    st.markdown(f"{'🟢' if reg_ok else '🔴'} Regressor  `{'Loaded' if reg_ok else 'Not Found'}`")

    st.markdown("---")
    st.caption("Dataset: Student Placement (B.csv)")
    st.caption("Soal 3 — Monolithic Deployment")

# MAIN
st.markdown('<div class="brand-header">PlacementIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-sub">Student Placement & Salary Prediction</div>', unsafe_allow_html=True)

if not clf_ok:
    st.error("⚠️ Model belum ditemukan. Pastikan file `models/best_classifier.pkl` tersedia.")
    st.stop()

st.markdown('<div class="section-title">Input Data Mahasiswa</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Profil**")
        gender     = st.selectbox("Gender", ["Male", "Female"])
        extra      = st.selectbox("Ekstrakurikuler", ["Yes", "No"])
        backlogs   = st.number_input("Backlogs", min_value=0, max_value=10, value=0)
        attendance = st.slider("Attendance %", 60, 100, 85)

    with col2:
        st.markdown("**Akademik**")
        ssc      = st.slider("SSC %",    50, 100, 75)
        hsc      = st.slider("HSC %",    50, 100, 75)
        deg      = st.slider("Degree %", 55, 100, 75)
        cgpa     = st.slider("CGPA", 5.5, 10.0, 8.5, step=0.1)
        entrance = st.slider("Entrance Exam Score", 40, 100, 80)

    with col3:
        st.markdown("**Skill & Pengalaman**")
        tech_skill  = st.slider("Technical Skill Score", 40, 100, 85)
        soft_skill  = st.slider("Soft Skill Score",      40, 100, 78)
        internships = st.number_input("Internship Count",         min_value=0, max_value=10, value=2)
        projects    = st.number_input("Live Projects",            min_value=0, max_value=10, value=3)
        exp_months  = st.number_input("Work Experience (months)", min_value=0, max_value=60, value=6)
        certs       = st.number_input("Certifications",           min_value=0, max_value=20, value=3)

    submitted = st.form_submit_button("Prediksi Sekarang")

# =============================================================================
# HASIL
# =============================================================================

if submitted:
    row = {
        "gender"                    : gender,
        "ssc_percentage"            : ssc,
        "hsc_percentage"            : hsc,
        "degree_percentage"         : deg,
        "cgpa"                      : cgpa,
        "entrance_exam_score"       : entrance,
        "technical_skill_score"     : tech_skill,
        "soft_skill_score"          : soft_skill,
        "internship_count"          : internships,
        "live_projects"             : projects,
        "work_experience_months"    : exp_months,
        "certifications"            : certs,
        "attendance_percentage"     : attendance,
        "backlogs"                  : backlogs,
        "extracurricular_activities": extra,
    }

    result = predict_single(row)

    st.markdown("---")
    st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)

    # Banner
    if result["placed"] == 1:
        sal_text = f"Estimasi Salary: <b>{result['salary']:.2f} LPA</b>" if result["salary"] else ""
        st.markdown(f"""<div class="result-placed">
            <div class="result-title">✅ PLACED</div>
            <div class="result-subtitle">{sal_text}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="result-not-placed">
            <div class="result-title">❌ NOT PLACED</div>
            <div class="result-subtitle">Probabilitas Placed: {result['placed_prob']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Gauge + Metric cards
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(gauge_chart(result["placed_prob"]), use_container_width=True)
    with c2:
        st.markdown("")
        academic_avg  = (ssc + hsc + deg) / 3
        overall_skill = (tech_skill + soft_skill) / 2
        exp_score     = internships * 10 + exp_months * 0.5 + projects * 5 + certs * 3

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Academic Avg</div>
                <div class="metric-value blue">{academic_avg:.1f}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Overall Skill</div>
                <div class="metric-value purple">{overall_skill:.1f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        m3, m4 = st.columns(2)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Experience Score</div>
                <div class="metric-value orange">{exp_score:.1f}</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            color = "green" if result["placed"] == 1 else "orange"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Placed Probability</div>
                <div class="metric-value {color}">{result['placed_prob']*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
