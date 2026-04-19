"""
app.py
======
Monolithic Streamlit App — Placement & Salary Prediction
Muat model .pkl dari pipeline dan prediksi via:
  1. Form manual (satu mahasiswa)
  2. Upload CSV (batch prediksi)
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "PlacementIQ",
    page_icon  = "🎓",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1525 !important;
    border-right: 1px solid #1e2a45;
}
[data-testid="stSidebar"] * {
    color: #c8cfe0 !important;
}

/* Header brand */
.brand-header {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #4f8ef7 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.brand-sub {
    font-size: 0.85rem;
    color: #5a6a8a;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #4f8ef7; }
.metric-label {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5a6a8a;
    font-weight: 500;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e8eaf0;
    line-height: 1;
}
.metric-value.green  { color: #34d399; }
.metric-value.blue   { color: #60a5fa; }
.metric-value.purple { color: #a78bfa; }
.metric-value.orange { color: #fb923c; }

/* Result banner */
.result-placed {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #34d399;
    border-radius: 16px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-not-placed {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #f87171;
    border-radius: 16px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-subtitle { font-size: 0.9rem; color: #94a3b8; }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8eaf0;
    letter-spacing: 0.5px;
    margin-bottom: 0.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2a45;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e2a45;
    margin: 1.5rem 0;
}

/* Form fields */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    background: #111827 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #a78bfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Tabs */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: #5a6a8a !important;
    border-bottom: 2px solid transparent;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #4f8ef7 !important;
    border-bottom: 2px solid #4f8ef7;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Upload area */
[data-testid="stFileUploader"] {
    background: #111827;
    border: 1.5px dashed #1e2d4a;
    border-radius: 12px;
    padding: 1rem;
}

/* Info / warning boxes */
.stAlert { border-radius: 10px !important; }

/* Slider */
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, #4f8ef7, #a78bfa) !important;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-placed     { background: #064e3b; color: #34d399; border: 1px solid #34d399; }
.badge-not-placed { background: #450a0a; color: #f87171; border: 1px solid #f87171; }
</style>
""", unsafe_allow_html=True)



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
    gender_map = {"Female": 0, "Male": 1}
    extra_map  = {"No": 0, "Yes": 1}
    df["gender_enc"] = df["gender"].map(gender_map)
    df["extra_enc"]  = df["extracurricular_activities"].map(extra_map)
    return df

FEATURE_COLS = [
    "ssc_percentage", "hsc_percentage", "degree_percentage", "cgpa",
    "entrance_exam_score", "technical_skill_score", "soft_skill_score",
    "internship_count", "live_projects", "work_experience_months",
    "certifications", "attendance_percentage", "backlogs",
    "academic_avg", "overall_skill", "experience_score",
    "high_cgpa", "clean_record", "gender_enc", "extra_enc",
]

RAW_COLS = [
    "gender", "ssc_percentage", "hsc_percentage", "degree_percentage",
    "cgpa", "entrance_exam_score", "technical_skill_score", "soft_skill_score",
    "internship_count", "live_projects", "work_experience_months",
    "certifications", "attendance_percentage", "backlogs",
    "extracurricular_activities",
]

def predict_single(row: dict) -> dict:
    df = pd.DataFrame([row])
    df = engineer_features(df)
    X  = df[FEATURE_COLS]

    placed_prob = clf_model.predict_proba(X)[0][1]
    placed      = int(clf_model.predict(X)[0])

    salary = None
    if placed and reg_model:
        salary = float(reg_model.predict(X)[0])

    return {"placed": placed, "placed_prob": placed_prob, "salary": salary}


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = engineer_features(df)
    X     = df_fe[FEATURE_COLS]

    probs   = clf_model.predict_proba(X)[:, 1]
    labels  = clf_model.predict(X)

    salaries = []
    for i, lbl in enumerate(labels):
        if lbl == 1 and reg_model:
            sal = float(reg_model.predict(X.iloc[[i]])[0])
        else:
            sal = None
        salaries.append(sal)

    out = df.copy()
    out["placement_prediction"] = ["✅ Placed" if l == 1 else "❌ Not Placed" for l in labels]
    out["placement_probability"] = (probs * 100).round(1).astype(str) + "%"
    out["predicted_salary_lpa"]  = [f"{s:.2f}" if s else "-" for s in salaries]
    return out



def gauge_chart(prob: float) -> go.Figure:
    pct = prob * 100
    color = "#34d399" if pct >= 50 else "#f87171"
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = pct,
        number = {"suffix": "%", "font": {"size": 40, "color": "#e8eaf0", "family": "Syne"}},
        gauge = {
            "axis": {"range": [0, 100], "tickcolor": "#5a6a8a", "tickfont": {"color": "#5a6a8a"}},
            "bar" : {"color": color, "thickness": 0.3},
            "bgcolor"     : "#111827",
            "bordercolor" : "#1e2d4a",
            "steps": [
                {"range": [0,  40],  "color": "#1a0a0a"},
                {"range": [40, 60],  "color": "#1a1a0a"},
                {"range": [60, 100], "color": "#0a1a0a"},
            ],
            "threshold": {
                "line" : {"color": color, "width": 4},
                "thickness": 0.75,
                "value": pct,
            },
        },
        title = {"text": "Placement Probability", "font": {"color": "#5a6a8a", "size": 13}},
    ))
    fig.update_layout(
        height    = 260,
        margin    = dict(t=40, b=10, l=30, r=30),
        paper_bgcolor = "rgba(0,0,0,0)",
        font_color    = "#e8eaf0",
    )
    return fig


def radar_chart(row: dict) -> go.Figure:
    categories = ["SSC %", "HSC %", "Degree %", "CGPA×10", "Technical", "Soft Skill", "Attendance"]
    values = [
        row["ssc_percentage"],
        row["hsc_percentage"],
        row["degree_percentage"],
        row["cgpa"] * 10,
        row["technical_skill_score"],
        row["soft_skill_score"],
        row["attendance_percentage"],
    ]
    fig = go.Figure(go.Scatterpolar(
        r      = values + [values[0]],
        theta  = categories + [categories[0]],
        fill   = "toself",
        fillcolor = "rgba(79, 142, 247, 0.15)",
        line  = dict(color="#4f8ef7", width=2),
        name  = "Student Profile",
    ))
    fig.update_layout(
        polar = dict(
            bgcolor   = "#0f1525",
            radialaxis = dict(visible=True, range=[0, 100], tickfont=dict(color="#5a6a8a", size=9), gridcolor="#1e2a45"),
            angularaxis = dict(tickfont=dict(color="#c8cfe0", size=11), gridcolor="#1e2a45"),
        ),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        showlegend    = False,
        height        = 300,
        margin        = dict(t=30, b=30, l=40, r=40),
    )
    return fig


# SIDEBAR

with st.sidebar:
    st.markdown('<div class="brand-header">PlacementIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">ML Prediction Dashboard</div>', unsafe_allow_html=True)

    st.markdown("---")
    mode = st.radio(
        "**Mode Prediksi**",
        ["📝 Form Manual", "📂 Upload CSV"],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown("**Model Status**")
    clf_ok = clf_model is not None
    reg_ok = reg_model is not None
    st.markdown(f"{'🟢' if clf_ok else '🔴'} Classifier  `{'Loaded' if clf_ok else 'Not Found'}`")
    st.markdown(f"{'🟢' if reg_ok else '🔴'} Regressor  `{'Loaded' if reg_ok else 'Not Found'}`")

    st.markdown("---")
    st.caption("Dataset: Student Placement (B.csv)")
    st.caption("Pipeline: sklearn + MLflow")


# MAIN CONTENT

st.markdown('<div class="brand-header">PlacementIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-sub">Student Placement & Salary Prediction</div>', unsafe_allow_html=True)

if not clf_ok:
    st.error("⚠️ Model belum ditemukan. Pastikan file `models/best_classifier.pkl` tersedia.")
    st.stop()


# MODE 1: FORM MANUAL
if mode == "Form Manual":

    st.markdown('<div class="section-title">Input Data Mahasiswa</div>', unsafe_allow_html=True)
    st.markdown("")

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Profil**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            extra  = st.selectbox("Ekstrakurikuler", ["Yes", "No"])
            backlogs = st.number_input("Backlogs", min_value=0, max_value=10, value=0)
            attendance = st.slider("Attendance %", 60, 100, 85)

        with col2:
            st.markdown("**📚 Akademik**")
            ssc  = st.slider("SSC %",    50, 100, 75)
            hsc  = st.slider("HSC %",    50, 100, 75)
            deg  = st.slider("Degree %", 55, 100, 75)
            cgpa = st.slider("CGPA", 5.5, 10.0, 7.5, step=0.1)
            entrance = st.slider("Entrance Exam Score", 40, 100, 70)

        with col3:
            st.markdown("**💼 Skill & Pengalaman**")
            tech_skill  = st.slider("Technical Skill Score", 40, 100, 70)
            soft_skill  = st.slider("Soft Skill Score",      40, 100, 70)
            internships = st.number_input("Internship Count",          min_value=0, max_value=10, value=1)
            projects    = st.number_input("Live Projects",             min_value=0, max_value=10, value=1)
            exp_months  = st.number_input("Work Experience (months)",  min_value=0, max_value=60,  value=6)
            certs       = st.number_input("Certifications",            min_value=0, max_value=20,  value=2)

        submitted = st.form_submit_button("🔍 Prediksi Sekarang")

    if submitted:
        row = {
            "gender": gender, "ssc_percentage": ssc, "hsc_percentage": hsc,
            "degree_percentage": deg, "cgpa": cgpa, "entrance_exam_score": entrance,
            "technical_skill_score": tech_skill, "soft_skill_score": soft_skill,
            "internship_count": internships, "live_projects": projects,
            "work_experience_months": exp_months, "certifications": certs,
            "attendance_percentage": attendance, "backlogs": backlogs,
            "extracurricular_activities": extra,
        }

        result = predict_single(row)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)
        st.markdown("")

        # Result banner
        if result["placed"] == 1:
            sal_text = f"Estimasi Salary: <b>{result['salary']:.2f} LPA</b>" if result["salary"] else ""
            st.markdown(f"""
            <div class="result-placed">
                <div class="result-title">✅ PLACED</div>
                <div class="result-subtitle">{sal_text}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-not-placed">
                <div class="result-title">❌ NOT PLACED</div>
                <div class="result-subtitle">Probabilitas placement di bawah threshold</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Gauge + Radar
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(gauge_chart(result["placed_prob"]), use_container_width=True)
        with c2:
            st.plotly_chart(radar_chart(row), use_container_width=True)

        # Metric cards
        st.markdown("")
        m1, m2, m3, m4 = st.columns(4)
        academic_avg  = (ssc + hsc + deg) / 3
        overall_skill = (tech_skill + soft_skill) / 2
        exp_score     = internships * 10 + exp_months * 0.5 + projects * 5 + certs * 3

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
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Experience Score</div>
                <div class="metric-value orange">{exp_score:.1f}</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            color = "green" if result["placed"] == 1 else "orange"
            prob_display = f"{result['placed_prob']*100:.1f}%"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Placed Probability</div>
                <div class="metric-value {color}">{prob_display}</div>
            </div>""", unsafe_allow_html=True)


# MODE 2: UPLOAD CSV
else:
    st.markdown('<div class="section-title">Upload Dataset untuk Prediksi Batch</div>', unsafe_allow_html=True)
    st.markdown("")

    # Template download
    template_df = pd.DataFrame(columns=RAW_COLS)
    template_df.loc[0] = ["Male", 75, 72, 70, 7.5, 75, 78, 72, 2, 2, 6, 3, 88, 0, "Yes"]
    template_csv = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Template CSV",
        data      = template_csv,
        file_name = "template_input.csv",
        mime      = "text/csv",
    )

    st.markdown("")
    uploaded = st.file_uploader("Upload CSV (format sama seperti template)", type=["csv"])

    if uploaded:
        try:
            df_input = pd.read_csv(uploaded)

            # Validasi kolom
            missing = [c for c in RAW_COLS if c not in df_input.columns]
            if missing:
                st.error(f"Kolom tidak lengkap: {missing}")
                st.stop()

            st.markdown('<div class="section-title">Preview Data Input</div>', unsafe_allow_html=True)
            st.dataframe(df_input.head(10), use_container_width=True)

            if st.button("🔍 Jalankan Prediksi Batch"):
                with st.spinner("Memproses prediksi..."):
                    result_df = predict_batch(df_input)

                st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)
                st.markdown("")

                # Summary metrics
                n_total  = len(result_df)
                n_placed = (result_df["placement_prediction"].str.contains("Placed") &
                            ~result_df["placement_prediction"].str.contains("Not")).sum()
                n_not    = n_total - n_placed
                pct      = n_placed / n_total * 100

                placed_sal = result_df[result_df["predicted_salary_lpa"] != "-"]["predicted_salary_lpa"].astype(float)
                avg_sal = placed_sal.mean() if len(placed_sal) > 0 else 0

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Total Mahasiswa</div>
                        <div class="metric-value blue">{n_total}</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Predicted Placed</div>
                        <div class="metric-value green">{n_placed}</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Placement Rate</div>
                        <div class="metric-value purple">{pct:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Avg Salary (LPA)</div>
                        <div class="metric-value orange">{avg_sal:.2f}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("")

                # Visualisasi
                c1, c2 = st.columns(2)

                with c1:
                    fig_pie = go.Figure(go.Pie(
                        labels = ["Placed", "Not Placed"],
                        values = [n_placed, n_not],
                        hole   = 0.55,
                        marker = dict(colors=["#34d399", "#f87171"], line=dict(color="#0a0e1a", width=2)),
                        textfont = dict(color="#e8eaf0"),
                    ))
                    fig_pie.update_layout(
                        title  = dict(text="Placement Distribution", font=dict(color="#e8eaf0", family="Syne", size=14)),
                        paper_bgcolor = "rgba(0,0,0,0)",
                        legend = dict(font=dict(color="#c8cfe0")),
                        height = 300,
                        margin = dict(t=50, b=10),
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with c2:
                    if len(placed_sal) > 0:
                        fig_hist = go.Figure(go.Histogram(
                            x         = placed_sal,
                            nbinsx    = 20,
                            marker_color = "#4f8ef7",
                            opacity   = 0.85,
                        ))
                        fig_hist.update_layout(
                            title  = dict(text="Distribusi Predicted Salary (LPA)", font=dict(color="#e8eaf0", family="Syne", size=14)),
                            xaxis  = dict(title="Salary (LPA)", color="#5a6a8a", gridcolor="#1e2a45"),
                            yaxis  = dict(title="Count", color="#5a6a8a", gridcolor="#1e2a45"),
                            paper_bgcolor = "rgba(0,0,0,0)",
                            plot_bgcolor  = "rgba(0,0,0,0)",
                            height = 300,
                            margin = dict(t=50, b=10),
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                # Tabel hasil
                st.markdown("")
                st.markdown('<div class="section-title">Tabel Hasil Lengkap</div>', unsafe_allow_html=True)
                display_cols = list(df_input.columns) + [
                    "placement_prediction", "placement_probability", "predicted_salary_lpa"
                ]
                st.dataframe(result_df[display_cols], use_container_width=True)

                # Download hasil
                csv_out = result_df[display_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Hasil Prediksi (CSV)",
                    data      = csv_out,
                    file_name = "hasil_prediksi.csv",
                    mime      = "text/csv",
                )

        except Exception as e:
            st.error(f"Error memproses file: {e}")
