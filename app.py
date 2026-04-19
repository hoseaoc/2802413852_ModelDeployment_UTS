import requests
import plotly.graph_objects as go
import streamlit as st

# CONFIG
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="PlacementIQ Client",
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



# HELPER
def call_api(endpoint: str, payload: dict) -> dict:
    resp = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()

def check_health() -> bool:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return resp.json().get("status") == "ok"
    except Exception:
        return False

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
    st.markdown('<div class="brand-sub">Decoupled Client</div>', unsafe_allow_html=True)

    st.markdown("---")
    API_BASE_URL = st.text_input("FastAPI Base URL", value=API_BASE_URL)

    if st.button("Cek Koneksi API"):
        if check_health():
            st.success("API Terhubung!")
        else:
            st.error("Tidak bisa terhubung ke API.")

    st.markdown("---")
    mode = st.radio(
        "Mode Prediksi",
        ["Klasifikasi", "Regresi", "Full (Klasifikasi + Regresi)"]
    )

    st.markdown("---")
    st.caption(f"Backend: {API_BASE_URL}")
    st.caption("Soal 4 — Decoupled Architecture")


# MAIN
st.markdown('<div class="brand-header">PlacementIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-sub">Student Placement & Salary Prediction — Decoupled</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Input Data Mahasiswa</div>', unsafe_allow_html=True)

with st.form("form_prediction"):
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

    submitted = st.form_submit_button("Kirim ke API & Prediksi")

# PAYLOAD & HASIL

payload = {
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

if submitted:
    st.markdown("---")
    st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)

    endpoint = {
        "Klasifikasi"                 : "/predict/classification",
        "Regresi"                     : "/predict/regression",
        "Full (Klasifikasi + Regresi)": "/predict/full",
    }[mode]
    st.caption(f"POST {API_BASE_URL}{endpoint}")

    try:
        result = call_api(endpoint, payload)

        # KLASIFIKASI 
        if mode == "Klasifikasi":
            prob  = result["placement_probability"]
            label = result["placement_status"]

            if label == 1:
                st.markdown(f"""<div class="result-placed">
                    <div class="result-title">✅ PLACED</div>
                    <div class="result-subtitle">Probabilitas: {prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-not-placed">
                    <div class="result-title">❌ NOT PLACED</div>
                    <div class="result-subtitle">Probabilitas Placed: {prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(gauge_chart(prob), use_container_width=True)
            with c2:
                st.markdown("**Raw API Response**")
                st.json(result)

        # REGRESI
        elif mode == "Regresi":
            salary = result["predicted_salary_lpa"]

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Predicted Salary</div>
                    <div class="metric-value orange">{salary:.2f} LPA</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Per Bulan (Est.)</div>
                    <div class="metric-value blue">{salary/12:.2f} LPA</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Range Est.</div>
                    <div class="metric-value purple">{salary*0.9:.1f} – {salary*1.1:.1f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            st.info(f"📝 {result['note']}")
            st.markdown("**Raw API Response**")
            st.json(result)

        # FULL 
        else:
            prob   = result["placement_probability"]
            label  = result["placement_status"]
            salary = result["predicted_salary_lpa"]

            if label == 1:
                sal_text = f"Estimasi Salary: <b>{salary:.2f} LPA</b>" if salary else ""
                st.markdown(f"""<div class="result-placed">
                    <div class="result-title">✅ PLACED</div>
                    <div class="result-subtitle">{sal_text}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-not-placed">
                    <div class="result-title">❌ NOT PLACED</div>
                    <div class="result-subtitle">Probabilitas Placed: {prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Status</div>
                    <div class="metric-value {'green' if label==1 else 'orange'}">{result['placement_label']}</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Probability</div>
                    <div class="metric-value blue">{prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                sal_display = f"{salary:.2f}" if salary else "N/A"
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Est. Salary (LPA)</div>
                    <div class="metric-value purple">{sal_display}</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">CGPA</div>
                    <div class="metric-value orange">{cgpa}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(gauge_chart(prob), use_container_width=True)
            with c2:
                st.markdown("**Raw API Response**")
                st.json(result)

        with st.expander("Lihat Payload yang Dikirim ke API"):
            st.json(payload)

    except requests.exceptions.ConnectionError:
        st.error(f"Tidak bisa terhubung ke API. Pastikan FastAPI berjalan di: {API_BASE_URL}")
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
