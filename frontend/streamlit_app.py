"""
Streamlit frontend — Salary Prediction App
Calls the Flask API at http://127.0.0.1:5000
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API_BASE = "https://salary-prediction-api2.onrender.com"

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(90deg, #3b82d4, #7c5cd8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #57606a; font-size: 1rem; margin-bottom: 1.5rem; }
    .result-card {
        background: #f0f4ff; border-radius: 12px; padding: 2rem;
        border-left: 5px solid #3b82d4; text-align: center;
    }
    .salary-amount {
        font-size: 3rem; font-weight: 800; color: #3b82d4;
    }
    .metric-box {
        background: white; border-radius: 8px; padding: 1rem;
        border: 1px solid #e5e7eb; text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── helper: fetch meta from API ───────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_meta():
    try:
        r = requests.get(f"{API_BASE}/meta", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"⚠️ Cannot reach API at {API_BASE}. Start the Flask server first.\n\n{e}")
        st.stop()


# ── helper: call predict endpoint ─────────────────────────────────────────────
def predict_salary(payload: dict) -> dict:
    r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


# ── helper: build salary-vs-experience chart ──────────────────────────────────
def build_exp_chart(meta: dict, base_payload: dict) -> go.Figure:
    experiences = list(range(meta["Experience_min"], meta["Experience_max"] + 1))
    salaries = []
    for exp in experiences:
        p = {**base_payload, "Experience": exp}
        try:
            resp = requests.post(f"{API_BASE}/predict", json=p, timeout=5)
            salaries.append(resp.json().get("predicted_salary", 0))
        except Exception:
            salaries.append(0)

    df = pd.DataFrame({"Experience (Years)": experiences, "Predicted Salary ($)": salaries})
    fig = px.line(
        df,
        x="Experience (Years)",
        y="Predicted Salary ($)",
        title="Salary Growth vs Experience",
        markers=True,
        color_discrete_sequence=["#3b82d4"],
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Segoe UI, sans-serif", size=13),
        title_font_size=16,
        hovermode="x unified",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
    )
    # highlight selected experience
    sel_exp = base_payload.get("Experience", 0)
    if sel_exp in experiences:
        idx = experiences.index(sel_exp)
        fig.add_vline(
            x=sel_exp,
            line_dash="dash",
            line_color="#7c5cd8",
            annotation_text=f"Your: {sel_exp} yrs",
            annotation_position="top right",
        )
    return fig


def build_comparison_chart(meta: dict, base_payload: dict) -> go.Figure:
    """Bar chart comparing salary across all job titles for current profile."""
    titles = meta["Job_Title"]
    salaries = []
    for title in titles:
        p = {**base_payload, "Job_Title": title}
        try:
            resp = requests.post(f"{API_BASE}/predict", json=p, timeout=5)
            salaries.append(resp.json().get("predicted_salary", 0))
        except Exception:
            salaries.append(0)

    df = pd.DataFrame({"Job Title": titles, "Predicted Salary ($)": salaries}).sort_values(
        "Predicted Salary ($)", ascending=True
    )
    colors = [
        "#7c5cd8" if t == base_payload["Job_Title"] else "#3b82d4" for t in df["Job Title"]
    ]
    fig = go.Figure(
        go.Bar(
            x=df["Predicted Salary ($)"],
            y=df["Job Title"],
            orientation="h",
            marker_color=colors,
            text=[f"${s:,.0f}" for s in df["Predicted Salary ($)"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Salary by Job Title (your profile)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Segoe UI, sans-serif", size=13),
        xaxis=dict(tickprefix="$", tickformat=",.0f", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=False),
        title_font_size=16,
        margin=dict(l=10, r=80, t=50, b=10),
    )
    return fig


def build_education_chart(meta: dict, base_payload: dict) -> go.Figure:
    """Grouped bar: salary across education levels for each gender."""
    edu_levels = meta["Education"]
    genders = meta["Gender"]

    data = {}
    for gender in genders:
        data[gender] = []
        for edu in edu_levels:
            p = {**base_payload, "Education": edu, "Gender": gender}
            try:
                resp = requests.post(f"{API_BASE}/predict", json=p, timeout=5)
                data[gender].append(resp.json().get("predicted_salary", 0))
            except Exception:
                data[gender].append(0)

    colors = {"Male": "#3b82d4", "Female": "#7c5cd8"}
    fig = go.Figure()
    for gender in genders:
        fig.add_trace(
            go.Bar(
                name=gender,
                x=edu_levels,
                y=data[gender],
                marker_color=colors.get(gender, "#888"),
                text=[f"${s:,.0f}" for s in data[gender]],
                textposition="outside",
            )
        )
    fig.update_layout(
        title="Salary by Education & Gender (your profile)",
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Segoe UI, sans-serif", size=13),
        yaxis=dict(tickprefix="$", tickformat=",.0f", showgrid=True, gridcolor="#f0f0f0"),
        title_font_size=16,
        legend_title="Gender",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
meta = fetch_meta()

# ── header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">💼 Employee Salary Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Predict the expected salary of a new employee based on profile & experience using Machine Learning.</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── sidebar: inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧑‍💼 Employee Profile")
    st.caption("Fill in the candidate details below.")

    experience = st.slider(
        "Years of Experience",
        min_value=meta["Experience_min"],
        max_value=meta["Experience_max"],
        value=5,
        step=1,
        help="Total years of professional work experience",
    )

    age = st.slider(
        "Age",
        min_value=meta["Age_min"],
        max_value=meta["Age_max"],
        value=28,
        step=1,
    )

    education = st.selectbox("Education Level", meta["Education"])
    job_title = st.selectbox("Job Title", meta["Job_Title"])
    location = st.selectbox("Location", meta["Location"])
    gender = st.selectbox("Gender", meta["Gender"])

    st.divider()
    predict_btn = st.button("🔮 Predict Salary", use_container_width=True, type="primary")

# ── main panel ────────────────────────────────────────────────────────────────
payload = {
    "Education": education,
    "Experience": experience,
    "Location": location,
    "Job_Title": job_title,
    "Age": age,
    "Gender": gender,
}

# Auto-predict on load; also trigger on button press
if "last_payload" not in st.session_state or st.session_state.last_payload != payload or predict_btn:
    st.session_state.last_payload = payload
    with st.spinner("Predicting salary…"):
        try:
            result = predict_salary(payload)
            st.session_state.result = result
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

result = st.session_state.get("result", None)

if result:
    salary = result["predicted_salary"]

    # ── result card ──
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(
            f"""
            <div class="result-card">
                <p style="color:#57606a; margin:0; font-size:1rem;">Predicted Annual Salary</p>
                <p class="salary-amount">${salary:,.0f}</p>
                <p style="color:#57606a; margin:0; font-size:0.85rem;">USD / year</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        monthly = salary / 12
        st.metric("Monthly Salary", f"${monthly:,.0f}")
        st.metric("Weekly Salary", f"${salary/52:,.0f}")

    with col3:
        st.metric("Daily (260 work days)", f"${salary/260:,.0f}")
        st.metric("Hourly (~2080 hrs)", f"${salary/2080:,.1f}")

    st.divider()

    # ── tabs: charts ──
    tab1, tab2, tab3 = st.tabs(
        ["📈 Experience Growth", "📊 Job Title Comparison", "🎓 Education & Gender"]
    )

    with tab1:
        with st.spinner("Building experience chart…"):
            fig_exp = build_exp_chart(meta, payload)
        st.plotly_chart(fig_exp, use_container_width=True)
        st.caption(
            "Holds all other factors constant (education, location, job title, gender) "
            "and sweeps experience from min to max."
        )

    with tab2:
        with st.spinner("Building comparison chart…"):
            fig_cmp = build_comparison_chart(meta, payload)
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.caption(
            "Purple bar = your selected job title. All other factors remain the same."
        )

    with tab3:
        with st.spinner("Building education/gender chart…"):
            fig_edu = build_education_chart(meta, payload)
        st.plotly_chart(fig_edu, use_container_width=True)

    st.divider()

    # ── input summary table ──
    with st.expander("📋 Input Summary", expanded=False):
        summary_df = pd.DataFrame(
            list(result["inputs"].items()), columns=["Feature", "Value"]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<br><hr><p style='text-align:center;color:#57606a;font-size:0.8rem;'>"
    "Salary Predictor · Gradient Boosting Model · Trained on salary_data.csv"
    "</p>",
    unsafe_allow_html=True,
)
