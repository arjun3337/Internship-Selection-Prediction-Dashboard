
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Employability Insights",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Employability Insights")
st.caption("Explore applicant skills, academic performance, and employability indicators.")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Internship_Selection_Dataset.csv"

@st.cache_data
def load_data(path, modified_time):
    return pd.read_csv(path)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)

if "selected" not in df.columns:
    st.error("The dataset must contain a 'selected' column.")
    st.stop()

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.subheader("Insight Filters")

filtered_df = df.copy()

if "college_tier" in df.columns:
    tiers = sorted(df["college_tier"].dropna().unique().tolist())
    selected_tiers = st.sidebar.multiselect(
        "College Tier",
        tiers,
        default=tiers
    )
    filtered_df = filtered_df[
        filtered_df["college_tier"].isin(selected_tiers)
    ]

if "extracurricular" in df.columns:
    activities = sorted(
        df["extracurricular"].dropna().unique().tolist(),
        key=str
    )
    selected_activities = st.sidebar.multiselect(
        "Extracurricular",
        activities,
        default=activities
    )
    filtered_df = filtered_df[
        filtered_df["extracurricular"].isin(selected_activities)
    ]

if "placement_training" in df.columns:
    training_options = sorted(
        df["placement_training"].dropna().unique().tolist(),
        key=str
    )
    selected_training = st.sidebar.multiselect(
        "Placement Training",
        training_options,
        default=training_options
    )
    filtered_df = filtered_df[
        filtered_df["placement_training"].isin(selected_training)
    ]

if filtered_df.empty:
    st.warning("No applicants match the selected filters.")
    st.stop()

# --------------------------------------------------
# KPI metrics
# --------------------------------------------------
st.subheader("📊 Employability Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Applicants Analyzed", f"{len(filtered_df):,}")

if "CGPA" in filtered_df.columns:
    c2.metric(
        "Average CGPA",
        f"{filtered_df['CGPA'].mean():.2f}"
    )
else:
    c2.metric("Average CGPA", "N/A")

if "skills_score" in filtered_df.columns:
    c3.metric(
        "Average Skills Score",
        f"{filtered_df['skills_score'].mean():.2f}"
    )
else:
    c3.metric("Average Skills Score", "N/A")

if "coding_test_score" in filtered_df.columns:
    c4.metric(
        "Average Coding Score",
        f"{filtered_df['coding_test_score'].mean():.2f}"
    )
else:
    c4.metric("Average Coding Score", "N/A")

st.divider()

# --------------------------------------------------
# Skill analysis
# --------------------------------------------------
st.subheader("🧠 Skill Performance Analysis")

skill_candidates = [
    "skills_score",
    "coding_test_score",
    "communication_score",
    "soft_skills_score",
    "aptitude_score",
    "resume_score",
    "github_score",
    "linkedin_activity_score",
    "consistency_score"
]

skill_columns = [
    col for col in skill_candidates
    if col in filtered_df.columns
    and pd.api.types.is_numeric_dtype(filtered_df[col])
]

if skill_columns:
    skill_means = (
        filtered_df[skill_columns]
        .mean()
        .rename_axis("Skill")
        .reset_index(name="Average Score")
    )

    skill_means["Skill"] = (
        skill_means["Skill"]
        .str.replace("_", " ")
        .str.title()
    )

    fig = px.bar(
        skill_means.sort_values("Average Score"),
        x="Average Score",
        y="Skill",
        orientation="h",
        title="Average Applicant Scores by Skill",
        color="Average Score",
        color_continuous_scale="Blues",
        template="plotly_white"
    )

    fig.update_layout(
        height=450,
        yaxis_title="",
        xaxis_title="Average Score",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No numeric skill columns were found.")

st.divider()

# --------------------------------------------------
# Academic performance vs selection
# --------------------------------------------------
st.subheader("🎓 Academic Performance and Selection")

if "CGPA" in filtered_df.columns:
    cgpa_data = (
        filtered_df.groupby("selected", dropna=False)["CGPA"]
        .mean()
        .reset_index()
    )

    cgpa_data["Outcome"] = cgpa_data["selected"].map({
        0: "Not Selected",
        1: "Selected"
    }).fillna(cgpa_data["selected"].astype(str))

    fig_cgpa = px.bar(
        cgpa_data,
        x="Outcome",
        y="CGPA",
        color="Outcome",
        text_auto=".2f",
        title="Average CGPA by Recorded Selection Outcome",
        template="plotly_white"
    )

    fig_cgpa.update_layout(
        showlegend=False,
        yaxis_title="Average CGPA",
        xaxis_title=""
    )

    st.plotly_chart(fig_cgpa, use_container_width=True)

# --------------------------------------------------
# Strengths and development areas
# --------------------------------------------------
st.subheader("💡 Applicant Development Insights")

if skill_columns:
    average_scores = filtered_df[skill_columns].mean().dropna()

    if not average_scores.empty:
        strongest = average_scores.idxmax()
        improvement = average_scores.idxmin()

        left, right = st.columns(2)

        with left:
            st.success(
                f"**Highest average indicator:** "
                f"{strongest.replace('_', ' ').title()} "
                f"({average_scores[strongest]:.2f})"
            )

        with right:
            st.info(
                f"**Lowest average indicator:** "
                f"{improvement.replace('_', ' ').title()} "
                f"({average_scores[improvement]:.2f})"
            )

        st.caption(
            "These are relative averages in the filtered dataset, "
            "not validated measures of employability or individual ability."
        )

# --------------------------------------------------
# Download report
# --------------------------------------------------
st.divider()
st.subheader("📥 Export Insights")

report = filtered_df.copy()

csv_data = report.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Applicant Data (CSV)",
    data=csv_data,
    file_name="employability_insights.csv",
    mime="text/csv"
)

st.caption(
    "Educational analysis using a synthetic dataset. "
    "These indicators should not be used as definitive hiring criteria."
)
