
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Executive Overview",
    page_icon="🎓",
    layout="wide",
)

# =========================================================
# 1. PATHS AND DATA
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "Internship_Selection_Dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

@st.cache_data
def load_data(path, modified_time):
    data = pd.read_csv(path)
    data.columns = data.columns.astype(str).str.strip()

    if "selected" not in data.columns:
        raise ValueError("Dataset must contain a 'selected' column.")

    data["selected"] = pd.to_numeric(data["selected"], errors="coerce")
    data = data.dropna(subset=["selected"]).copy()

    if not set(data["selected"].unique()).issubset({0, 1}):
        raise ValueError("'selected' must contain only 0 and 1.")

    data["selected"] = data["selected"].astype(int)
    return data

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()

# =========================================================
# 2. HEADER
# =========================================================
st.title("🎓 Internship Selection Analytics")
st.markdown("### Executive Overview")
st.caption(
    "Applicant statistics, selection patterns, assessment insights, "
    "and machine-learning evaluation."
)

st.info(
    "This dashboard uses a synthetic dataset for educational analysis. "
    "It is not a validated recruitment decision system."
)

# =========================================================
# 3. SIDEBAR FILTERS
# =========================================================
st.sidebar.header("Dashboard Filters")

filtered_df = df.copy()

if "college_tier" in filtered_df.columns:
    tiers = sorted(
        filtered_df["college_tier"].dropna().astype(str).unique()
    )
    selected_tiers = st.sidebar.multiselect(
        "College Tier",
        tiers,
        default=tiers,
    )

    filtered_df = filtered_df[
        filtered_df["college_tier"].astype(str).isin(selected_tiers)
    ]

if "extracurricular" in filtered_df.columns:
    activities = sorted(
        filtered_df["extracurricular"].dropna().astype(str).unique()
    )
    selected_activities = st.sidebar.multiselect(
        "Extracurricular Activities",
        activities,
        default=activities,
    )

    filtered_df = filtered_df[
        filtered_df["extracurricular"].astype(str).isin(
            selected_activities
        )
    ]

if "placement_training" in filtered_df.columns:
    training_options = sorted(
        filtered_df["placement_training"].dropna().astype(str).unique()
    )
    selected_training = st.sidebar.multiselect(
        "Placement Training",
        training_options,
        default=training_options,
    )

    filtered_df = filtered_df[
        filtered_df["placement_training"].astype(str).isin(
            selected_training
        )
    ]

st.sidebar.caption(
    f"{len(filtered_df):,} of {len(df):,} records match the filters."
)

# =========================================================
# 4. KPI CARDS
# =========================================================
st.subheader("📌 Key Performance Indicators")

total = len(filtered_df)
selected_count = int((filtered_df["selected"] == 1).sum())
not_selected_count = int((filtered_df["selected"] == 0).sum())
selection_rate = selected_count / total * 100 if total else 0

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Applicants", f"{total:,}")
k2.metric("Selected", f"{selected_count:,}")
k3.metric("Not Selected", f"{not_selected_count:,}")
k4.metric("Selection Rate", f"{selection_rate:.2f}%")

st.divider()

# =========================================================
# 5. SELECTION DISTRIBUTION
# =========================================================
st.subheader("📊 Selection Overview")

left, right = st.columns(2)

with left:
    selection_df = pd.DataFrame({
        "Selection Status": ["Selected", "Not Selected"],
        "Applicants": [selected_count, not_selected_count],
    })

    if total:
        fig = px.pie(
            selection_df,
            names="Selection Status",
            values="Applicants",
            hole=0.58,
            title="Applicant Outcome Distribution",
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )
        fig.update_layout(
            legend_title_text="Outcome",
            margin=dict(t=60, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No applicants match the selected filters.")

with right:
    if total:
        outcome_counts = selection_df.copy()

        fig = px.bar(
            outcome_counts,
            x="Selection Status",
            y="Applicants",
            text="Applicants",
            color="Selection Status",
            title="Applicant Counts by Outcome",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False,
            yaxis_title="Number of applicants",
            xaxis_title="",
            margin=dict(t=60, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for this filter combination.")

st.divider()

# =========================================================
# 6. ACADEMIC PERFORMANCE
# =========================================================
st.subheader("🎓 Academic Performance")

if "CGPA" in filtered_df.columns and total:
    cgpa_data = filtered_df.copy()
    cgpa_data["CGPA"] = pd.to_numeric(
        cgpa_data["CGPA"], errors="coerce"
    )

    cgpa_summary = (
        cgpa_data.groupby("selected")["CGPA"]
        .mean()
        .reindex([0, 1])
        .reset_index()
    )

    cgpa_summary["Selection Status"] = cgpa_summary["selected"].map({
        0: "Not Selected",
        1: "Selected",
    })

    cgpa_summary = cgpa_summary.dropna(subset=["CGPA"])

    if not cgpa_summary.empty:
        fig = px.bar(
            cgpa_summary,
            x="Selection Status",
            y="CGPA",
            color="Selection Status",
            text= cgpa_summary["CGPA"].map(lambda x: f"{x:.2f}"),
            title="Average CGPA by Selection Outcome",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False,
            yaxis_title="Average CGPA",
            xaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("CGPA values are unavailable for the selected records.")
else:
    st.info("CGPA column is unavailable or no records match the filters.")

# =========================================================
# 7. SKILLS AND ASSESSMENT
# =========================================================
st.subheader("💡 Skills & Assessment Insights")

skill_features = [
    "skills_score",
    "coding_test_score",
    "communication_score",
    "interview_score",
    "aptitude_score",
    "soft_skills_score",
    "resume_score",
]

available_features = [
    feature for feature in skill_features
    if feature in filtered_df.columns
]

if available_features and total:
    selected_features = st.multiselect(
        "Choose assessment features",
        options=available_features,
        default=available_features[:5],
    )

    if selected_features:
        score_df = filtered_df.copy()

        for feature in selected_features:
            score_df[feature] = pd.to_numeric(
                score_df[feature], errors="coerce"
            )

        averages = (
            score_df.groupby("selected")[selected_features]
            .mean()
            .reindex([0, 1])
            .T
        )

        averages.columns = ["Not Selected", "Selected"]
        averages.index.name = "Assessment"
        averages = averages.reset_index()

        fig = px.bar(
            averages,
            x="Assessment",
            y=["Not Selected", "Selected"],
            barmode="group",
            title="Average Assessment Scores by Outcome",
            labels={"value": "Average score", "variable": "Outcome"},
        )

        fig.update_layout(
            xaxis_title="Assessment feature",
            yaxis_title="Average score",
            legend_title="Selection outcome",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one assessment feature.")
else:
    st.info("No assessment score columns are available.")

st.divider()

# =========================================================
# 8. MODEL PERFORMANCE SUMMARY
# =========================================================
st.subheader("🤖 Machine-Learning Model Performance")

metrics_path = OUTPUT_DIR / "best_model_holdout_metrics.csv"
summary_path = OUTPUT_DIR / "selected_model_summary.csv"

if metrics_path.exists():
    try:
        metrics_df = pd.read_csv(metrics_path)

        if not metrics_df.empty:
            metrics = metrics_df.iloc[0]
            model_name = metrics.get("Model", "Saved model")

            st.markdown(f"**Selected model:** {model_name}")

            m1, m2, m3 = st.columns(3)

            for column, key, label in [
                (m1, "accuracy", "Holdout Accuracy"),
                (m2, "balanced_accuracy", "Balanced Accuracy"),
                (m3, "roc_auc", "Holdout ROC-AUC"),
            ]:
                value = metrics.get(key)

                if pd.notna(value):
                    column.metric(label, f"{float(value):.2%}")
                else:
                    column.metric(label, "N/A")

            st.caption(
                "These metrics come from the saved holdout evaluation. "
                "They are not recalculated using the dashboard filters."
            )

            if float(metrics.get("roc_auc", 1)) < 0.65:
                st.warning(
                    "The saved model has limited predictive discrimination. "
                    "Interpret its results cautiously."
                )

            if summary_path.exists():
                summary_df = pd.read_csv(summary_path)

                if not summary_df.empty and "Best_Params" in summary_df:
                    with st.expander("View selected model parameters"):
                        st.code(str(summary_df.iloc[0]["Best_Params"]))

    except Exception as error:
        st.warning(f"Could not load model evaluation results: {error}")
else:
    st.info(
        "Model evaluation results have not been saved yet. "
        "The applicant statistics and charts above remain available."
    )

st.divider()

# =========================================================
# 9. DATASET PREVIEW
# =========================================================
with st.expander("🔍 Preview Applicant Dataset"):
    st.write(f"Showing up to 100 of {len(filtered_df):,} filtered records.")
    st.dataframe(
        filtered_df.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Filtered Dataset",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_internship_applicants.csv",
        mime="text/csv",
    )

st.caption(
    "Internship Selection Analytics Dashboard | "
    "Synthetic data for educational purposes. "
    "Observed associations do not establish causation, validity, or fairness."
)
