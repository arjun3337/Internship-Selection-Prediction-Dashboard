
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Selection Patterns",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# 1. LOAD AND VALIDATE DATA
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "Internship_Selection_Dataset.csv"


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    if "selected" not in data.columns:
        raise ValueError("The dataset must contain a 'selected' column.")

    data["selected"] = pd.to_numeric(
        data["selected"], errors="coerce"
    )

    data = data[data["selected"].isin([0, 1])].copy()
    data["selected"] = data["selected"].astype(int)

    # Convert columns that contain numeric values
    for column in data.columns:
        if column == "student_id":
            continue

        converted = pd.to_numeric(data[column], errors="coerce")
        original_count = data[column].notna().sum()

        if original_count > 0:
            numeric_ratio = converted.notna().sum() / original_count

            if numeric_ratio >= 0.9:
                data[column] = converted

    return data


try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

if df.empty:
    st.error("No valid records were found in the dataset.")
    st.stop()


# ==================================================
# 2. HEADER
# ==================================================

st.title("📊 Internship Selection Patterns")

st.caption(
    "Explore academic performance, assessment scores, "
    "and experience alongside recorded internship outcomes."
)

st.info(
    "This dataset is synthetic. The charts describe the supplied records; "
    "they do not establish real-world hiring rules or causal relationships."
)


# ==================================================
# 3. SIDEBAR FILTERS
# ==================================================

st.sidebar.header("Analysis Filters")

status_options = {
    "All applicants": None,
    "Selected": 1,
    "Not selected": 0
}

status_label = st.sidebar.selectbox(
    "Applicant group",
    list(status_options.keys())
)

status_value = status_options[status_label]

data = df.copy()

if status_value is not None:
    data = data[data["selected"] == status_value]

if data.empty:
    st.warning("No applicants match this filter.")
    st.stop()


# ==================================================
# 4. SUMMARY METRICS
# ==================================================

total = len(data)
selected_count = int((data["selected"] == 1).sum())
not_selected_count = int((data["selected"] == 0).sum())

view_selection_rate = (
    selected_count / total * 100 if total else 0
)

overall_selection_rate = df["selected"].mean() * 100

m1, m2, m3, m4 = st.columns(4)

m1.metric("Applicants in View", f"{total:,}")
m2.metric("Selected in View", f"{selected_count:,}")
m3.metric("Not Selected in View", f"{not_selected_count:,}")
m4.metric("Selection Rate in View", f"{view_selection_rate:.1f}%")

st.caption(
    f"Full dataset selection rate: {overall_selection_rate:.1f}%"
)

st.divider()


# ==================================================
# 5. IDENTIFY AVAILABLE FEATURES
# ==================================================

score_candidates = [
    "CGPA",
    "skills_score",
    "communication_score",
    "aptitude_score",
    "coding_test_score",
    "interview_score",
    "soft_skills_score",
    "resume_score",
    "github_score",
    "consistency_score",
    "linkedin_activity_score",
    "hackathons_participated",
    "backlogs"
]

experience_candidates = [
    "projects_count",
    "internships_done",
    "certifications_count"
]

available_scores = [
    col for col in score_candidates
    if col in df.columns
    and pd.api.types.is_numeric_dtype(df[col])
    and df[col].notna().any()
]

available_experience = [
    col for col in experience_candidates
    if col in df.columns
    and pd.api.types.is_numeric_dtype(df[col])
    and df[col].notna().any()
]


# ==================================================
# 6. ACADEMIC AND ASSESSMENT COMPARISON
# ==================================================

st.subheader("1. Academic and Assessment Comparison")

if available_scores:
    chosen_features = st.multiselect(
        "Choose features to compare",
        options=available_scores,
        default=[
            col for col in [
                "CGPA",
                "skills_score",
                "coding_test_score",
                "interview_score"
            ]
            if col in available_scores
        ]
    )

    if chosen_features:
        comparison = (
            df.groupby("selected")[chosen_features]
            .mean()
            .T
            .rename_axis("Feature")
            .reset_index()
            .melt(
                id_vars="Feature",
                var_name="selected",
                value_name="Average Score"
            )
        )

        comparison["Selection Status"] = comparison["selected"].map({
            0: "Not Selected",
            1: "Selected"
        })

        fig = px.bar(
            comparison,
            x="Feature",
            y="Average Score",
            color="Selection Status",
            barmode="group",
            color_discrete_map={
                "Selected": "#16A34A",
                "Not Selected": "#DC2626"
            },
            title="Average Scores by Recorded Outcome"
        )

        fig.update_layout(
            xaxis_title="Feature",
            yaxis_title="Average score",
            legend_title="Outcome"
        )

        st.plotly_chart(fig, use_container_width=True)

        average_table = comparison.pivot(
            index="Feature",
            columns="Selection Status",
            values="Average Score"
        ).round(2)

        st.dataframe(
            average_table,
            use_container_width=True
        )
    else:
        st.info("Choose at least one feature to compare.")
else:
    st.info("No numeric assessment features were found.")


# ==================================================
# 7. SIDE-BY-SIDE SCORE DISTRIBUTION
# ==================================================

st.divider()
st.subheader("2. Score Distribution")

if available_scores:
    default_index = (
        available_scores.index("CGPA")
        if "CGPA" in available_scores else 0
    )

    distribution_feature = st.selectbox(
        "Choose a score",
        options=available_scores,
        index=default_index,
        key="distribution_feature"
    )

    distribution_data = data[
        [distribution_feature, "selected"]
    ].dropna().copy()

    distribution_data["Selection Status"] = (
        distribution_data["selected"].map({
            1: "Selected",
            0: "Not Selected"
        })
    )

    if distribution_data.empty:
        st.warning("There are no valid values for this score in the current group.")
    else:
        fig = px.histogram(
            distribution_data,
            x=distribution_feature,
            color="Selection Status",
            barmode="group",
            nbins=20,
            color_discrete_map={
                "Selected": "#16A34A",
                "Not Selected": "#DC2626"
            },
            title=f"{distribution_feature} Distribution by Outcome"
        )

        fig.update_layout(
            xaxis_title=distribution_feature,
            yaxis_title="Number of applicants",
            legend_title="Outcome"
        )

        st.plotly_chart(fig, use_container_width=True)

        score_summary = (
            distribution_data
            .groupby("Selection Status")[distribution_feature]
            .agg(["count", "mean", "median"])
            .rename(columns={
                "count": "Applicants",
                "mean": "Average",
                "median": "Median"
            })
            .round(2)
        )

        st.dataframe(
            score_summary,
            use_container_width=True
        )

        st.caption(
            "Bars are grouped side by side rather than overlaid. "
            "When a group filter is active, this chart displays only that group."
        )


# ==================================================
# 8. SELECTION RATE BY SCORE BAND
# ==================================================

st.divider()
st.subheader("3. Selection Rate by Score Range")

st.caption(
    "This analysis always uses the full dataset so both outcome groups "
    "remain available, even when a sidebar group filter is active."
)

if available_scores:
    rate_feature = st.selectbox(
        "Choose a score to analyze",
        options=available_scores,
        index=(
            available_scores.index("CGPA")
            if "CGPA" in available_scores else 0
        ),
        key="rate_feature"
    )

    rate_data = df[[rate_feature, "selected"]].dropna().copy()

    unique_values = rate_data[rate_feature].nunique()

    if unique_values >= 2:
        try:
            number_of_bins = min(5, unique_values)

            rate_data["Score Band"] = pd.qcut(
                rate_data[rate_feature],
                q=number_of_bins,
                duplicates="drop"
            )

            band_summary = (
                rate_data.groupby("Score Band", observed=True)
                .agg(
                    Applicants=("selected", "size"),
                    Selected=("selected", "sum"),
                    Selection_Rate=("selected", "mean")
                )
                .reset_index()
            )

            band_summary["Not Selected"] = (
                band_summary["Applicants"] - band_summary["Selected"]
            )

            band_summary["Selection Rate (%)"] = (
                band_summary["Selection_Rate"] * 100
            )

            band_summary["Score Band"] = (
                band_summary["Score Band"].astype(str)
            )

            band_summary = band_summary.sort_values(
                "Selection_Rate", ascending=True
            )

            fig = px.bar(
                band_summary,
                x="Selection Rate (%)",
                y="Score Band",
                orientation="h",
                text=band_summary["Selection Rate (%)"].map(
                    lambda value: f"{value:.1f}%"
                ),
                color="Selection Rate (%)",
                color_continuous_scale="Blues",
                title=f"Selection Rate by {rate_feature} Band",
                hover_data={
                    "Applicants": True,
                    "Selected": True,
                    "Not Selected": True,
                    "Selection Rate (%)": ":.2f"
                }
            )

            fig.update_traces(
                textposition="outside",
                cliponaxis=False
            )

            fig.update_layout(
                xaxis_title="Selection rate (%)",
                yaxis_title="Score band",
                xaxis_range=[0, 100],
                coloraxis_showscale=False
            )

            st.plotly_chart(fig, use_container_width=True)

            table_columns = [
                "Score Band",
                "Applicants",
                "Selected",
                "Not Selected",
                "Selection Rate (%)"
            ]

            st.dataframe(
                band_summary[table_columns].round(2),
                use_container_width=True,
                hide_index=True
            )

            # Downloadable band summary
            csv = band_summary[table_columns].to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Score-Band Analysis CSV",
                data=csv,
                file_name="selection_rate_by_score_band.csv",
                mime="text/csv"
            )

        except (ValueError, TypeError) as error:
            st.warning(f"Could not create score bands: {error}")
    else:
        st.info("This feature needs at least two distinct values.")


# ==================================================
# 9. PROJECTS AND EXPERIENCE
# ==================================================

st.divider()
st.subheader("4. Projects, Internships and Certifications")

if available_experience:
    experience_comparison = (
        df.groupby("selected")[available_experience]
        .mean()
        .T
        .rename_axis("Feature")
        .reset_index()
        .melt(
            id_vars="Feature",
            var_name="selected",
            value_name="Average Count"
        )
    )

    experience_comparison["Selection Status"] = (
        experience_comparison["selected"].map({
            0: "Not Selected",
            1: "Selected"
        })
    )

    fig = px.bar(
        experience_comparison,
        x="Feature",
        y="Average Count",
        color="Selection Status",
        barmode="group",
        color_discrete_map={
            "Selected": "#16A34A",
            "Not Selected": "#DC2626"
        },
        title="Average Experience Indicators by Outcome"
    )

    fig.update_layout(
        xaxis_title="Experience feature",
        yaxis_title="Average count",
        legend_title="Outcome"
    )

    st.plotly_chart(fig, use_container_width=True)

    experience_table = experience_comparison.pivot(
        index="Feature",
        columns="Selection Status",
        values="Average Count"
    ).round(2)

    st.dataframe(
        experience_table,
        use_container_width=True
    )
else:
    st.info("No numeric project or experience columns were found.")


# ==================================================
# 10. CORRELATION WITH RECORDED SELECTION
# ==================================================

st.divider()
st.subheader("5. Numeric Feature Association")

numeric_data = df.select_dtypes(include="number").copy()

for excluded in ["student_id"]:
    if excluded in numeric_data.columns:
        numeric_data = numeric_data.drop(columns=[excluded])

if "selected" in numeric_data.columns:
    correlation = numeric_data.corr(numeric_only=True)

    target_correlation = (
        correlation["selected"]
        .drop(labels=["selected"], errors="ignore")
        .dropna()
        .sort_values()
        .reset_index()
    )

    target_correlation.columns = [
        "Feature",
        "Correlation with Selection"
    ]

    if not target_correlation.empty:
        fig = px.bar(
            target_correlation,
            x="Correlation with Selection",
            y="Feature",
            orientation="h",
            color="Correlation with Selection",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Correlation of Numeric Features with Selection"
        )

        fig.update_layout(
            xaxis_title="Pearson correlation",
            yaxis_title="Feature"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            target_correlation.round(3),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Correlation measures linear association, not causation. "
            "A weak correlation does not rule out nonlinear relationships, "
            "and correlation alone does not establish predictive value."
        )
    else:
        st.info("No usable numeric correlations could be calculated.")
else:
    st.info("The target column is unavailable for correlation analysis.")


# ==================================================
# 11. AUTOMATIC DESCRIPTIVE FINDINGS
# ==================================================

st.divider()
st.subheader("6. Key Findings")

st.caption(
    "The following statements describe average differences in the full "
    "dataset. They are not causal conclusions."
)

finding_features = [
    feature for feature in available_scores + available_experience
    if feature in df.columns
]

findings = []

for feature in finding_features:
    selected_values = df.loc[
        df["selected"] == 1, feature
    ].dropna()

    not_selected_values = df.loc[
        df["selected"] == 0, feature
    ].dropna()

    if selected_values.empty or not not_selected_values.empty is False:
        continue

    selected_mean = selected_values.mean()
    not_selected_mean = not_selected_values.mean()
    difference = selected_mean - not_selected_mean

    if abs(difference) < 0.01:
        findings.append(
            f"**{feature}:** the group averages are nearly identical "
            f"({selected_mean:.2f} vs. {not_selected_mean:.2f})."
        )
    elif difference > 0:
        findings.append(
            f"**{feature}:** the selected group has a higher average "
            f"({selected_mean:.2f}) than the not-selected group "
            f"({not_selected_mean:.2f})."
        )
    else:
        findings.append(
            f"**{feature}:** the selected group has a lower average "
            f"({selected_mean:.2f}) than the not-selected group "
            f"({not_selected_mean:.2f})."
        )

if findings:
    for finding in findings:
        st.markdown(f"- {finding}")
else:
    st.info("There is not enough data to generate descriptive findings.")


# ==================================================
# 12. FOOTER
# ==================================================

st.divider()

with st.expander("How should these results be interpreted?"):
    st.markdown(
        """
        - **Selection rate** is the percentage of records marked selected
          within a score band.
        - **Average differences** summarize the two recorded outcome groups.
        - **Correlation** measures linear association between numeric variables.
        - None of these analyses proves that changing a score will change
          an applicant's outcome.
        - Because the dataset is synthetic, these patterns may not represent
          real recruitment decisions.
        """
    )

st.caption("Internship Selection Dashboard · Selection Patterns")
