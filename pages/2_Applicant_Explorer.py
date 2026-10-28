
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Applicant Explorer",
    page_icon="👥",
    layout="wide"
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION AND DATA LOADING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "Internship_Selection_Dataset.csv"


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    if "selected" not in data.columns:
        raise ValueError("The dataset must contain a 'selected' column.")

    # Convert target to numeric safely
    data["selected"] = pd.to_numeric(
        data["selected"], errors="coerce"
    )
    data = data[data["selected"].isin([0, 1])].copy()
    data["selected"] = data["selected"].astype(int)

    # Convert known numeric columns safely
    numeric_candidates = [
        "CGPA",
        "skills_score",
        "projects_count",
        "internships_done",
        "certifications_count",
        "communication_score",
        "aptitude_score",
        "coding_test_score",
        "interview_score"
    ]

    for column in numeric_candidates:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    return data


try:
    df = load_data()
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()

if df.empty:
    st.error("No valid applicant records were found.")
    st.stop()


# --------------------------------------------------
# 2. PAGE HEADER
# --------------------------------------------------

st.title("👥 Applicant Explorer")
st.caption(
    "Search applicant records, compare profiles, "
    "and explore selection outcomes."
)

st.divider()


# --------------------------------------------------
# 3. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.title("🔎 Applicant Filters")

filtered = df.copy()

# Student ID search
if "student_id" in df.columns:
    student_search = st.sidebar.text_input(
        "Search Student ID",
        placeholder="Enter all or part of an ID"
    ).strip()

    if student_search:
        filtered = filtered[
            filtered["student_id"]
            .astype("string")
            .str.contains(
                student_search,
                case=False,
                na=False,
                regex=False
            )
        ]

# Selection status
status_options = {
    "All Applicants": None,
    "Selected": 1,
    "Not Selected": 0
}

status_label = st.sidebar.selectbox(
    "Selection Status",
    list(status_options.keys())
)

status_value = status_options[status_label]

if status_value is not None:
    filtered = filtered[
        filtered["selected"] == status_value
    ]

# Numeric range filters
def add_numeric_filter(data, column, label):
    """Apply a numeric range filter using the full dataset's range."""
    if column not in df.columns:
        return data

    numeric_values = df[column].dropna()

    if numeric_values.empty:
        return data

    minimum = float(numeric_values.min())
    maximum = float(numeric_values.max())

    if minimum == maximum:
        st.sidebar.caption(
            f"{label}: all available values are {minimum:g}"
        )
        return data

    selected_range = st.sidebar.slider(
        label,
        min_value=minimum,
        max_value=maximum,
        value=(minimum, maximum)
    )

    return data[
        data[column].between(
            selected_range[0],
            selected_range[1],
            inclusive="both"
        )
    ]


filtered = add_numeric_filter(filtered, "CGPA", "CGPA Range")
filtered = add_numeric_filter(
    filtered, "interview_score", "Interview Score Range"
)
filtered = add_numeric_filter(
    filtered, "skills_score", "Skills Score Range"
)

st.sidebar.divider()
st.sidebar.caption(
    "Filters apply to the table, metrics, charts, and applicant profile."
)


# --------------------------------------------------
# 4. SUMMARY METRICS
# --------------------------------------------------

total_filtered = len(filtered)
selected_count = int((filtered["selected"] == 1).sum())
not_selected_count = int((filtered["selected"] == 0).sum())

selection_rate = (
    selected_count / total_filtered * 100
    if total_filtered > 0
    else 0
)

overall_rate = float(df["selected"].mean() * 100)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Matching Applicants", f"{total_filtered:,}")
m2.metric("Selected", f"{selected_count:,}")
m3.metric("Not Selected", f"{not_selected_count:,}")
m4.metric("Selection Rate", f"{selection_rate:.1f}%")

st.caption(
    f"Dataset selection rate: {overall_rate:.1f}% · "
    "Filtered selection rate describes these records only."
)

st.divider()


# --------------------------------------------------
# 5. APPLICANT ANALYTICS
# --------------------------------------------------

st.subheader("📊 Applicant Analytics")

if filtered.empty:
    st.warning(
        "No applicants match these filters. "
        "Try widening the ranges or clearing the Student ID search."
    )
else:
    chart_left, chart_right = st.columns(2)

    with chart_left:
        status_data = pd.DataFrame({
            "Outcome": ["Selected", "Not Selected"],
            "Applicants": [selected_count, not_selected_count]
        })

        fig_status = px.bar(
            status_data,
            x="Outcome",
            y="Applicants",
            color="Outcome",
            text="Applicants",
            title="Selection Outcomes",
            color_discrete_map={
                "Selected": "#16A34A",
                "Not Selected": "#DC2626"
            }
        )
        fig_status.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Number of Applicants"
        )
        st.plotly_chart(fig_status, use_container_width=True)

    with chart_right:
        score_column = None

        for candidate in [
            "CGPA",
            "interview_score",
            "skills_score"
        ]:
            if candidate in filtered.columns:
                score_column = candidate
                break

        if score_column:
            chart_data = filtered[
                [score_column, "selected"]
            ].dropna().copy()

            chart_data["Outcome"] = chart_data["selected"].map({
                1: "Selected",
                0: "Not Selected"
            })

            fig_score = px.histogram(
                chart_data,
                x=score_column,
                color="Outcome",
                barmode="overlay",
                opacity=0.7,
                nbins=20,
                title=f"{score_column} Distribution by Outcome",
                color_discrete_map={
                    "Selected": "#16A34A",
                    "Not Selected": "#DC2626"
                }
            )
            fig_score.update_layout(
                xaxis_title=score_column,
                yaxis_title="Number of Applicants"
            )
            st.plotly_chart(fig_score, use_container_width=True)
        else:
            st.info("No numeric score columns are available for charting.")


# --------------------------------------------------
# 6. APPLICANT RECORDS TABLE
# --------------------------------------------------

st.divider()
st.subheader("📋 Applicant Records")

if filtered.empty:
    st.info("There are no records to display.")
else:
    preferred_columns = [
        "student_id",
        "CGPA",
        "skills_score",
        "projects_count",
        "internships_done",
        "certifications_count",
        "communication_score",
        "aptitude_score",
        "coding_test_score",
        "interview_score",
        "selected"
    ]

    all_columns = filtered.columns.tolist()

    default_columns = [
        column for column in preferred_columns
        if column in all_columns
    ]

    if not default_columns:
        default_columns = all_columns[:min(8, len(all_columns))]

    display_columns = st.multiselect(
        "Choose columns to display",
        options=all_columns,
        default=default_columns
    )

    if display_columns:
        display_df = filtered[display_columns].copy()

        if "selected" in display_df.columns:
            display_df["selected"] = display_df["selected"].map({
                1: "Selected",
                0: "Not Selected"
            })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=420
        )
    else:
        st.info("Choose at least one column to display.")

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Applicants CSV",
        data=csv_data,
        file_name="filtered_applicants.csv",
        mime="text/csv",
        use_container_width=False
    )


# --------------------------------------------------
# 7. INDIVIDUAL APPLICANT PROFILE
# --------------------------------------------------

st.divider()
st.subheader("🧑 Individual Applicant Profile")

if filtered.empty:
    st.info("No applicant profile is available for the current filters.")
else:
    if "student_id" in filtered.columns:
        # Use string labels so IDs with mixed types are handled safely.
        profile_data = filtered.copy()
        profile_data["_id_label"] = (
            profile_data["student_id"].astype(str)
        )

        id_labels = profile_data["_id_label"].drop_duplicates().tolist()

        chosen_label = st.selectbox(
            "Select an applicant",
            options=id_labels
        )

        applicant = profile_data[
            profile_data["_id_label"] == chosen_label
        ].iloc[0].drop(labels=["_id_label"])
    else:
        st.info(
            "This dataset has no student_id column. "
            "Choose a row number instead."
        )
        row_number = st.selectbox(
            "Applicant row",
            options=list(range(len(filtered))),
            format_func=lambda number: f"Applicant {number + 1}"
        )
        applicant = filtered.iloc[row_number]

    # Applicant headline
    if "student_id" in applicant.index:
        st.markdown(f"### Student ID: {applicant['student_id']}")

    outcome = (
        "Selected"
        if int(applicant["selected"]) == 1
        else "Not Selected"
    )

    st.write(f"**Recorded outcome:** {outcome}")

    # Show common numeric profile metrics
    profile_metrics = [
        ("CGPA", "CGPA"),
        ("Skills Score", "skills_score"),
        ("Interview Score", "interview_score"),
        ("Coding Test Score", "coding_test_score"),
        ("Communication Score", "communication_score"),
        ("Aptitude Score", "aptitude_score"),
        ("Projects", "projects_count"),
        ("Internships", "internships_done"),
        ("Certifications", "certifications_count")
    ]

    available_metrics = [
        (label, column)
        for label, column in profile_metrics
        if column in applicant.index and pd.notna(applicant[column])
    ]

    if available_metrics:
        metric_columns = st.columns(min(4, len(available_metrics)))

        for index, (label, column) in enumerate(available_metrics):
            value = applicant[column]

            if isinstance(value, (int, float)):
                formatted_value = (
                    f"{value:.2f}"
                    if isinstance(value, float)
                    else f"{value:,}"
                )
            else:
                formatted_value = str(value)

            metric_columns[index % len(metric_columns)].metric(
                label, formatted_value
            )

    with st.expander("View complete applicant record"):
        details = applicant.to_frame(name="Value")
        details.index.name = "Field"
        st.dataframe(
            details,
            use_container_width=True
        )

    single_csv = applicant.to_frame().T.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download This Applicant's Record",
        data=single_csv,
        file_name="applicant_record.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# 8. DATA QUALITY NOTE
# --------------------------------------------------

st.divider()

with st.expander("ℹ️ How to interpret this page"):
    st.write(
        "This page describes the applicant records in the supplied dataset. "
        "The selection rate is the percentage of records marked as selected "
        "within the current filters; it is not a prediction of future outcomes."
    )
    st.write(
        "The dataset is synthetic, so observed patterns may not represent "
        "real internship recruitment. These charts should not be used alone "
        "to make hiring decisions."
    )

st.caption("Internship Selection Dashboard · Applicant Explorer")
