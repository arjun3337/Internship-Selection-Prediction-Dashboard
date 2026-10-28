
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Data Quality",
    page_icon="🧹",
    layout="wide"
)

st.title("🧹 Data Quality")
st.caption(
    "Inspect dataset completeness, duplicates, data types, "
    "and potential outliers."
)

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

# --------------------------------------------------
# Dataset overview
# --------------------------------------------------
st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Rows", f"{df.shape[0]:,}")
col2.metric("Total Columns", f"{df.shape[1]:,}")
col3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
col4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

st.divider()

# --------------------------------------------------
# Missing values
# --------------------------------------------------
st.subheader("🔍 Missing Value Analysis")

missing = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isna().sum().values,
    "Missing Percentage": (
        df.isna().mean().values * 100
    ).round(2)
})

missing = missing.sort_values(
    "Missing Percentage",
    ascending=False
)

missing_columns = missing[missing["Missing Values"] > 0]

if missing_columns.empty:
    st.success("No missing values were found in the dataset.")
else:
    st.warning(
        f"{len(missing_columns)} columns contain missing values."
    )

    fig_missing = px.bar(
        missing_columns,
        x="Column",
        y="Missing Percentage",
        title="Missing Values by Column (%)",
        text="Missing Percentage",
        template="plotly_white"
    )
    fig_missing.update_layout(
        xaxis_tickangle=-45,
        yaxis_title="Missing Values (%)"
    )
    st.plotly_chart(fig_missing, use_container_width=True)

with st.expander("View missing-value report"):
    st.dataframe(missing, use_container_width=True)

st.divider()

# --------------------------------------------------
# Duplicate rows
# --------------------------------------------------
st.subheader("📑 Duplicate Record Analysis")

duplicate_count = int(df.duplicated().sum())

if duplicate_count == 0:
    st.success("No completely duplicated rows were found.")
else:
    st.warning(
        f"{duplicate_count:,} completely duplicated rows were found."
    )

if duplicate_count > 0:
    with st.expander("Preview duplicate rows"):
        st.dataframe(
            df[df.duplicated(keep=False)].head(100),
            use_container_width=True
        )

st.caption(
    "Duplicate checks compare complete rows. Repeated values in "
    "individual columns are not necessarily duplicate records."
)

st.divider()

# --------------------------------------------------
# Data types
# --------------------------------------------------
st.subheader("🧾 Column Data Types")

dtype_report = pd.DataFrame({
    "Column": df.columns,
    "Data Type": [str(dtype) for dtype in df.dtypes],
    "Unique Values": [
        df[col].nunique(dropna=True) for col in df.columns
    ],
    "Missing Values": [
        int(df[col].isna().sum()) for col in df.columns
    ]
})

st.dataframe(dtype_report, use_container_width=True)

st.divider()

# --------------------------------------------------
# Numeric summary and potential outliers
# --------------------------------------------------
st.subheader("📈 Numeric Distribution and Outliers")

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

if numeric_columns:
    selected_column = st.selectbox(
        "Choose a numeric column",
        numeric_columns
    )

    numeric_values = df[selected_column].dropna()

    if not numeric_values.empty:
        fig_hist = px.histogram(
            numeric_values,
            x=numeric_values,
            nbins=30,
            title=f"Distribution of {selected_column}",
            labels={"x": selected_column, "count": "Records"},
            template="plotly_white"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        q1 = numeric_values.quantile(0.25)
        q3 = numeric_values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        potential_outliers = numeric_values[
            (numeric_values < lower_bound) |
            (numeric_values > upper_bound)
        ]

        a, b, c = st.columns(3)
        a.metric("Potential Outliers", f"{len(potential_outliers):,}")
        b.metric("Lower IQR Bound", f"{lower_bound:.2f}")
        c.metric("Upper IQR Bound", f"{upper_bound:.2f}")

        st.caption(
            "Potential outliers are flagged using the 1.5 × IQR rule. "
            "They are not automatically errors and have not been removed."
        )

        with st.expander("View potential outlier values"):
            st.dataframe(
                pd.DataFrame({
                    selected_column: potential_outliers
                }).head(200),
                use_container_width=True
            )
else:
    st.info("No numeric columns were found.")

st.divider()

# --------------------------------------------------
# Target validation
# --------------------------------------------------
st.subheader("🎯 Target Column Validation")

if "selected" in df.columns:
    target_counts = (
        df["selected"]
        .value_counts(dropna=False)
        .rename_axis("Selection Label")
        .reset_index(name="Count")
    )

    st.dataframe(target_counts, use_container_width=True)

    if df["selected"].isna().any():
        st.warning("The target column contains missing labels.")

    st.caption(
        "Check that target labels match the intended coding "
        "before training or evaluating a model."
    )
else:
    st.warning("The expected target column 'selected' was not found.")

st.divider()

# --------------------------------------------------
# Export reports
# --------------------------------------------------
st.subheader("📥 Download Quality Reports")

left, right = st.columns(2)

with left:
    st.download_button(
        "Download Missing-Value Report",
        data=missing.to_csv(index=False).encode("utf-8"),
        file_name="missing_value_report.csv",
        mime="text/csv"
    )

with right:
    st.download_button(
        "Download Data-Type Report",
        data=dtype_report.to_csv(index=False).encode("utf-8"),
        file_name="data_type_report.csv",
        mime="text/csv"
    )

st.caption(
    "This page reports potential data-quality issues. "
    "It does not modify or delete the original dataset."
)

# --------------------------------------------------
# Valid-value range checks
# --------------------------------------------------
st.divider()
st.subheader("✅ Valid-Value Range Checks")

st.write(
    "Check whether numeric values fall within expected ranges. "
    "These rules are based on the intended scale of each feature."
)

# Expected ranges for known score and count columns
range_rules = {
    "CGPA": (0, 10),
    "resume_score": (0, 10),
    "skills_score": (0, 10),
    "coding_test_score": (0, 10),
    "communication_score": (0, 10),
    "soft_skills_score": (0, 10),
    "interview_score": (0, 10),
    "aptitude_score": (0, 10),
    "github_score": (0, 10),
    "linkedin_activity_score": (0, 10),
    "consistency_score": (0, 10),
    "internships_done": (0, None),
    "projects_count": (0, None),
    "certifications_count": (0, None),
    "hackathons_participated": (0, None),
    "backlogs": (0, None),
}

range_results = []

for column, (lower, upper) in range_rules.items():
    if column not in df.columns:
        continue

    values = pd.to_numeric(df[column], errors="coerce")
    invalid_mask = values.isna() & df[column].notna()

    if lower is not None:
        invalid_mask = invalid_mask | (values < lower)

    if upper is not None:
        invalid_mask = invalid_mask | (values > upper)

    range_results.append({
        "Column": column,
        "Expected Minimum": lower if lower is not None else "No limit",
        "Expected Maximum": upper if upper is not None else "No limit",
        "Invalid Values": int(invalid_mask.sum()),
        "Status": (
            "Pass" if invalid_mask.sum() == 0 else "Review"
        )
    })

range_report = pd.DataFrame(range_results)

if range_report.empty:
    st.info("No configured numeric columns were found to validate.")
else:
    invalid_total = int(range_report["Invalid Values"].sum())
    passing_columns = int((range_report["Status"] == "Pass").sum())

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Columns Checked", len(range_report))
    metric2.metric("Columns Passing", passing_columns)
    metric3.metric("Invalid Values Found", invalid_total)

    if invalid_total == 0:
        st.success("All checked values fall within the configured ranges.")
    else:
        st.warning(
            "Some values need review. Check the report below before "
            "changing or removing any data."
        )

    st.dataframe(range_report, use_container_width=True)

    # Preview rows containing invalid values
    with st.expander("View invalid records"):
        invalid_records = []

        for column, (lower, upper) in range_rules.items():
            if column not in df.columns:
                continue

            values = pd.to_numeric(df[column], errors="coerce")
            invalid_mask = values.isna() & df[column].notna()

            if lower is not None:
                invalid_mask = invalid_mask | (values < lower)

            if upper is not None:
                invalid_mask = invalid_mask | (values > upper)

            if invalid_mask.any():
                rows = df.loc[invalid_mask].copy()
                rows.insert(0, "Invalid Column", column)
                invalid_records.append(rows)

        if invalid_records:
            st.dataframe(
                pd.concat(invalid_records, ignore_index=True),
                use_container_width=True
            )
        else:
            st.success("No invalid records to display.")

    st.download_button(
        "Download Range Validation Report",
        data=range_report.to_csv(index=False).encode("utf-8"),
        file_name="range_validation_report.csv",
        mime="text/csv"
    )

st.caption(
    "Range checks flag possible issues; they do not automatically "
    "correct or remove records. Count columns are checked for "
    "negative values, not against an assumed maximum."
)
