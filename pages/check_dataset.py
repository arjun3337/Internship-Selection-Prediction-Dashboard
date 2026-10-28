
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Check Dataset",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] a[href*="check_dataset"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧪 Check Dataset")
st.caption(
    "Fast dataset validation without triggering the heavy model-training pipeline."
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Internship_Selection_Dataset.csv"

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()


@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


df = load_dataset(str(DATA_PATH))

row_count = len(df)
col_count = len(df.columns)
missing_cells = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

left, middle, right, end = st.columns(4)
left.metric("Rows", f"{row_count:,}")
middle.metric("Columns", f"{col_count:,}")
right.metric("Missing Cells", f"{missing_cells:,}")
end.metric("Duplicate Rows", f"{duplicate_rows:,}")

st.divider()

if "selected" in df.columns:
    target_counts = df["selected"].value_counts(dropna=False).sort_index()
    st.subheader("🎯 Target Distribution")
    st.bar_chart(target_counts)

    if df["selected"].isna().any():
        st.warning("The target column contains missing values.")
else:
    st.warning("The dataset does not contain the expected 'selected' column.")

st.subheader("📊 Data Quality Snapshot")
missing_info = pd.DataFrame(
    {
        "Column": df.columns,
        "Missing Values": df.isna().sum().values,
        "Missing %": (df.isna().mean().values * 100).round(2),
    }
).sort_values(["Missing Values", "Missing %"], ascending=[False, False])

if missing_info["Missing Values"].sum() == 0:
    st.success("No missing values were found in the dataset.")
else:
    st.warning(f"{int(missing_info['Missing Values'].sum())} missing values were identified.")
    st.dataframe(missing_info, use_container_width=True)

constant_columns = [
    col for col in df.columns if df[col].nunique(dropna=False) <= 1
]
if constant_columns:
    st.warning(f"Constant or single-value columns: {', '.join(constant_columns)}")
else:
    st.info("No constant columns were detected.")

if duplicate_rows > 0:
    st.warning(f"Found {duplicate_rows} duplicated rows.")
    with st.expander("Preview duplicate rows"):
        st.dataframe(df[df.duplicated(keep=False)].head(20), use_container_width=True)
else:
    st.success("No fully duplicated rows were found.")

st.subheader("🔎 Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

st.caption(
    "This page is designed for quick validation only. It intentionally avoids expensive model retraining "
    "so the dashboard stays responsive."
)
