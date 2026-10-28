
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Internship Selection Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Internship_Selection_Dataset.csv"
CSS_PATH = BASE_DIR / "assets" / "styles.css"

# Load custom CSS from the separate CSS file
if CSS_PATH.exists():
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

st.title("🎓 Internship Selection Prediction")
st.subheader("Employability & Recruitment Analytics Dashboard")
st.write(
    "Explore internship-selection patterns, model results, "
    "applicant predictions, and employability insights."
)

@st.cache_data
def load_local_dataset(path):
    return pd.read_csv(path)

if DATA_PATH.exists():
    try:
        df = load_local_dataset(str(DATA_PATH))
    except Exception as error:
        st.error(f"Unable to load the dataset: {error}")
        st.stop()
else:
    uploaded_file = st.file_uploader(
        "Upload Internship_Selection_Dataset.csv",
        type=["csv"]
    )
    if uploaded_file is None:
        st.stop()
    df = pd.read_csv(uploaded_file)

c1, c2, c3 = st.columns(3)
c1.metric("Applicants", f"{len(df):,}")
c2.metric("Columns", len(df.columns))

if "selected" in df.columns and pd.api.types.is_numeric_dtype(df["selected"]):
    c3.metric("Selection Rate", f"{df['selected'].mean() * 100:.1f}%")
else:
    c3.metric("Selection Rate", "N/A")

st.dataframe(df.head(20), use_container_width=True)
st.info("Use the sidebar pages for the extended dashboard sections.")
