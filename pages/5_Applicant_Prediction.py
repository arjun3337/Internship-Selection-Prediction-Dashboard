
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Applicant Prediction",
    page_icon="🎯",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "Internship_Selection_Dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "best_internship_model.pkl"

st.title("🎯 Applicant Prediction")
st.caption(
    "Explore applicant profiles using the selected machine-learning model."
)

st.warning(
    "Educational demonstration using a synthetic dataset. "
    "Predictions are experimental and must not be used for real hiring decisions."
)

# ---------------------------------------------------------
# 1. CHECK FILES
# ---------------------------------------------------------

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(
        f"Saved model not found: {MODEL_PATH}\n\n"
        "Run this command from your project root first:\n"
        "python pages/check_dataset.py"
    )
    st.stop()


# ---------------------------------------------------------
# 2. LOAD THE DATA AND SAVED MODEL
# ---------------------------------------------------------

@st.cache_data
def load_dataset(path, modified_time):
    data = pd.read_csv(path)
    data.columns = data.columns.astype(str).str.strip()

    if "selected" not in data.columns:
        raise ValueError("The dataset must contain a 'selected' column.")

    data = data.drop_duplicates().copy()
    data["selected"] = pd.to_numeric(
        data["selected"], errors="coerce"
    )
    data = data.dropna(subset=["selected"]).copy()
    data["selected"] = data["selected"].astype(int)

    if not set(data["selected"].unique()).issubset({0, 1}):
        raise ValueError("'selected' must contain only 0 and 1.")

    return data


@st.cache_resource
def load_saved_model(path, modified_time):
    return joblib.load(path)


try:
    data = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    model = load_saved_model(
        str(MODEL_PATH), MODEL_PATH.stat().st_mtime
    )
except Exception as error:
    st.error(f"Could not load the dataset or saved model: {error}")
    st.stop()

X = data.drop(
    columns=["selected", "student_id"], errors="ignore"
).copy()
y = data["selected"].copy()

# Use the exact feature names recorded when the pipeline was trained.
model_features = list(getattr(model, "feature_names_in_", X.columns))

missing_features = [c for c in model_features if c not in X.columns]
if missing_features:
    st.error(
        "The dataset is missing columns required by the saved model: "
        + ", ".join(missing_features)
    )
    st.stop()

X = X[model_features].replace([np.inf, -np.inf], np.nan)

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = [
    c for c in X.columns if c not in numeric_features
]

# ---------------------------------------------------------
# 3. HOLDOUT METRICS
# ---------------------------------------------------------

st.subheader("📊 Model Overview")

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Evaluate the saved pipeline on the same reproducible split.
    # The model was trained using all training records in the tuning script,
    # so these numbers are descriptive only if that exact split was used
    # during training. For a fully independent test, keep a separate test set.
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
    }

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records", f"{len(X):,}")
    c2.metric("Input Features", len(X.columns))
    c3.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c4.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")

    st.caption(
        "These metrics are a diagnostic view, not a fresh independent "
        "holdout evaluation, because the saved pipeline may already have "
        "seen some of these records during training."
    )

    if metrics["roc_auc"] < 0.65:
        st.warning(
            "The model has limited predictive ability. Treat all scores "
            "as experimental, not as reliable selection probabilities."
        )

except Exception as error:
    st.warning(f"Could not calculate the overview metrics: {error}")
    st.metric("Dataset Records", f"{len(X):,}")
    st.metric("Input Features", len(X.columns))


# ---------------------------------------------------------
# 4. APPLICANT INPUT FORM
# ---------------------------------------------------------

st.divider()
st.subheader("👤 Applicant Profile")
st.write("Enter the details below to explore a model prediction.")

def feature_group(feature):
    name = feature.lower()

    if any(term in name for term in [
        "cgpa", "backlog", "aptitude", "college_tier",
        "degree", "branch", "academic", "graduation"
    ]):
        return "Academic Background"

    if any(term in name for term in [
        "skill", "coding", "communication", "soft",
        "interview", "consistency"
    ]):
        return "Skills & Assessment"

    if any(term in name for term in [
        "project", "internship", "hackathon",
        "certification", "github", "linkedin",
        "extracurricular", "resume"
    ]):
        return "Experience & Activities"

    return "Other Details"


group_order = [
    "Academic Background",
    "Skills & Assessment",
    "Experience & Activities",
    "Other Details",
]

feature_groups = {group: [] for group in group_order}
for feature in X.columns:
    feature_groups[feature_group(feature)].append(feature)

with st.form("applicant_prediction_form"):
    applicant = {}

    for group in group_order:
        features = feature_groups[group]
        if not features:
            continue

        st.markdown(f"### {group}")
        columns = st.columns(2)

        for index, feature in enumerate(features):
            with columns[index % 2]:
                label = feature.replace("_", " ").title()

                if feature in numeric_features:
                    values = pd.to_numeric(
                        X[feature], errors="coerce"
                    ).replace([np.inf, -np.inf], np.nan).dropna()

                    if values.empty:
                        minimum, maximum, default = 0.0, 100.0, 0.0
                    else:
                        minimum = float(values.min())
                        maximum = float(values.max())
                        default = float(values.median())

                    if minimum == maximum:
                        minimum -= 1.0
                        maximum += 1.0

                    is_integer = (
                        pd.api.types.is_integer_dtype(X[feature])
                        or (
                            not values.empty
                            and np.allclose(
                                values.to_numpy(),
                                np.round(values.to_numpy())
                            )
                        )
                    )

                    if is_integer:
                        applicant[feature] = st.number_input(
                            label,
                            min_value=int(np.floor(minimum)),
                            max_value=int(np.ceil(maximum)),
                            value=int(round(default)),
                            step=1,
                            key=f"input_{feature}",
                        )
                    else:
                        applicant[feature] = st.number_input(
                            label,
                            min_value=float(minimum),
                            max_value=float(maximum),
                            value=float(np.clip(default, minimum, maximum)),
                            step=0.1 if maximum - minimum <= 20 else 1.0,
                            format="%.2f",
                            key=f"input_{feature}",
                        )
                else:
                    categories = (
                        X[feature].dropna().astype(str)
                        .value_counts().index.tolist()
                    )
                    if not categories:
                        categories = ["Unknown"]

                    applicant[feature] = st.selectbox(
                        label,
                        options=categories,
                        key=f"input_{feature}",
                    )

    submitted = st.form_submit_button(
        "🔍 Predict Applicant Outcome",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------
# 5. PREDICTION RESULT
# ---------------------------------------------------------

if submitted:
    input_df = pd.DataFrame([applicant], columns=X.columns)

    for feature in numeric_features:
        input_df[feature] = pd.to_numeric(
            input_df[feature], errors="coerce"
        )

    try:
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        classifier = model.steps[-1][1]
        classes = list(classifier.classes_)
        selected_index = classes.index(1)

        selected_score = float(probabilities[selected_index])
        not_selected_score = 1.0 - selected_score

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    st.session_state["last_applicant_input"] = input_df
    st.session_state["last_prediction"] = prediction
    st.session_state["last_selected_score"] = selected_score

    st.divider()
    st.subheader("🎯 Prediction Results")

    result_col, chart_col = st.columns(2)

    with result_col:
        if prediction == 1:
            st.success("Predicted Outcome: Selected")
        else:
            st.warning("Predicted Outcome: Not Selected")

        st.metric("Model Score for Selected", f"{selected_score:.1%}")
        st.progress(float(np.clip(selected_score, 0, 1)))

        st.caption(
            "This score is not a calibrated probability of actual selection."
        )

    with chart_col:
        score_df = pd.DataFrame({
            "Outcome": ["Selected", "Not Selected"],
            "Model Score": [selected_score, not_selected_score],
        })

        fig = px.bar(
            score_df,
            x="Outcome",
            y="Model Score",
            range_y=[0, 1],
            text=score_df["Model Score"].map(
                lambda value: f"{value:.1%}"
            ),
            title="Estimated Class Scores",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Experimental result only. Do not use this prediction to accept, "
        "reject, or rank real applicants."
    )

    with st.expander("Review applicant inputs"):
        st.dataframe(
            input_df.T.rename(columns={0: "Input Value"}),
            use_container_width=True,
        )

    result_csv = input_df.copy()
    result_csv["predicted_selected"] = prediction
    result_csv["estimated_selected_score"] = selected_score
    result_csv["estimated_not_selected_score"] = not_selected_score

    st.download_button(
        "⬇️ Download Prediction Result CSV",
        data=result_csv.to_csv(index=False).encode("utf-8"),
        file_name="applicant_prediction.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------
# 6. FEATURE IMPORTANCE / MODEL COEFFICIENTS
# ---------------------------------------------------------

st.divider()
st.subheader("📊 What Influences the Model?")

st.caption(
    "This chart describes the model's global feature influence. "
    "It does not prove causation or explain an individual decision."
)

try:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.steps[-1][1]
    transformed_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        chart_title = "Random Forest Feature Importance"
        value_label = "Feature importance"
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_[0])
        chart_title = "Logistic Regression Coefficient Magnitudes"
        value_label = "Absolute coefficient magnitude"
    else:
        values = None

    if values is None:
        st.info("This model does not expose feature importance or coefficients.")
    else:
        importance_df = pd.DataFrame({
            "Feature": transformed_names,
            "Importance": values,
        })

        importance_df["Feature"] = (
            importance_df["Feature"]
            .str.replace("numeric__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
        )

        # Combine encoded levels into their original feature name.
        for category_feature in categorical_features:
            matching = importance_df["Feature"].str.startswith(
                category_feature + "_"
            )
            importance_df.loc[matching, "Feature"] = category_feature

        importance_df = (
            importance_df.groupby("Feature", as_index=False)["Importance"]
            .sum()
            .sort_values("Importance", ascending=False)
        )

        max_features = min(20, len(importance_df))
        top_n = st.slider(
            "Number of features to display",
            min_value=1,
            max_value=max_features,
            value=min(10, max_features),
        )

        top = importance_df.head(top_n).sort_values(
            "Importance", ascending=True
        )

        fig = px.bar(
            top,
            x="Importance",
            y="Feature",
            orientation="h",
            title=chart_title,
            labels={"Importance": value_label, "Feature": "Input feature"},
        )
        fig.update_layout(height=max(400, top_n * 35), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            importance_df.head(top_n).reset_index(drop=True),
            use_container_width=True,
        )

except Exception as error:
    st.warning(f"Feature influence chart could not be displayed: {error}")


# ---------------------------------------------------------
# 7. CLASSIFICATION REPORT
# ---------------------------------------------------------

st.divider()

with st.expander("📈 View diagnostic classification report"):
    try:
        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Not Selected", "Selected"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    except Exception as error:
        st.info(f"Report unavailable: {error}")

st.divider()
st.caption(
    "Dataset: Internship_Selection_Dataset.csv. "
    "This synthetic dataset does not establish real-world hiring validity or fairness."
)
