
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"

st.title("📊 Model Evaluation")
st.caption(
    "Compare tuned models, review holdout performance, "
    "and explore model feature importance."
)

st.info(
    "This page reads existing evaluation files. "
    "It does not retrain models."
)

# ==========================================================
# 2. LOAD SAVED RESULTS
# ==========================================================
def load_csv(filename, index_col=None):
    path = OUTPUT_DIR / filename

    if not path.exists():
        return None

    try:
        return pd.read_csv(path, index_col=index_col)
    except Exception as error:
        st.warning(f"Could not read {filename}: {error}")
        return None


cv_results = load_csv("cross_validation_results.csv")
holdout_metrics = load_csv("best_model_holdout_metrics.csv")
confusion_matrix_df = load_csv(
    "best_model_holdout_confusion_matrix.csv",
    index_col=0,
)
predictions = load_csv("best_model_holdout_predictions.csv")
selected_summary = load_csv("selected_model_summary.csv")
feature_importance = load_csv("random_forest_feature_importance.csv")

if not OUTPUT_DIR.exists():
    st.error("The outputs folder does not exist.")
    st.stop()

missing_files = []
for filename, result in [
    ("cross_validation_results.csv", cv_results),
    ("best_model_holdout_metrics.csv", holdout_metrics),
    ("best_model_holdout_confusion_matrix.csv", confusion_matrix_df),
]:
    if result is None:
        missing_files.append(filename)

if missing_files:
    st.error(
        "Some evaluation files are missing:\n\n"
        + "\n".join(f"- {name}" for name in missing_files)
    )
    st.write(
        "Make sure the existing tuning script has generated these files "
        "in your project's outputs folder."
    )
    st.stop()

if cv_results.empty or holdout_metrics.empty:
    st.error("One or more required evaluation files are empty.")
    st.stop()

# ==========================================================
# 3. SELECTED MODEL AND HOLDOUT METRICS
# ==========================================================
st.subheader("🏆 Selected Model")

metrics_row = holdout_metrics.iloc[0]
selected_model_name = str(metrics_row.get("Model", "Unknown"))

summary_row = (
    selected_summary.iloc[0]
    if selected_summary is not None and not selected_summary.empty
    else None
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Selected Model", selected_model_name)

for column, metric_name, label in [
    (col2, "accuracy", "Holdout Accuracy"),
    (col3, "balanced_accuracy", "Balanced Accuracy"),
    (col4, "roc_auc", "Holdout ROC-AUC"),
]:
    value = metrics_row.get(metric_name)
    if pd.notna(value):
        column.metric(label, f"{float(value):.2%}")
    else:
        column.metric(label, "N/A")

metric_cols = st.columns(3)

for column, metric_name, label in [
    (metric_cols[0], "precision_selected", "Selected Precision"),
    (metric_cols[1], "recall_selected", "Selected Recall"),
    (metric_cols[2], "f1_selected", "Selected F1 Score"),
]:
    value = metrics_row.get(metric_name)
    if pd.notna(value):
        column.metric(label, f"{float(value):.2%}")
    else:
        column.metric(label, "N/A")

if summary_row is not None:
    st.markdown("**Selected model configuration**")
    st.write(f"Model: {summary_row.get('Model', selected_model_name)}")
    st.write(
        f"Best cross-validation ROC-AUC: "
        f"{float(summary_row['Best_CV_ROC_AUC']):.2%}"
        if pd.notna(summary_row.get("Best_CV_ROC_AUC"))
        else "Cross-validation ROC-AUC: N/A"
    )

    if pd.notna(summary_row.get("Best_Params")):
        with st.expander("View selected model parameters"):
            st.code(str(summary_row["Best_Params"]))

st.caption(
    "Holdout metrics are taken directly from the saved evaluation results. "
    "They describe the test split used by the training script."
)

st.divider()

# ==========================================================
# 4. CROSS-VALIDATION MODEL COMPARISON
# ==========================================================
st.subheader("🔬 Cross-Validation Model Comparison")

st.write(
    "Compare the candidate models using their cross-validation results. "
    "Higher ROC-AUC generally indicates better ranking discrimination."
)

if "Model" not in cv_results.columns:
    st.warning("The cross-validation file has no 'Model' column.")
else:
    available_metrics = {
        "roc_auc_mean": "Mean ROC-AUC",
        "accuracy_mean": "Mean Accuracy",
        "balanced_accuracy_mean": "Mean Balanced Accuracy",
        "f1_mean": "Mean F1 Score",
        "precision_mean": "Mean Precision",
        "recall_mean": "Mean Recall",
    }

    available_metrics = {
        key: label
        for key, label in available_metrics.items()
        if key in cv_results.columns
    }

    if available_metrics:
        selected_metric = st.selectbox(
            "Choose comparison metric",
            options=list(available_metrics.keys()),
            format_func=lambda key: available_metrics[key],
        )

        chart_data = cv_results.copy()
        chart_data[selected_metric] = pd.to_numeric(
            chart_data[selected_metric], errors="coerce"
        )
        chart_data = chart_data.dropna(subset=[selected_metric])
        chart_data = chart_data.sort_values(
            selected_metric, ascending=True
        )

        fig = px.bar(
            chart_data,
            x=selected_metric,
            y="Model",
            orientation="h",
            text=chart_data[selected_metric].map(
                lambda value: f"{value:.1%}"
            ),
            title=available_metrics[selected_metric],
            labels={
                selected_metric: available_metrics[selected_metric],
                "Model": "",
            },
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            height=max(350, len(chart_data) * 65),
            xaxis=dict(range=[0, 1], tickformat=".0%"),
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    display_columns = [
        column for column in [
            "Model",
            "accuracy_mean",
            "balanced_accuracy_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "roc_auc_mean",
            "roc_auc_std",
            "best_params",
        ]
        if column in cv_results.columns
    ]

    with st.expander("View full cross-validation table"):
        st.dataframe(
            cv_results[display_columns],
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# ==========================================================
# 5. CONFUSION MATRIX
# ==========================================================
st.subheader("🎯 Holdout Confusion Matrix")

st.write(
    "The confusion matrix shows correct and incorrect predictions "
    "for the held-out records."
)

if confusion_matrix_df is not None and not confusion_matrix_df.empty:
    try:
        cm = confusion_matrix_df.apply(pd.to_numeric, errors="coerce")
        cm = cm.dropna(axis=0, how="all").dropna(axis=1, how="all")

        if cm.shape == (2, 2):
            values = cm.to_numpy(dtype=int)

            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=values,
                    x=["Predicted Not Selected", "Predicted Selected"],
                    y=["Actual Not Selected", "Actual Selected"],
                    text=values,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=True,
                )
            )

            fig_cm.update_layout(
                title="Actual vs Predicted Outcomes",
                xaxis_title="Predicted Class",
                yaxis_title="Actual Class",
                height=430,
            )

            st.plotly_chart(fig_cm, use_container_width=True)

            tn, fp, fn, tp = values.ravel()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("True Negatives", f"{tn:,}")
            c2.metric("False Positives", f"{fp:,}")
            c3.metric("False Negatives", f"{fn:,}")
            c4.metric("True Positives", f"{tp:,}")
        else:
            st.warning(
                "The saved confusion matrix does not contain a 2 × 2 table."
            )
            st.dataframe(confusion_matrix_df, use_container_width=True)

    except Exception as error:
        st.warning(f"Could not visualize the confusion matrix: {error}")
        st.dataframe(confusion_matrix_df, use_container_width=True)

st.divider()

# ==========================================================
# 6. RANDOM FOREST FEATURE IMPORTANCE
# ==========================================================
st.subheader("🌲 Random Forest Feature Importance")

st.caption(
    "This chart describes the saved Random Forest model, not necessarily "
    "the selected model shown above. Feature importance does not prove "
    "causation."
)

if feature_importance is None or feature_importance.empty:
    st.warning(
        "The Random Forest feature-importance file is not available."
    )
else:
    feature_column = (
        "feature"
        if "feature" in feature_importance.columns
        else "transformed_feature"
        if "transformed_feature" in feature_importance.columns
        else None
    )

    importance_column = (
        "importance"
        if "importance" in feature_importance.columns
        else None
    )

    if feature_column and importance_column:
        importance_data = feature_importance[
            [feature_column, importance_column]
        ].copy()

        importance_data[importance_column] = pd.to_numeric(
            importance_data[importance_column], errors="coerce"
        )
        importance_data = importance_data.dropna(
            subset=[importance_column]
        )

        # Combine transformed one-hot encoded features into original
        # feature names when the saved file contains repeated names.
        importance_data = (
            importance_data.groupby(feature_column, as_index=False)[
                importance_column
            ].sum()
        )

        max_features = min(20, len(importance_data))

        if max_features:
            top_n = st.slider(
                "Number of features to display",
                min_value=1,
                max_value=max_features,
                value=min(10, max_features),
            )

            top = importance_data.nlargest(
                top_n, importance_column
            ).sort_values(importance_column)

            fig = px.bar(
                top,
                x=importance_column,
                y=feature_column,
                orientation="h",
                title=f"Top {top_n} Random Forest Features",
                labels={
                    importance_column: "Relative importance",
                    feature_column: "Feature",
                },
            )

            fig.update_layout(
                height=max(400, top_n * 35),
                yaxis_title="",
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("View feature-importance data"):
                st.dataframe(
                    importance_data.sort_values(
                        importance_column, ascending=False
                    ).reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.warning(
            "The feature-importance CSV does not contain the expected "
            "'feature' and 'importance' columns."
        )

st.divider()

# ==========================================================
# 7. HOLDOUT PREDICTIONS
# ==========================================================
st.subheader("📋 Holdout Prediction Records")

if predictions is not None and not predictions.empty:
    st.caption(
        "These are records from the saved holdout evaluation, "
        "not predictions for newly entered applicants."
    )

    with st.expander("Inspect holdout predictions"):
        if {"Actual", "Predicted"}.issubset(predictions.columns):
            filter_choice = st.selectbox(
                "Filter records",
                [
                    "All records",
                    "Correct predictions",
                    "Incorrect predictions",
                ],
            )

            view = predictions.copy()

            if filter_choice == "Correct predictions":
                view = view[view["Actual"] == view["Predicted"]]
            elif filter_choice == "Incorrect predictions":
                view = view[view["Actual"] != view["Predicted"]]

            st.dataframe(view.head(500), use_container_width=True)
            st.caption(
                f"Showing up to 500 of {len(view):,} matching records."
            )
        else:
            st.dataframe(predictions.head(500), use_container_width=True)

        st.download_button(
            "⬇️ Download Holdout Predictions",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="holdout_predictions.csv",
            mime="text/csv",
        )

st.divider()

# ==========================================================
# 8. INTERPRETATION AND LIMITATIONS
# ==========================================================
st.subheader("📝 Interpretation")

st.markdown(
    """
    - **Accuracy** is the proportion of all predictions that were correct.
    - **Balanced accuracy** gives equal weight to the recall of each class.
    - **Precision** measures how many predicted positive outcomes were correct.
    - **Recall** measures how many actual positive outcomes were identified.
    - **F1 score** balances precision and recall.
    - **ROC-AUC** measures how well the model ranks positive examples above
      negative examples across classification thresholds.
    """
)

st.warning(
    "The dataset is synthetic, and the observed metrics do not establish "
    "real-world hiring validity, fairness, or calibrated probabilities. "
    "Do not use this model to accept, reject, or rank real applicants."
)

st.caption("Internship Selection Prediction Dashboard · Model Evaluation")
