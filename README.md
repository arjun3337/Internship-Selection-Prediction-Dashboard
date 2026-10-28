# Internship Selection Prediction & Employability Analytics

An interactive machine learning and data analytics dashboard built with **Python and Streamlit** to explore internship applicant data, understand selection patterns, evaluate a classification model, and analyze employability-related indicators.

## Project Overview

The Internship Selection Prediction & Employability Analytics project brings applicant exploration, interactive visualizations, machine learning evaluation, individual predictions, and data-quality checks together in one dashboard.

The project uses `Internship_Selection_Dataset.csv`, containing 10,000 applicant records and 21 columns.

**Project type:** AI / Machine Learning and Data Analytics  
**Internship ID:** CITS4485  
**Interface:** Streamlit web dashboard

## Dashboard Modules

The application contains seven pages:

1. **Executive Overview** — applicant KPIs, selection distribution, academic comparisons, and skill analysis.
2. **Applicant Explorer** — search, filter, inspect, and export applicant records.
3. **Selection Patterns** — analyze recorded selection outcomes across applicant characteristics.
4. **Model Evaluation** — review model metrics, cross-validation results, confusion matrix, and feature analysis.
5. **Applicant Prediction** — enter applicant details and view the model's predicted outcome and scores.
6. **Employability Insights** — explore skill indicators, academic performance, and development areas.
7. **Data Quality** — inspect missing values, duplicate records, data types, potential outliers, and configured value ranges.

## Dataset

**File:** `Internship_Selection_Dataset.csv`

- Records: 10,000
- Columns: 21
- Target column: `selected`
- `selected = 1`: Selected
- `selected = 0`: Not Selected

The dataset includes academic performance, internship experience, projects, certifications, coding and interview scores, communication indicators, college tier, and other applicant characteristics.

### Student ID handling

The student ID is retained for applicant lookup and identification. It is excluded from machine learning training because it is an identifier rather than a meaningful predictive feature.

### Target distribution

| Recorded outcome | Applicants | Percentage |
|---|---:|---:|
| Selected | 7,374 | 73.74% |
| Not Selected | 2,626 | 26.26% |
| **Total** | **10,000** | **100%** |

## Model Evaluation

The current evaluation reports the following results:

| Metric | Result |
|---|---:|
| Accuracy | 56.75% |
| ROC-AUC | 60.19% |

These results indicate limited predictive performance. The target classes are imbalanced, so accuracy should be interpreted alongside metrics such as precision, recall, F1-score, balanced accuracy, and ROC-AUC.

Model scores are outputs of the trained model and are not guaranteed probabilities of real-world internship selection.

## Technology Stack

- **Python** — application and model logic
- **Streamlit** — interactive web dashboard
- **Pandas** — data processing and analysis
- **Plotly** — interactive charts
- **Scikit-learn** — machine learning pipeline and evaluation
- **Joblib** — model serialization and loading

## Project Structure

```text
Internship_Selection_Dashboard/
├── app.py
├── README.md
├── requirements.txt
├── Internship_Selection_Dataset.csv
├── models/
│   └── best_internship_model.pkl
├── outputs/
│   ├── cross_validation_results.csv
│   ├── best_model_holdout_metrics.csv
│   ├── best_model_holdout_confusion_matrix.csv
│   ├── best_model_holdout_predictions.csv
│   ├── selected_model_summary.csv
│   └── random_forest_feature_importance.csv
├── pages/
│   ├── 1_Executive_Overview.py
│   ├── 2_Applicant_Explorer.py
│   ├── 3_Selection_Patterns.py
│   ├── 4_Model_Evaluation.py
│   ├── 5_Applicant_Prediction.py
│   ├── 6_Employability_Insights.py
│   ├── 7_Data_Quality.py
│   └── check_dataset.py
└── screenshots/
```

*This is the intended structure. Adjust the tree to match the files and folders actually present in your project.*

## Run on Windows

### 1. Open the project

Open the project root folder in VS Code and launch a terminal in that directory.

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Launch the dashboard

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal if the browser does not launch automatically.

## Using the Dashboard

1. Launch the Streamlit application.
2. Select a dashboard page from the sidebar.
3. Explore applicants and selection patterns using filters.
4. Review model evaluation metrics and visualizations.
5. Enter applicant information on the prediction page.
6. Download available applicant and analytical CSV reports.
7. Use the Data Quality page to inspect potential dataset issues.

## Limitations and Responsible Use

- The dataset is synthetic and may not represent real recruitment patterns.
- The current model has limited predictive performance.
- The selection classes are imbalanced.
- Feature associations do not establish causation.
- Model scores should not be interpreted as calibrated real-world hiring probabilities.
- Predictions are intended for educational demonstration, not as the sole basis for hiring decisions.
- Real-world deployment would require representative data, additional validation, fairness assessments, privacy safeguards, and human oversight.

## Future Improvements

- Improve model performance and evaluate alternative algorithms.
- Strengthen cross-validation and test-set evaluation.
- Analyze class-specific errors and probability calibration.
- Add automated tests and stronger input validation.
- Assess fairness and model behavior across relevant applicant groups.
- Improve deployment, documentation, and reproducibility.

## Author


## Author

- **Name:** Arjun P.
- **USN:** 25MCAR0025
- **Course:** Master of Computer Applications (MCA)
- **University:** Jain (Deemed-to-be University)
- **Project Title:** Internship Selection Prediction & Employability Analytics
- **Project Type:** Academic Project
- **Subject:** Predictive Analysis and Data Visualization
- **Faculty Guide:** Prof. Rahul Pawar


## License and Dataset Attribution

Add the appropriate license and dataset attribution according to the dataset's original source and the requirements of your internship submission.
