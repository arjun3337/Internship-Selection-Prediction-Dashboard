
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def build_preprocessor(numerical_columns, categorical_columns):
    """
    Build preprocessing steps for numerical and categorical features.
    """

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    return preprocessor


def build_model(
    numerical_columns,
    categorical_columns,
    C=1.0,
    kernel="rbf"
):
    """
    Build the preprocessing and SVM classification pipeline.
    """

    preprocessor = build_preprocessor(
        numerical_columns,
        categorical_columns
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", SVC(
            C=C,
            kernel=kernel,
            class_weight="balanced",
            random_state=42
        ))
    ])

    return model
