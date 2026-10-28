from pathlib import Path
import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The dataset is empty.")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = result.columns.str.strip()
    return result


def validate_target(df: pd.DataFrame, target: str = "selected") -> bool:
    return target in df.columns
