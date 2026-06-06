from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, NEGATIVE_LABEL, POSITIVE_LABEL, RANDOM_STATE, TARGET_COLUMN


@dataclass(frozen=True)
class DatasetBundle:
    raw: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the public mushroom classification dataset from a repo-relative path."""
    data = pd.read_csv(path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset.")
    return data


def prepare_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = raw.drop(columns=[TARGET_COLUMN]).copy()
    target = raw[TARGET_COLUMN].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1})

    if target.isna().any():
        unexpected = sorted(raw.loc[target.isna(), TARGET_COLUMN].dropna().unique())
        raise ValueError(f"Unexpected class labels found: {unexpected}")

    for column in features.columns:
        features[column] = features[column].astype("category")

    return features, target.astype(int)


def build_dataset_bundle(test_size: float = 0.25, random_state: int = RANDOM_STATE) -> DatasetBundle:
    raw = load_raw_data()
    features, target = prepare_features(raw)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return DatasetBundle(raw, features, target, x_train, x_test, y_train, y_test)


def profile_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    profile = pd.DataFrame(
        {
            "column": raw.columns,
            "dtype": [str(raw[column].dtype) for column in raw.columns],
            "missing": [int(raw[column].isna().sum()) for column in raw.columns],
            "missing_rate": [float(raw[column].isna().mean()) for column in raw.columns],
            "unique_values": [int(raw[column].nunique(dropna=True)) for column in raw.columns],
        }
    )
    return profile


def class_balance(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw[TARGET_COLUMN]
        .map({NEGATIVE_LABEL: "Edible", POSITIVE_LABEL: "Poisonous"})
        .value_counts()
        .rename_axis("class")
        .reset_index(name="records")
    )
