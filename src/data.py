from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder
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


@dataclass(frozen=True)
class LabelEncodedBundle:
    raw: pd.DataFrame
    encoded: pd.DataFrame
    encoders: dict[str, LabelEncoder]
    mappings: dict[str, dict[str, int]]
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


def label_encode_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder], dict[str, dict[str, int]]]:
    """Encode each categorical column with sklearn LabelEncoder, matching the original course approach."""
    encoded = raw.copy()
    encoders: dict[str, LabelEncoder] = {}
    mappings: dict[str, dict[str, int]] = {}

    for column in encoded.columns:
        encoder = LabelEncoder()
        encoded[column] = encoder.fit_transform(encoded[column].astype(str))
        encoders[column] = encoder
        mappings[column] = {label: int(code) for code, label in enumerate(encoder.classes_)}

    return encoded, encoders, mappings


def build_label_encoded_bundle(
    test_size: float = 0.30,
    random_state: int = 0,
    stratify: bool = True,
) -> LabelEncodedBundle:
    raw = load_raw_data()
    encoded, encoders, mappings = label_encode_data(raw)
    features = encoded.drop(columns=[TARGET_COLUMN])
    target = encoded[TARGET_COLUMN]

    stratify_target = target if stratify else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    return LabelEncodedBundle(
        raw=raw,
        encoded=encoded,
        encoders=encoders,
        mappings=mappings,
        features=features,
        target=target,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )


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
