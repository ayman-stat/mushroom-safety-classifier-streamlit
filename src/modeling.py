from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .config import RANDOM_STATE
from .features import build_preprocessor


@dataclass(frozen=True)
class ModelResult:
    name: str
    pipeline: Pipeline
    y_probability: np.ndarray
    y_prediction: np.ndarray
    metrics: dict[str, float]
    confusion: np.ndarray


def candidate_models(random_state: int = RANDOM_STATE) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=9,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "Support Vector Machine": SVC(
            C=2.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def build_pipeline(model: object, categorical_columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_columns)),
            ("model", model),
        ]
    )


def predict_with_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> tuple[dict[str, float], np.ndarray]:
    predictions = predict_with_threshold(probabilities, threshold)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    true_edible, false_alarm, false_safe, true_poisonous = matrix.ravel()
    poison_count = max(int((y_true == 1).sum()), 1)

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "poisonous_precision": precision_score(y_true, predictions, pos_label=1, zero_division=0),
        "poisonous_recall": recall_score(y_true, predictions, pos_label=1, zero_division=0),
        "false_safe_count": float(false_safe),
        "false_safe_rate": false_safe / poison_count,
        "false_alarm_count": float(false_alarm),
        "true_edible": float(true_edible),
        "true_poisonous": float(true_poisonous),
    }
    return metrics, matrix


def train_and_evaluate(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    threshold: float = 0.35,
) -> list[ModelResult]:
    categorical_columns = list(x_train.columns)
    results: list[ModelResult] = []

    for name, estimator in candidate_models().items():
        pipeline = build_pipeline(estimator, categorical_columns)
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics, matrix = evaluate_predictions(y_test, probabilities, threshold)
        results.append(
            ModelResult(
                name=name,
                pipeline=pipeline,
                y_probability=probabilities,
                y_prediction=predict_with_threshold(probabilities, threshold),
                metrics=metrics,
                confusion=matrix,
            )
        )

    return sorted(
        results,
        key=lambda result: (
            result.metrics["false_safe_count"],
            -result.metrics["poisonous_recall"],
            -result.metrics["roc_auc"],
        ),
    )


def leaderboard(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "model": result.name,
                "false_safe_count": int(result.metrics["false_safe_count"]),
                "false_safe_rate": result.metrics["false_safe_rate"],
                "poisonous_recall": result.metrics["poisonous_recall"],
                "poisonous_precision": result.metrics["poisonous_precision"],
                "roc_auc": result.metrics["roc_auc"],
                "average_precision": result.metrics["average_precision"],
                "balanced_accuracy": result.metrics["balanced_accuracy"],
                "accuracy": result.metrics["accuracy"],
            }
        )
    return pd.DataFrame(rows)


def score_records(pipeline: Pipeline, records: pd.DataFrame, threshold: float) -> pd.DataFrame:
    probabilities = pipeline.predict_proba(records)[:, 1]
    scored = records.copy()
    scored["poisonous_probability"] = probabilities
    scored["decision"] = np.where(probabilities >= threshold, "Review as poisonous risk", "Lower-risk edible prediction")
    return scored.sort_values("poisonous_probability", ascending=False)
