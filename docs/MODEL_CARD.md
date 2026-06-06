# Model Card

## Model Purpose

Classify public mushroom records as edible or poisonous using categorical physical attributes.

This is an educational decision-support demo, not a certified safety system.

## Intended Use

- Portfolio demonstration of applied ML workflow design.
- Streamlit app demonstration for classification, threshold tuning, and model comparison.
- Example of risk-aware metrics for asymmetric error costs.

## Not Intended For

- Real mushroom identification.
- Food safety decisions.
- Medical, agricultural, or regulatory use.
- Autonomous decisions without expert review.

## Target Definition

- Negative class: edible
- Positive class: poisonous

Poisonous is the positive class because missing a poisonous mushroom is the highest-risk error.

## Candidate Models

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine

All candidates are trained through scikit-learn pipelines with one-hot encoding for categorical features.

## Primary Metrics

- False-safe count: poisonous records predicted as edible.
- False-safe rate: false-safe count divided by poisonous records.
- Poisonous recall: ability to capture poisonous records.
- ROC AUC and average precision: ranking quality.
- Balanced accuracy: class-balanced correctness.

Accuracy is reported but is not the primary decision metric.

## Risks and Limitations

- Public educational dataset may not represent real-world mushroom variation.
- The model uses tabular categorical observations, not images or expert biological assessment.
- No external validation has been performed.
- The app must not be used for consumption decisions.

## Governance Controls

- Keep a human review step for safety-sensitive outputs.
- Log model version, data version, threshold, score timestamp, and final decision.
- Monitor false-safe incidents as the key harm metric.
- Review threshold changes before promotion.
