# Mushroom Safety Classification Decision App

Completed Coursera Streamlit project upgraded into a professional, risk-aware machine learning product demo.

This repository starts from the classic binary mushroom classification lab and refactors it into a portfolio-ready project while preserving the original learning experience: LabelEncoder preprocessing, classifier selection, hyperparameter controls, metric plot choices, raw data display, encoded data display, and model evaluation visuals.

> Educational project only. Do not use this app for real mushroom consumption or safety decisions.

## Why This Project Matters

The original lab proves that a classifier can separate edible and poisonous mushroom records. A stronger applied ML portfolio needs more than a classifier. This version shows how to frame the problem as a decision system where the most important error is a **false-safe prediction**: a poisonous mushroom classified as edible.

## What Was Upgraded

- Replaced the hard-coded Coursera file path with repo-relative data loading.
- Preserved the original course-style `LabelEncoder` workflow and made the target mapping explicit: edible = 0, poisonous = 1.
- Restored the Streamlit model selector for SVM, Logistic Regression, and Random Forest.
- Restored model hyperparameter controls, selected metric plots, raw data view, encoded data view, and LabelEncoder mapping view.
- Added a reusable `src/` package for data loading, preprocessing, modeling, evaluation, and visuals.
- Added multiple candidate models: Logistic Regression, Random Forest, Gradient Boosting, and SVM.
- Added risk-aware model comparison using poisonous recall, false-safe count, false-safe rate, ROC AUC, average precision, and balanced accuracy.
- Added a threshold policy control to show how operating decisions change model risk.
- Added batch and manual scoring views.
- Added model card, architecture notes, governance notes, and smoke tests.

## App Pages

- `app.py` - course-compatible interactive classifier workbench with model choices, hyperparameters, LabelEncoder mappings, metrics, and visuals.
- `pages/1_Data_Quality.py` - schema, missing values, duplicates, and feature distributions.
- `pages/2_Model_Lab.py` - model comparison, confusion matrix, ROC, and precision-recall curves.
- `pages/3_Safety_Scoring.py` - manual scoring and batch scoring preview.
- `pages/4_Governance_Rollout.py` - rollout controls, limitations, and next enhancements.

## Tech Stack

- Python
- Streamlit
- pandas and NumPy
- scikit-learn pipelines
- Plotly
- pytest

## Repository Structure

```text
.
├── app.py
├── data/raw/mushrooms.csv
├── docs/
├── pages/
├── src/
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── runtime.txt
```

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

## Run Tests

```bash
pytest
```

## Portfolio Positioning

This project is intentionally simple in domain but mature in execution. It demonstrates:

- Translating a model into a decision-support workflow.
- Choosing metrics based on risk, not accuracy alone.
- Building a clean Streamlit app that can be deployed.
- Communicating limitations and governance for public portfolio work.

## Data

The dataset is a public categorical mushroom classification dataset included for educational use. The app keeps the original target labels:

- `e` = edible
- `p` = poisonous

The model maps poisonous records to the positive class because that is the safety-critical outcome.
