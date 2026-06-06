# Mushroom Safety Classification Decision App

Completed Coursera Streamlit project upgraded into a professional, risk-aware machine learning product demo.

[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-pipelines-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)

> **Live demo:** _add your Streamlit Cloud URL here after deployment_ (e.g. `https://mushroom-safety-classifier.streamlit.app`).

The interface uses a unified dark product theme (branded hero headers, styled metric cards, and a
consistent interactive Plotly chart template across every page) so the project reads as a finished
product rather than a notebook export.

This repository starts from the classic binary mushroom classification lab and refactors it into a portfolio-ready project while preserving the original learning experience: LabelEncoder preprocessing, classifier selection, hyperparameter controls, metric plot choices, raw data display, encoded data display, and model evaluation visuals.

> Educational project only. Do not use this app for real mushroom consumption or safety decisions.

## Why This Project Matters

The original lab proves that a classifier can separate edible and poisonous mushroom records. A stronger applied ML portfolio needs more than a classifier. This version shows how to frame the problem as a decision system where the most important error is a **false-safe prediction**: a poisonous mushroom classified as edible.

## What Was Upgraded

- Replaced the hard-coded Coursera file path with repo-relative data loading.
- Preserved the original course-style `LabelEncoder` workflow and made the target mapping explicit: edible = 0, poisonous = 1.
- Restored the Streamlit model selector for SVM, Logistic Regression, and Random Forest.
- Restored model hyperparameter controls, selected metric plots, raw data view, encoded data view, and LabelEncoder mapping view.
- Added a poisonous-class decision threshold so accuracy, precision, recall, F1, and the confusion matrix are clearly tied to the selected operating threshold.
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
- Streamlit (multipage app + custom theme)
- pandas and NumPy
- scikit-learn pipelines
- Plotly (shared interactive chart theme)
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

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (already configured as `origin`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, choose this repo, branch `main`, and main file `app.py`.
4. Deploy. Streamlit installs `requirements.txt` and pins Python via `runtime.txt` (3.11).
5. Copy the public URL into the **Live demo** badge at the top of this README.

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

## Metric Interpretation

Accuracy, precision, recall, F1, and the confusion matrix are threshold-based metrics. ROC AUC and Average Precision are ranking metrics calculated from model scores across thresholds. It is therefore possible for ROC AUC and Average Precision to round to `1.000` while recall is lower at the selected decision threshold.
