# Architecture

## Objective

Turn a beginner Streamlit binary-classification lab into a clean ML decision-support app.

## Flow

```text
Raw mushroom CSV
        |
        v
Data quality profile
        |
        v
Categorical feature preparation
        |
        v
scikit-learn pipelines with one-hot encoding
        |
        v
Candidate model comparison
        |
        v
Threshold-based poisonous risk decision
        |
        v
Manual and batch scoring views
```

## Design Choices

- The project uses scikit-learn `Pipeline` and `ColumnTransformer` so preprocessing and model inference stay together.
- The target is mapped as poisonous = positive class because false-safe decisions are the main risk.
- The Streamlit interface exposes threshold tuning so model performance can be discussed as an operating policy, not a fixed score.
- Documentation is included because senior ML work needs communication around limitations, controls, and deployment readiness.

## Future Production Enhancements

- Add data contracts and validation checks.
- Persist model artifacts with model version metadata.
- Add experiment tracking with MLflow.
- Add scheduled retraining and monitoring.
- Add audit logs for scored records and human review outcomes.
