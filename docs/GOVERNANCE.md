# Governance Notes

## Safety Framing

This project intentionally reframes a simple binary classifier as a safety-sensitive decision system. The important question is not whether the model reaches high accuracy; it is whether the operating policy avoids false-safe outcomes.

## False-Safe Error

A false-safe error happens when:

```text
Actual class = poisonous
Predicted decision = edible or low-risk
```

This is more severe than a false alarm because it creates unsafe confidence.

## Threshold Policy

The Streamlit app allows users to adjust the poisonous probability threshold. Lower thresholds generally reduce false-safe risk but increase false alarms. In a real project, this threshold should be approved through domain review and monitored after deployment.

## Deployment Readiness Checklist

- Data validation rules documented.
- Train/test split and random seed controlled.
- Model metrics reviewed against risk policy.
- Threshold selected and versioned.
- Model card published.
- Human review path defined.
- Monitoring plan defined.

## Portfolio Ethics

The project is public because it uses educational data and contains no employer or client data. Its limitations are intentionally visible to avoid implying a real safety-certified product.
