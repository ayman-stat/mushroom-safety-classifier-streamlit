from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="Governance", layout="wide")
st.title("Governance & Rollout Notes")
st.caption("How the demo would be controlled before any real-world decision use.")

st.subheader("Decision Controls")
st.markdown(
    """
- Keep the model as decision support, not autonomous approval.
- Treat false-safe predictions as the critical risk.
- Review threshold changes with documented rationale.
- Log model version, data version, threshold, prediction, and final human decision.
- Monitor class balance, feature drift, and false-safe incidents over time.
"""
)

st.subheader("Portfolio Positioning")
st.write(
    "This project demonstrates how a beginner Streamlit classifier can be upgraded into a more mature ML decision product: "
    "clear risk framing, repeatable data loading, candidate model comparison, threshold policy, scoring workflow, and governance documentation."
)

st.subheader("Limits")
st.markdown(
    """
- Public educational dataset, not a real safety-certified product.
- No external validation against live mushroom observations.
- No image recognition or expert mycology review.
- Must not be used for consumption decisions.
"""
)

st.subheader("Next Enhancements")
st.markdown(
    """
- Add automated data validation with Great Expectations or pandera.
- Package model training as a CLI pipeline.
- Add MLflow experiment tracking and model registry notes.
- Add a lightweight API endpoint for batch scoring.
- Deploy a public Streamlit Cloud demo after repository publication.
"""
)
