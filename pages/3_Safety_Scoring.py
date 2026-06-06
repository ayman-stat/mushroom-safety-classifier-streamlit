from __future__ import annotations

import streamlit as st

from src.config import FEATURE_LABELS
from src.data import build_dataset_bundle
from src.modeling import score_records, train_and_evaluate
from src.theme import footer, page_setup


st.set_page_config(page_title="Safety Scoring", page_icon="🍄", layout="wide")
page_setup(
    page_title="Safety Scoring",
    title="🛡️ Safety Scoring Workbench",
    subtitle="Score a single record by hand or a batch from the test set, and watch how the "
    "threshold policy changes each decision.",
    pills=[("Manual scoring", ""), ("Batch preview", "alt")],
)


@st.cache_data(show_spinner=False)
def get_dataset():
    return build_dataset_bundle()


@st.cache_resource(show_spinner=False)
def get_results(threshold: float):
    data = get_dataset()
    return train_and_evaluate(data.x_train, data.x_test, data.y_train, data.y_test, threshold)


threshold = st.slider("Poisonous probability threshold", 0.05, 0.95, 0.35, 0.05)
data = get_dataset()
best = get_results(threshold)[0]

st.subheader("Manual Record Builder")
with st.form("record_form"):
    input_values = {}
    columns = st.columns(3)
    for index, feature in enumerate(data.features.columns):
        options = sorted(data.raw[feature].dropna().unique().tolist())
        input_values[feature] = columns[index % 3].selectbox(
            FEATURE_LABELS.get(feature, feature),
            options=options,
            key=feature,
        )
    submitted = st.form_submit_button("Score record")

if submitted:
    record = data.features.iloc[[0]].copy()
    for feature, value in input_values.items():
        record.loc[record.index[0], feature] = value
    scored = score_records(best.pipeline, record, threshold)
    probability = float(scored["poisonous_probability"].iloc[0])
    flagged = probability >= threshold

    result_cols = st.columns([1, 2])
    result_cols[0].metric("Poisonous probability", f"{probability:.1%}")
    with result_cols[1]:
        st.write("")
        if flagged:
            st.error(f"⚠️ Flagged: review as poisonous risk (≥ {threshold:.0%} threshold).")
        else:
            st.success(f"Lower-risk edible prediction (< {threshold:.0%} threshold).")

st.subheader("Batch Scoring Preview")
sample_size = st.slider("Rows to score from test sample", 10, 200, 50, 10)
scored_sample = score_records(best.pipeline, data.x_test.head(sample_size), threshold)
st.dataframe(scored_sample, use_container_width=True, hide_index=True, height=420)

st.info(
    "In a production setting, this scoring layer would write outputs to an audited table with model version, threshold version, score timestamp, and reviewer action."
)
footer()
