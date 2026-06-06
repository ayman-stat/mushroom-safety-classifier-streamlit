from __future__ import annotations

import streamlit as st

from src.config import BUSINESS_FRAME
from src.data import build_dataset_bundle, class_balance
from src.modeling import leaderboard, train_and_evaluate
from src.visuals import class_balance_chart, leaderboard_chart, metric_table


st.set_page_config(
    page_title="Mushroom Safety Classification",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_dataset():
    return build_dataset_bundle()


@st.cache_resource(show_spinner=False)
def get_results(threshold: float):
    data = get_dataset()
    return train_and_evaluate(data.x_train, data.x_test, data.y_train, data.y_test, threshold)


st.title("Mushroom Safety Classification Decision App")
st.caption("A completed Coursera Streamlit project upgraded into a risk-aware ML product demo.")

with st.sidebar:
    st.header("Decision Policy")
    threshold = st.slider(
        "Poisonous probability threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
        help="Lower thresholds classify more records as poisonous, reducing false-safe risk at the cost of more false alarms.",
    )
    st.info(f"{BUSINESS_FRAME['primary_risk']}: {BUSINESS_FRAME['definition']}")

data = get_dataset()
results = get_results(threshold)
board = leaderboard(results)
best = results[0]
balance = class_balance(data.raw)

records = len(data.raw)
poisonous_records = int((data.raw["type"] == "p").sum())
edible_records = records - poisonous_records

metric_cols = st.columns(4)
metric_cols[0].metric("Records", f"{records:,}")
metric_cols[1].metric("Features", f"{data.features.shape[1]}")
metric_cols[2].metric("Poisonous records", f"{poisonous_records:,}")
metric_cols[3].metric("Current false-safe count", f"{int(best.metrics['false_safe_count'])}")

st.subheader("Executive View")
left, right = st.columns([1.05, 1])
with left:
    st.write(
        "This app treats mushroom classification as a safety-sensitive decision problem. "
        "The critical error is not a generic misclassification; it is predicting edible when the record is poisonous."
    )
    st.dataframe(metric_table(board), use_container_width=True, hide_index=True)
with right:
    st.plotly_chart(leaderboard_chart(board), use_container_width=True)

st.subheader("Dataset Overview")
overview_left, overview_right = st.columns([1, 1])
with overview_left:
    st.plotly_chart(class_balance_chart(balance), use_container_width=True)
with overview_right:
    st.write(
        "The original Coursera lab demonstrated binary classification in Streamlit. "
        "This version adds clean packaging, model governance language, threshold policy, and decision metrics."
    )
    st.metric("Edible records", f"{edible_records:,}")
    st.metric("Poisonous share", f"{poisonous_records / records:.1%}")

st.warning(
    "Educational project only. This app must not be used for real mushroom consumption or safety decisions."
)
