from __future__ import annotations

import streamlit as st

from src.data import build_dataset_bundle
from src.modeling import leaderboard, train_and_evaluate
from src.visuals import confusion_heatmap, metric_table, precision_recall_chart, roc_chart


st.set_page_config(page_title="Model Lab", layout="wide")
st.title("Model Lab")
st.caption("Compare candidate classifiers using safety-oriented metrics, not accuracy alone.")


@st.cache_data(show_spinner=False)
def get_dataset():
    return build_dataset_bundle()


@st.cache_resource(show_spinner=False)
def get_results(threshold: float):
    data = get_dataset()
    return train_and_evaluate(data.x_train, data.x_test, data.y_train, data.y_test, threshold)


threshold = st.slider(
    "Poisonous probability threshold",
    0.05,
    0.95,
    0.35,
    0.05,
    help="Tune the policy threshold to balance false-safe risk against false alarms.",
)

data = get_dataset()
results = get_results(threshold)
board = leaderboard(results)

st.subheader("Leaderboard")
st.dataframe(metric_table(board), use_container_width=True, hide_index=True)

model_names = [result.name for result in results]
selected_name = st.selectbox("Inspect model", model_names)
selected = next(result for result in results if result.name == selected_name)

cols = st.columns(4)
cols[0].metric("False-safe count", f"{int(selected.metrics['false_safe_count'])}")
cols[1].metric("Poisonous recall", f"{selected.metrics['poisonous_recall']:.1%}")
cols[2].metric("ROC AUC", f"{selected.metrics['roc_auc']:.3f}")
cols[3].metric("Balanced accuracy", f"{selected.metrics['balanced_accuracy']:.3f}")

left, right = st.columns(2)
with left:
    st.plotly_chart(confusion_heatmap(selected.confusion), use_container_width=True)
    st.plotly_chart(roc_chart(data.y_test, selected.y_probability), use_container_width=True)
with right:
    st.plotly_chart(precision_recall_chart(data.y_test, selected.y_probability), use_container_width=True)
    st.write(
        "The selected model is evaluated against a configurable decision threshold. "
        "For safety-sensitive classification, the threshold should be reviewed with domain experts and operational constraints."
    )
