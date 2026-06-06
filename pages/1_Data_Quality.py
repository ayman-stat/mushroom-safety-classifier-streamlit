from __future__ import annotations

import streamlit as st
import plotly.express as px

from src.config import FEATURE_LABELS
from src.data import build_dataset_bundle, class_balance, profile_dataset
from src.theme import PRIMARY, footer, page_setup, style_figure
from src.visuals import class_balance_chart


st.set_page_config(page_title="Data Quality", page_icon="🍄", layout="wide")
page_setup(
    page_title="Data Quality",
    title="🔍 Data Quality Review",
    subtitle="A clean audit layer for the public mushroom classification dataset: schema, "
    "missing values, duplicates, class balance, and feature distributions.",
    pills=[("Schema profile", ""), ("Distributions", "alt")],
)


@st.cache_data(show_spinner=False)
def get_dataset():
    return build_dataset_bundle()


data = get_dataset()
profile = profile_dataset(data.raw)

missing_total = int(profile["missing"].sum())
duplicate_rows = int(data.raw.duplicated().sum())

cols = st.columns(4)
cols[0].metric("Rows", f"{len(data.raw):,}")
cols[1].metric("Columns", f"{data.raw.shape[1]:,}")
cols[2].metric("Missing values", f"{missing_total:,}")
cols[3].metric("Duplicate rows", f"{duplicate_rows:,}")

left, right = st.columns([1.3, 1])
with left:
    st.subheader("Schema Profile")
    st.dataframe(profile, use_container_width=True, hide_index=True, height=360)
with right:
    st.subheader("Class Balance")
    st.plotly_chart(class_balance_chart(class_balance(data.raw)), use_container_width=True)

st.subheader("Feature Distributions")
feature = st.selectbox(
    "Inspect categorical distribution",
    options=list(data.features.columns),
    format_func=lambda value: FEATURE_LABELS.get(value, value),
)

distribution = data.raw[feature].value_counts().reset_index()
distribution.columns = [feature, "records"]
fig = px.bar(
    distribution,
    x=feature,
    y="records",
    text="records",
    title=f"{FEATURE_LABELS.get(feature, feature)} distribution",
    color_discrete_sequence=[PRIMARY],
)
fig.update_traces(textposition="outside", cliponaxis=False)
fig.update_layout(xaxis_title="", yaxis_title="Records")
st.plotly_chart(style_figure(fig, height=380), use_container_width=True)

st.info(
    "Lead-level note: this layer exists so a future production workflow can add validation rules, source checks, and drift monitoring before model scoring."
)
footer()
