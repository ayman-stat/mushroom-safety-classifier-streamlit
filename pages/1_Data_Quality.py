from __future__ import annotations

import streamlit as st
import plotly.express as px

from src.config import FEATURE_LABELS
from src.data import build_dataset_bundle, profile_dataset


st.set_page_config(page_title="Data Quality", layout="wide")
st.title("Data Quality Review")
st.caption("A clean audit layer for the public mushroom classification dataset.")


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

st.subheader("Schema Profile")
st.dataframe(profile, use_container_width=True, hide_index=True)

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
    color_discrete_sequence=["#00e88f"],
)
fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
st.plotly_chart(fig, use_container_width=True)

st.info(
    "Lead-level note: this layer exists so a future production workflow can add validation rules, source checks, and drift monitoring before model scoring."
)
