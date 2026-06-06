from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve


def format_rate(value: float) -> str:
    return f"{value:.1%}"


def metric_table(data: pd.DataFrame) -> pd.DataFrame:
    formatted = data.copy()
    percent_columns = [
        "false_safe_rate",
        "poisonous_recall",
        "poisonous_precision",
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "accuracy",
    ]
    for column in percent_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
    return formatted


def class_balance_chart(balance: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        balance,
        x="class",
        y="records",
        color="class",
        color_discrete_map={"Edible": "#2d8cff", "Poisonous": "#00e88f"},
        text="records",
        title="Dataset class balance",
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def leaderboard_chart(leaderboard: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        leaderboard,
        x="poisonous_recall",
        y="false_safe_count",
        size="roc_auc",
        color="model",
        hover_data=["poisonous_precision", "balanced_accuracy"],
        title="Model comparison: recall vs false-safe risk",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), yaxis_title="False-safe count")
    return fig


def confusion_heatmap(confusion: object) -> go.Figure:
    z = confusion.tolist()
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted edible", "Predicted poisonous"],
            y=["Actual edible", "Actual poisonous"],
            colorscale=[[0, "#102033"], [1, "#00e88f"]],
            text=z,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(title="Confusion matrix", margin=dict(l=20, r=20, t=60, b=20))
    return fig


def roc_chart(y_true, probabilities) -> go.Figure:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=false_positive_rate, y=true_positive_rate, mode="lines", name="Model"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    return fig


def precision_recall_chart(y_true, probabilities) -> go.Figure:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Model"))
    fig.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision")
    return fig
