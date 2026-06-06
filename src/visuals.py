from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve

from .theme import ACCENT, PANEL, PRIMARY, style_figure


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
        color_discrete_map={"Edible": ACCENT, "Poisonous": PRIMARY},
        text="records",
        title="Dataset class balance",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Records")
    return style_figure(fig, height=340)


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
    fig.update_layout(yaxis_title="False-safe count", xaxis_title="Poisonous recall")
    return style_figure(fig, height=380)


def confusion_heatmap(confusion: object, title: str = "Confusion matrix") -> go.Figure:
    z = confusion.tolist()
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted edible", "Predicted poisonous"],
            y=["Actual edible", "Actual poisonous"],
            colorscale=[[0, PANEL], [1, PRIMARY]],
            text=z,
            texttemplate="%{text}",
            textfont=dict(size=18),
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(title=title, yaxis=dict(autorange="reversed"))
    return style_figure(fig, height=360)


def roc_chart(y_true, probabilities, title: str = "ROC curve") -> go.Figure:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=false_positive_rate,
            y=true_positive_rate,
            mode="lines",
            name="Model",
            line=dict(color=PRIMARY, width=3),
            fill="tozeroy",
            fillcolor="rgba(0, 232, 143, 0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#6f8298", dash="dash"))
    )
    fig.update_layout(title=title, xaxis_title="False positive rate", yaxis_title="True positive rate")
    return style_figure(fig, height=360)


def precision_recall_chart(y_true, probabilities, title: str = "Precision-recall curve") -> go.Figure:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name="Model",
            line=dict(color=ACCENT, width=3),
            fill="tozeroy",
            fillcolor="rgba(45, 140, 255, 0.08)",
        )
    )
    fig.update_layout(title=title, xaxis_title="Recall", yaxis_title="Precision")
    return style_figure(fig, height=360)


def feature_importance_chart(importance: pd.DataFrame, title: str = "Top feature importances") -> go.Figure:
    """Horizontal bar chart of feature importances (expects 'feature' and 'importance' columns)."""
    ordered = importance.sort_values("importance", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=ordered["importance"],
            y=ordered["feature"],
            orientation="h",
            marker=dict(color=ordered["importance"], colorscale=[[0, ACCENT], [1, PRIMARY]]),
            hovertemplate="%{y}<br>Importance: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title="Importance", yaxis_title="")
    return style_figure(fig, height=380)
