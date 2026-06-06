from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from src.config import TARGET_COLUMN
from src.data import build_label_encoded_bundle
from src.theme import footer, page_setup
from src.visuals import (
    confusion_heatmap,
    feature_importance_chart,
    precision_recall_chart,
    roc_chart,
)


CLASS_NAMES = ["edible", "poisonous"]
POSITIVE_CLASS = 1


st.set_page_config(
    page_title="Mushroom Safety Classifier",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_setup(
    page_title="Mushroom Safety Classifier",
    title="🍄 Mushroom Safety Classifier",
    subtitle=(
        "A course-style binary classifier upgraded into a risk-aware decision tool. Train SVM, "
        "Logistic Regression, or Random Forest, then tune the poisonous decision threshold and read "
        "the metrics that actually matter for safety."
    ),
    pills=[("Interactive classifier", ""), ("Risk-aware metrics", "alt"), ("scikit-learn", "")],
)


@st.cache_data(show_spinner=False)
def get_encoded_data(test_size: float, random_state: int, stratify: bool):
    return build_label_encoded_bundle(
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def build_model(classifier: str):
    st.sidebar.subheader("Model Hyperparameters")

    if classifier == "Support Vector Machine (SVM)":
        c_value = st.sidebar.number_input("C (regularization)", 0.01, 20.0, value=1.0, step=0.01)
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear", "poly", "sigmoid"), index=0)
        gamma = st.sidebar.radio("Gamma", ("scale", "auto"), index=0)
        return SVC(C=c_value, kernel=kernel, gamma=gamma, probability=True)

    if classifier == "Logistic Regression":
        c_value = st.sidebar.number_input("C (regularization)", 0.01, 20.0, value=1.0, step=0.01)
        max_iter = st.sidebar.slider("Maximum iterations", 100, 2000, value=500, step=100)
        solver = st.sidebar.selectbox("Solver", ("lbfgs", "liblinear"), index=0)
        class_weight = st.sidebar.selectbox("Class weight", ("None", "balanced"), index=0)
        return LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            solver=solver,
            class_weight=None if class_weight == "None" else "balanced",
        )

    n_estimators = st.sidebar.number_input("Number of trees", 50, 5000, value=300, step=50)
    max_depth = st.sidebar.number_input("Maximum tree depth", 1, 50, value=12, step=1)
    min_samples_split = st.sidebar.slider("Minimum samples split", 2, 20, value=2)
    min_samples_leaf = st.sidebar.slider("Minimum samples leaf", 1, 20, value=1)
    bootstrap = st.sidebar.checkbox("Bootstrap samples", value=True)
    criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy", "log_loss"), index=0)
    return RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        bootstrap=bootstrap,
        criterion=criterion,
        n_jobs=-1,
        random_state=0,
    )


def get_scores(model, x_test: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, POSITIVE_CLASS]
    return model.decision_function(x_test)


def plot_selected_metrics(metrics: list[str], y_test, y_pred, y_score):
    if not metrics:
        return

    st.subheader("Visual Diagnostics")
    st.caption("Interactive plots for the selected model and current decision threshold.")

    figures = []
    if "Confusion Matrix" in metrics:
        matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
        figures.append(
            (
                confusion_heatmap(matrix),
                "Rows are actual classes; columns are predicted classes.",
            )
        )
    if "ROC Curve" in metrics:
        figures.append(
            (
                roc_chart(y_test, y_score),
                "Ranking performance across classification thresholds.",
            )
        )
    if "Precision-Recall Curve" in metrics:
        figures.append(
            (
                precision_recall_chart(y_test, y_score),
                "Precision and recall trade-off for the poisonous class.",
            )
        )

    grid = st.columns(2 if len(figures) > 1 else 1)
    for index, (fig, caption) in enumerate(figures):
        with grid[index % len(grid)]:
            st.plotly_chart(fig, use_container_width=True)
            st.caption(caption)


with st.sidebar:
    st.title("Classifier Controls")
    st.markdown("Are your mushrooms edible or poisonous?")

    st.subheader("Data Split")
    test_size = st.slider("Test size", 0.10, 0.50, value=0.30, step=0.05)
    random_state = st.number_input("Random state", 0, 999, value=0, step=1)
    stratify = st.checkbox("Stratify train/test split", value=True)

    st.subheader("Choose Classifier")
    classifier = st.selectbox(
        "Classifier",
        ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"),
    )
    model = build_model(classifier)

    selected_metrics = st.multiselect(
        "What metrics should be plotted?",
        ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
        default=("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
    )
    decision_threshold = st.slider(
        "Poisonous decision threshold",
        0.05,
        0.95,
        value=0.50,
        step=0.05,
        help="Threshold applied to the poisonous-class probability for accuracy, precision, recall, F1, and confusion matrix.",
    )

    show_raw_data = st.checkbox("Show raw data", value=False)
    show_encoded_data = st.checkbox("Show label-encoded data", value=False)
    show_mappings = st.checkbox("Show LabelEncoder mappings", value=True)
    run_model = st.button("Classify", type="primary", use_container_width=True)

bundle = get_encoded_data(test_size, int(random_state), stratify)

overview_cols = st.columns(4)
overview_cols[0].metric("Rows", f"{len(bundle.raw):,}")
overview_cols[1].metric("Features", f"{bundle.features.shape[1]}")
overview_cols[2].metric("Train rows", f"{len(bundle.x_train):,}")
overview_cols[3].metric("Test rows", f"{len(bundle.x_test):,}")

st.info(
    "This home page intentionally follows the original Coursera lab pattern: LabelEncoder preprocessing, model selection, "
    "hyperparameter controls, metric selection, and optional raw/encoded data views."
)

if show_mappings:
    st.subheader("LabelEncoder Target Mapping")
    target_mapping = bundle.mappings[TARGET_COLUMN]
    st.write(
        {
            "edible": target_mapping.get("e"),
            "poisonous": target_mapping.get("p"),
        }
    )

if show_raw_data:
    st.subheader("Raw Mushroom Dataset")
    st.dataframe(bundle.raw, use_container_width=True)

if show_encoded_data:
    st.subheader("Label-Encoded Dataset")
    st.dataframe(bundle.encoded, use_container_width=True)

if not run_model:
    st.warning("Choose a classifier and click **Classify** to train the model and show metrics.")
    footer()
    st.stop()

with st.spinner(f"Training {classifier}..."):
    model.fit(bundle.x_train, bundle.y_train)
    y_score = get_scores(model, bundle.x_test)
    y_pred = (y_score >= decision_threshold).astype(int)

accuracy = accuracy_score(bundle.y_test, y_pred)
precision = precision_score(bundle.y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
recall = recall_score(bundle.y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
f1 = f1_score(bundle.y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
roc_auc = roc_auc_score(bundle.y_test, y_score)
average_precision = average_precision_score(bundle.y_test, y_score)
true_edible, false_alarm, false_safe, true_poisonous = confusion_matrix(
    bundle.y_test,
    y_pred,
    labels=[0, 1],
).ravel()

st.subheader(f"{classifier} Results")
st.caption(f"Metrics below use a poisonous decision threshold of {decision_threshold:.2f}.")
metric_cols = st.columns(6)
metric_cols[0].metric("Accuracy", f"{accuracy:.3f}")
metric_cols[1].metric("Precision", f"{precision:.3f}")
metric_cols[2].metric("Recall", f"{recall:.3f}")
metric_cols[3].metric("F1", f"{f1:.3f}")
metric_cols[4].metric("ROC AUC", f"{roc_auc:.3f}")
metric_cols[5].metric("Avg Precision", f"{average_precision:.3f}")

confusion_cols = st.columns(4)
confusion_cols[0].metric("True edible", f"{true_edible:,}")
confusion_cols[1].metric("False alarms", f"{false_alarm:,}")
confusion_cols[2].metric("False-safe ⚠️", f"{false_safe:,}")
confusion_cols[3].metric("True poisonous", f"{true_poisonous:,}")

with st.expander("Why ROC AUC can be near 1.000 while recall is lower"):
    st.write(
        "Accuracy, precision, recall, F1, and the confusion matrix depend on the selected decision threshold. "
        "ROC AUC and Average Precision evaluate how well the model ranks poisonous records above edible records across many thresholds. "
        "So a model can have excellent ranking metrics while still missing some poisonous records at the current threshold."
    )

plot_selected_metrics(selected_metrics, bundle.y_test, y_pred, y_score)

if classifier == "Random Forest":
    st.subheader("Random Forest Feature Importance")
    importance = (
        pd.DataFrame(
            {
                "feature": bundle.x_train.columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(10)
    )
    left, right = st.columns([1.4, 1])
    with left:
        st.plotly_chart(feature_importance_chart(importance), use_container_width=True)
    with right:
        st.dataframe(importance, use_container_width=True, hide_index=True)

with st.expander("Model parameters"):
    st.json(model.get_params())

footer()
