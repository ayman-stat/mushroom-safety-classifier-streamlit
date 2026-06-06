from src.data import build_dataset_bundle, build_label_encoded_bundle
from src.modeling import leaderboard, train_and_evaluate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def test_dataset_loads_from_relative_path():
    data = build_dataset_bundle()
    assert len(data.raw) > 0
    assert data.features.shape[1] == 22
    assert set(data.target.unique()) == {0, 1}


def test_label_encoder_target_mapping_is_explicit():
    data = build_label_encoded_bundle()
    assert data.mappings["type"] == {"e": 0, "p": 1}
    assert set(data.encoded["type"].unique()) == {0, 1}


def test_course_style_label_encoded_models_train():
    data = build_label_encoded_bundle()
    models = [
        SVC(C=1.0, kernel="rbf", gamma="scale", probability=True),
        LogisticRegression(C=1.0, max_iter=500),
        RandomForestClassifier(n_estimators=25, max_depth=12, random_state=0),
    ]

    for model in models:
        model.fit(data.x_train, data.y_train)
        predictions = model.predict(data.x_test)
        assert accuracy_score(data.y_test, predictions) >= 0.85


def test_models_train_and_return_leaderboard():
    data = build_dataset_bundle()
    results = train_and_evaluate(data.x_train, data.x_test, data.y_train, data.y_test, threshold=0.35)
    board = leaderboard(results)
    assert len(results) >= 3
    assert "false_safe_count" in board.columns
    assert board["roc_auc"].min() >= 0.5
