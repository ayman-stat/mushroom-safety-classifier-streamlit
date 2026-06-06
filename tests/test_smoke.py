from src.data import build_dataset_bundle
from src.modeling import leaderboard, train_and_evaluate


def test_dataset_loads_from_relative_path():
    data = build_dataset_bundle()
    assert len(data.raw) > 0
    assert data.features.shape[1] == 22
    assert set(data.target.unique()) == {0, 1}


def test_models_train_and_return_leaderboard():
    data = build_dataset_bundle()
    results = train_and_evaluate(data.x_train, data.x_test, data.y_train, data.y_test, threshold=0.35)
    board = leaderboard(results)
    assert len(results) >= 3
    assert "false_safe_count" in board.columns
    assert board["roc_auc"].min() >= 0.5
