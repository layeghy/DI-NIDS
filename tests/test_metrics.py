import numpy as np

from dinids.metrics import one_class_metrics, to_one_class_labels


def test_label_conversion() -> None:
    converted = to_one_class_labels(np.array([0, 1, 0, 1]))
    np.testing.assert_array_equal(converted, np.array([1, -1, 1, -1]))


def test_metrics_name_attack_and_benign_classes() -> None:
    expected = np.array([-1, -1, 1, 1])
    predicted = np.array([-1, 1, -1, 1])
    scores = one_class_metrics(
        expected,
        predicted,
        decision_scores=np.array([-0.8, 0.2, -0.1, 0.9]),
    )
    assert scores["accuracy"] == 0.5
    assert scores["attack"]["precision"] == 0.5
    assert scores["attack"]["recall"] == 0.5
    assert scores["benign"]["f1"] == 0.5
    assert scores["confusion_matrix"] == {
        "attack_as_attack": 1,
        "attack_as_benign": 1,
        "benign_as_attack": 1,
        "benign_as_benign": 1,
    }
    assert scores["attack_roc_auc"] == 0.75
