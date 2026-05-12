import numpy as np

from forestseg.export_pkg.metrics import binary_metrics


def test_binary_metrics_returns_expected_values():
    y_true = np.array([1, 1, 0, 0], dtype=np.uint8)
    y_prob = np.array([0.9, 0.3, 0.7, 0.1], dtype=np.float32)

    metrics = binary_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["confusion_matrix"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert metrics["count"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["iou"] == 1 / 3
