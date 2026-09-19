"""Tests for the shared metrics module."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.metrics import (
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_roc_auc,
    compute_confusion_matrix,
    compute_all_metrics,
    print_classification_report,
)


class TestIndividualMetrics:
    """Test each metric function independently."""

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert compute_accuracy(y_true, y_pred) == 1.0
        assert compute_precision(y_true, y_pred) == 1.0
        assert compute_recall(y_true, y_pred) == 1.0
        assert compute_f1(y_true, y_pred) == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        assert compute_accuracy(y_true, y_pred) == 0.0
        assert compute_precision(y_true, y_pred) == 0.0
        assert compute_recall(y_true, y_pred) == 0.0

    def test_roc_auc_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_roc_auc(y_true, y_prob) == 1.0

    def test_roc_auc_random(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        auc = compute_roc_auc(y_true, y_prob)
        assert 0.4 <= auc <= 0.6  # ~0.5 for random

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
        assert cm.sum() == 5


class TestComputeAllMetrics:
    """Test the combined metrics function."""

    def test_returns_all_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.2, 0.8, 0.6, 0.9])
        result = compute_all_metrics(y_true, y_pred, y_prob)

        expected_keys = {
            "accuracy", "precision", "recall", "f1", "roc_auc",
            "confusion_matrix", "tn", "fp", "fn", "tp",
        }
        assert expected_keys == set(result.keys())

    def test_metrics_are_floats(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.2, 0.8, 0.6, 0.9])
        result = compute_all_metrics(y_true, y_pred, y_prob)

        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert isinstance(result[key], float), f"{key} is not float"
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"

    def test_confusion_matrix_sums(self):
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.9, 0.7, 0.3, 0.2])
        result = compute_all_metrics(y_true, y_pred, y_prob)
        assert result["tn"] + result["fp"] + result["fn"] + result["tp"] == 5


class TestClassificationReport:
    """Test report generation."""

    def test_report_is_string(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        report = print_classification_report(y_true, y_pred)
        assert isinstance(report, str)
        assert "ASD" in report
        assert "Non-ASD" in report
