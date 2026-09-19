"""
Reusable evaluation metrics for ASD classification models.
All models must use these functions to ensure consistent evaluation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute precision (positive = ASD)."""
    return float(precision_score(y_true, y_pred, zero_division=0))


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute recall / sensitivity (positive = ASD)."""
    return float(recall_score(y_true, y_pred, zero_division=0))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1-score."""
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute ROC-AUC from predicted probabilities."""
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        # Happens if only one class is present in y_true
        return 0.0


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix. Rows=actual, Cols=predicted."""
    return confusion_matrix(y_true, y_pred)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Compute all classification metrics.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_pred: Predicted binary labels (0/1).
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dictionary with all metric values.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_prob = np.asarray(y_prob).ravel()

    cm = compute_confusion_matrix(y_true, y_pred)

    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "precision": compute_precision(y_true, y_pred),
        "recall": compute_recall(y_true, y_pred),
        "f1": compute_f1(y_true, y_pred),
        "roc_auc": compute_roc_auc(y_true, y_prob),
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Return a formatted classification report string."""
    return classification_report(
        y_true,
        y_pred,
        target_names=["Non-ASD (0)", "ASD (1)"],
        digits=4,
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> None:
    """Plot and optionally save a confusion matrix."""
    import os
    import matplotlib.pyplot as plt

    if class_names is None:
        class_names = ["TD", "ASD"]

    cm = np.asarray(cm)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{int(cm[i, j])}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
