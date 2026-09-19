"""
Evaluation utilities for Dataset 2 PatchTST model.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any

from src.utils.metrics import compute_all_metrics, print_classification_report


def evaluate_patchtst(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate PatchTST model on a DataLoader.
    
    Returns:
        y_true: Ground truth binary labels (N,)
        y_pred: Predicted binary labels {0, 1} (N,)
        y_prob: Predicted ASD probabilities in [0, 1] (N,)
    """
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                batch_x, batch_y, _ = batch
            else:
                batch_x, batch_y = batch

            batch_x = batch_x.to(device)
            probs = model(batch_x)

            all_targets.extend(batch_y.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets, dtype=np.float32)
    y_prob = np.array(all_probs, dtype=np.float32)
    y_pred = (y_prob >= 0.5).astype(int)

    return y_true, y_pred, y_prob


def evaluate_and_report(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    label: str = "Test",
) -> Dict[str, Any]:
    """
    Evaluate PatchTST model and print formatted classification report.
    """
    y_true, y_pred, y_prob = evaluate_patchtst(model, dataloader, device)
    metrics = compute_all_metrics(y_true, y_pred, y_prob)

    print(f"\n--- PatchTST Classification Report ({label}) ---")
    print(print_classification_report(y_true, y_pred))
    print(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

    return metrics
