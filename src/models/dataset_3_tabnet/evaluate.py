import torch
import numpy as np
from typing import Tuple, Dict, Any

from src.utils.metrics import compute_all_metrics, print_classification_report

def evaluate_tabnet(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_prob = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs, _ = model(X_batch)
            outputs = outputs.view(-1)
            
            y_true.extend(y_batch.cpu().numpy().reshape(-1))
            y_prob.extend(outputs.cpu().numpy().reshape(-1))
            
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    
    return y_true, y_pred, y_prob

def evaluate_and_report(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, label: str = 'Test') -> Dict[str, Any]:
    y_true, y_pred, y_prob = evaluate_tabnet(model, dataloader, device)
    
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    print(f"\n{label} Classification Report:")
    print(print_classification_report(y_true, y_pred))
    
    return metrics
