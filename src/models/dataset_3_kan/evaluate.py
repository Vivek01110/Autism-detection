import torch
import numpy as np
from typing import Tuple, Dict, Any

try:
    from src.utils.metrics import compute_all_metrics, print_classification_report
except ImportError:
    pass

def evaluate_kan(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.cpu().numpy()
            
            outputs = model(batch_X)
            probs = outputs.cpu().numpy()
            
            all_targets.extend(batch_y)
            all_probs.extend(probs)
            
    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    
    return y_true, y_pred, y_prob

def evaluate_and_report(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, label: str = 'Test') -> Dict[str, Any]:
    y_true, y_pred, y_prob = evaluate_kan(model, dataloader, device)
    
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    
    print(f"\nClassification Report for {label}:")
    print(print_classification_report(y_true, y_pred))
    
    return metrics
