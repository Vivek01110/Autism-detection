import torch
import numpy as np
from typing import Dict, Any

from .model import TabNet

def load_tabnet_model(checkpoint_path: str, input_dim: int, config: Dict[str, Any]) -> torch.nn.Module:
    tabnet_cfg = config.get('tabnet', {})
    device_name = config.get('device', 'auto')
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device_name)
    
    model = TabNet(
        input_dim=input_dim,
        n_d=tabnet_cfg.get('n_d', 16),
        n_a=tabnet_cfg.get('n_a', 16),
        n_steps=tabnet_cfg.get('n_steps', 3),
        gamma=tabnet_cfg.get('gamma', 1.3),
        lambda_sparse=tabnet_cfg.get('lambda_sparse', 0.001),
        momentum=tabnet_cfg.get('momentum', 0.02),
        virtual_batch_size=tabnet_cfg.get('virtual_batch_size', 128),
        mask_type=tabnet_cfg.get('mask_type', 'sparsemax')
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def predict_asd_probability(model: torch.nn.Module, features: np.ndarray, scaler: Any, device: torch.device) -> float:
    model.eval()
    feat = np.asarray(features, dtype=np.float32)
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)
    if scaler is not None and feat.shape[1] >= 4:
        # Scale numerical features (first 4) if unscaled
        feat_scaled = feat.copy()
        feat_scaled[:, :4] = scaler.transform(feat[:, :4])
        feat = feat_scaled
        
    X_tensor = torch.FloatTensor(feat).to(device)
    with torch.no_grad():
        prob, _ = model(X_tensor)
        
    return float(prob.reshape(-1)[0].item())

def predict_batch(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        probs, _ = model(X_tensor)
        
    return np.asarray(probs.cpu().numpy()).reshape(-1)
