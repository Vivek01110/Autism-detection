import torch
import numpy as np
from typing import Dict, Any, Union

from .model import KAN

def load_kan_model(checkpoint_path: str, input_dim: int, config: Dict[str, Any]) -> KAN:
    model_config = config.get('kan', config.get('kan_model', {}))
    hidden_layers = model_config.get('hidden_layers', [64, 32])
    grid_size = model_config.get('grid_size', 5)
    spline_order = model_config.get('spline_order', 3)
    dropout = model_config.get('dropout', 0.1)
    
    device_name = config.get('device', 'auto')
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device_name)
    
    model = KAN(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=1,
        grid_size=grid_size,
        spline_order=spline_order,
        dropout=dropout
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    return model

def predict_batch(model: KAN, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        probs = model(X_tensor)
        return np.asarray(probs.cpu().numpy()).reshape(-1)

def predict_asd_probability(model: KAN, features: Union[Dict[str, Any], np.ndarray], scaler, device: torch.device) -> float:
    model.eval()
    feat = np.asarray(features, dtype=np.float32)
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)
    if scaler is not None and feat.shape[1] >= 4:
        feat_scaled = feat.copy()
        feat_scaled[:, :4] = scaler.transform(feat[:, :4])
        feat = feat_scaled
            
    with torch.no_grad():
        X_tensor = torch.FloatTensor(feat).to(device)
        prob = model(X_tensor)
        
    return float(prob.reshape(-1)[0].item())
