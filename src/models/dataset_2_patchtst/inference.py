"""
Inference utilities for Dataset 2 PatchTST model.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Union, Optional

from .model import PatchTSTClassifier


def load_patchtst_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    input_dim: int = 8,
    seq_len: int = 200,
) -> PatchTSTClassifier:
    """
    Load a trained PatchTSTClassifier checkpoint.
    """
    if device is None:
        device_name = config.get("device", "auto")
        if device_name == "auto":
            device_name = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        device = torch.device(device_name)

    patch_cfg = config.get("patchtst", {})
    patch_len = patch_cfg.get("patch_len", 16)
    stride = patch_cfg.get("stride", 8)
    d_model = patch_cfg.get("d_model", 64)
    n_heads = patch_cfg.get("n_heads", 4)
    d_ff = patch_cfg.get("d_ff", 128)
    n_layers = patch_cfg.get("n_layers", 2)
    dropout = patch_cfg.get("dropout", 0.1)

    model = PatchTSTClassifier(
        input_dim=input_dim,
        seq_len=seq_len,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict_batch(
    model: PatchTSTClassifier, X: np.ndarray, device: torch.device, batch_size: int = 64
) -> np.ndarray:
    """
    Predict ASD probabilities for a batch of eye-tracking sequences.
    
    Args:
        model: Loaded PatchTSTClassifier
        X: NumPy array of shape (B, 200, 8)
        device: Torch device
        batch_size: Sub-batch chunk size for memory efficiency
    Returns:
        NumPy array of ASD probabilities of shape (B,) in [0, 1]
    """
    model.eval()
    all_probs = []
    with torch.no_grad():
        num_samples = len(X)
        for start_idx in range(0, num_samples, batch_size):
            chunk = X[start_idx : start_idx + batch_size]
            X_tensor = torch.as_tensor(chunk, dtype=torch.float32, device=device)
            probs = model(X_tensor)
            all_probs.extend(probs.detach().cpu().numpy().tolist())
    return np.array(all_probs, dtype=np.float32)


def predict_asd_probability(
    model: PatchTSTClassifier,
    sequence: Union[np.ndarray, torch.Tensor],
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    Predict ASD probability for a single eye-tracking sequence.
    
    Args:
        model: Loaded PatchTSTClassifier
        sequence: Array of shape (200, 8) or (1, 200, 8)
        scaler: Optional fitted StandardScaler
        device: Torch device
    Returns:
        ASD probability as a float in [0.0, 1.0]
    """
    if device is None:
        device = next(model.parameters()).device

    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim == 2:
        seq = np.expand_dims(seq, axis=0)  # (1, 200, 8)

    if scaler is not None:
        b, l, d = seq.shape
        flat = seq.reshape(-1, d)
        seq = scaler.transform(flat).reshape(b, l, d)

    probs = predict_batch(model, seq, device)
    return float(probs[0])
