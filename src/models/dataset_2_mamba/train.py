"""
Training script for Dataset 2 Mamba Primary Model.
"""

import os
import sys
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.data_preprocessing.dataset3_preprocessing import load_config
from src.data_preprocessing.dataset2_preprocessing import Dataset2Preprocessor
from src.utils.metrics import compute_all_metrics, plot_confusion_matrix
from .model import MambaClassifier
from .evaluate import evaluate_and_report


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config_device: str = "auto") -> torch.device:
    """Detect and return torch device based on config and system availability."""
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config_device)


def train_mamba(
    config: Dict[str, Any], preprocessor: Optional[Dataset2Preprocessor] = None
) -> Dict[str, Any]:
    """
    Train Mamba Primary Model on Dataset 2 eye-tracking sequences.
    """
    set_seed(config.get("random_seed", 42))

    device = get_device(config.get("device", "auto"))
    print(f"[Mamba Training] Using compute device: {device}")

    mamba_cfg = config.get("mamba", {})
    d_model = mamba_cfg.get("d_model", 64)
    d_state = mamba_cfg.get("d_state", 16)
    d_conv = mamba_cfg.get("d_conv", 4)
    expand = mamba_cfg.get("expand", 2)
    n_layers = mamba_cfg.get("n_layers", 2)
    dropout = mamba_cfg.get("dropout", 0.1)
    lr = mamba_cfg.get("learning_rate", 5e-4)
    weight_decay = mamba_cfg.get("weight_decay", 1e-4)
    epochs = mamba_cfg.get("epochs", 25)
    batch_size = mamba_cfg.get("batch_size", 64)
    patience = mamba_cfg.get("early_stopping_patience", 8)

    # Initialize / load preprocessor if not provided
    if preprocessor is None:
        preprocessor = Dataset2Preprocessor(config)
        preprocessor.run()

    train_loader = preprocessor.get_dataloader("train", batch_size=batch_size, shuffle=True)
    val_loader = preprocessor.get_dataloader("val", batch_size=batch_size, shuffle=False)
    test_loader = preprocessor.get_dataloader("test", batch_size=batch_size, shuffle=False)

    num_features = len(preprocessor.FEATURE_NAMES)

    # Initialize Mamba Classifier
    model = MambaClassifier(
        input_dim=num_features,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    print(
        f"[Mamba Architecture] Layers={n_layers}, d_model={d_model}, d_state={d_state}, "
        f"d_conv={d_conv}, expand={expand}, dropout={dropout}"
    )

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Setup artifact paths
    paths = config.get("paths", {})
    checkpoint_dir = paths.get("checkpoints", "results/checkpoints")
    plots_dir = paths.get("plots", "results/plots")
    cm_dir = paths.get("confusion_matrices", "results/confusion_matrices")
    metrics_dir = paths.get("metrics", "results/metrics")

    for d in [checkpoint_dir, plots_dir, cm_dir, metrics_dir]:
        os.makedirs(d, exist_ok=True)

    best_checkpoint_path = os.path.join(checkpoint_dir, "mamba_best.pt")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    print(f"[Mamba Training] Starting {epochs} epochs (batch_size={batch_size}, lr={lr})...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        # ---- Training Phase ----
        model.train()
        running_train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            x_b = batch[0].to(device)
            y_b = batch[1].to(device)

            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * len(y_b)
            train_samples += len(y_b)

        epoch_train_loss = running_train_loss / train_samples

        # ---- Validation Phase ----
        model.eval()
        running_val_loss = 0.0
        val_samples = 0
        val_correct = 0

        with torch.no_grad():
            for batch in val_loader:
                x_b = batch[0].to(device)
                y_b = batch[1].to(device)

                preds = model(x_b)
                loss = criterion(preds, y_b)

                running_val_loss += loss.item() * len(y_b)
                val_samples += len(y_b)
                binary_preds = (preds >= 0.5).float()
                val_correct += (binary_preds == y_b).sum().item()

        epoch_val_loss = running_val_loss / val_samples
        epoch_val_acc = val_correct / val_samples

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)

        epoch_dur = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_dur:.1f}s] - "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

        # ---- Checkpoint & Early Stopping ----
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "val_accuracy": epoch_val_acc,
                    "config": mamba_cfg,
                },
                best_checkpoint_path,
            )
            print(f"  --> Saved new best Mamba checkpoint (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Mamba Early Stopping] Triggered after {epoch} epochs (patience={patience}).")
                break

    total_training_time = time.time() - start_time
    print(f"[Mamba Training] Completed in {total_training_time:.2f} seconds.")

    # ---- Final Evaluation on Test Set using Best Model ----
    print(f"\n[Mamba Evaluation] Loading best model from {best_checkpoint_path} for testing...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_and_report(model, test_loader, device, label="Test Set")

    # ---- Save Plots and Metrics ----
    # 1. Training Curves Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    plt.title("Mamba — Loss Curves", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["val_accuracy"], label="Val Accuracy", color="#2ca02c", linewidth=2)
    plt.title("Mamba — Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(plots_dir, "mamba_training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Saved Mamba training curve to {curve_path}")

    # 2. Confusion Matrix Plot
    cm_path = os.path.join(cm_dir, "mamba_confusion_matrix.png")
    plot_confusion_matrix(
        np.array(test_metrics["confusion_matrix"]),
        class_names=["TD", "ASD"],
        title="Mamba — Confusion Matrix (Dataset 2 Test Set)",
        save_path=cm_path,
    )
    print(f"Saved Mamba confusion matrix to {cm_path}")

    # 3. Save JSON Metrics
    metrics_path = os.path.join(metrics_dir, "mamba_metrics.json")
    save_payload = {
        "model": "Mamba",
        "dataset": "Dataset 2 (Eye Tracking)",
        "training_time_seconds": round(total_training_time, 2),
        "best_epoch": checkpoint.get("epoch", None),
        "test_metrics": test_metrics,
        "history": history,
    }
    with open(metrics_path, "w") as f:
        json.dump(save_payload, f, indent=2)
    print(f"Saved Mamba metrics to {metrics_path}")

    return {
        "model": model,
        "metrics": test_metrics,
        "history": history,
        "checkpoint_path": best_checkpoint_path,
        "training_time": total_training_time,
    }


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    train_mamba(cfg)
