import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from .model import TabNet
from .evaluate import evaluate_and_report
from src.data_preprocessing.dataset3_preprocessing import load_config, Dataset3Preprocessor
from src.utils.metrics import compute_all_metrics

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_training_curves(history, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(metrics, save_path):
    cm = np.array(metrics.get('confusion_matrix', [[0, 0], [0, 0]]))
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])
    
    thresh = np.max(cm) / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_tabnet(config: Dict[str, Any], preprocessor) -> Dict[str, Any]:
    set_seed(42)
    
    tabnet_cfg = config.get('tabnet', {})
    device_name = config.get('device', 'auto')
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device_name)
    
    batch_size = tabnet_cfg.get('batch_size', 256)
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders(
        batch_size=batch_size
    )
    
    input_dim = preprocessor.input_dim
    
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
    ).to(device)
    
    criterion = nn.BCELoss()
    lr = tabnet_cfg.get('learning_rate', 0.02)
    weight_decay = tabnet_cfg.get('weight_decay', 1e-4)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )
    
    epochs = tabnet_cfg.get('epochs', 100)
    patience = tabnet_cfg.get('early_stopping_patience', 15)
    
    os.makedirs('results/checkpoints', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/confusion_matrices', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()
            
            optimizer.zero_grad()
            outputs, sparsity_loss = model(X_batch)
            outputs = outputs.view(-1)
            y_batch = y_batch.view(-1)
            
            loss = criterion(outputs, y_batch) + sparsity_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()
                outputs, _ = model(X_batch)
                outputs = outputs.view(-1)
                y_batch = y_batch.view(-1)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'results/checkpoints/tabnet_best.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    plot_training_curves(history, 'results/plots/tabnet_training_curves.png')
    
    model.load_state_dict(torch.load('results/checkpoints/tabnet_best.pt'))
    
    metrics = evaluate_and_report(model, test_loader, device, label='Test')
    
    with open('results/metrics/tabnet_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    plot_confusion_matrix(metrics, 'results/confusion_matrices/tabnet_confusion_matrix.png')
    
    return {'history': history, 'metrics': metrics}

if __name__ == '__main__':
    config = load_config('configs/config.yaml')
    preprocessor = Dataset3Preprocessor(config)
    preprocessor.run()
    train_tabnet(config, preprocessor)
