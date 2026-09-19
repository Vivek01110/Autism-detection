import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from .model import KAN
from .evaluate import evaluate_and_report

# Attempt imports based on requirements
try:
    from src.data_preprocessing.dataset3_preprocessing import load_config, Dataset3Preprocessor
    from src.utils.metrics import compute_all_metrics
except ImportError:
    pass

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(config_device: str) -> torch.device:
    if config_device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(config_device)

def train_kan(config: Dict[str, Any], preprocessor) -> Dict[str, Any]:
    set_seed(42)
    
    device = get_device(config.get('device', 'auto'))
    print(f"Using device: {device}")
    
    model_config = config.get('kan', config.get('kan_model', {}))
    hidden_layers = model_config.get('hidden_layers', [64, 32])
    grid_size = model_config.get('grid_size', 5)
    spline_order = model_config.get('spline_order', 3)
    dropout = model_config.get('dropout', 0.1)
    batch_size = model_config.get('batch_size', 64)
    lr = model_config.get('learning_rate', 1e-3)
    weight_decay = model_config.get('weight_decay', 1e-4)
    epochs = model_config.get('epochs', 100)
    patience = model_config.get('early_stopping_patience', model_config.get('patience', 15))

    train_loader = preprocessor.get_dataloader('train', batch_size=batch_size, shuffle=True)
    val_loader = preprocessor.get_dataloader('val', batch_size=batch_size, shuffle=False)
    test_loader = preprocessor.get_dataloader('test', batch_size=batch_size, shuffle=False)
    
    input_dim = preprocessor.input_dim
    
    model = KAN(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=1,
        grid_size=grid_size,
        spline_order=spline_order,
        dropout=dropout
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs('results/checkpoints', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/confusion_matrices', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    best_model_path = 'results/checkpoints/kan_best.pt'
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).float()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    # Plot training curves
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('KAN Training Curves')
    plt.savefig('results/plots/kan_training_curves.png')
    plt.close()
    
    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    metrics = evaluate_and_report(model, test_loader, device, label='Test')
    
    # Save metrics
    with open('results/metrics/kan_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Generate confusion matrix (using seaborn if available)
    try:
        import seaborn as sns
        plt.figure(figsize=(6, 5))
        sns.heatmap(metrics.get('confusion_matrix', [[0,0],[0,0]]), annot=True, fmt='d', cmap='Blues')
        plt.title('KAN Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('results/confusion_matrices/kan_confusion_matrix.png')
        plt.close()
    except ImportError:
        plt.figure()
        plt.imshow(metrics.get('confusion_matrix', [[0,0],[0,0]]), cmap='Blues')
        plt.colorbar()
        plt.title('KAN Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('results/confusion_matrices/kan_confusion_matrix.png')
        plt.close()
        
    return {'history': history, 'metrics': metrics}

if __name__ == '__main__':
    from src.data_preprocessing.dataset3_preprocessing import load_config, Dataset3Preprocessor
    
    config = load_config('configs/config.yaml')
    preprocessor = Dataset3Preprocessor(config)
    preprocessor.prepare_data()
    
    train_kan(config, preprocessor)
