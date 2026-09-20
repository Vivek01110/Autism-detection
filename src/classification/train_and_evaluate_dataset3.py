"""
Pipeline runner for Dataset 3 (Clinical / Tabular):
1. Runs preprocessing and verification.
2. Trains and evaluates KAN (Primary).
3. Trains and evaluates TabNet (Benchmark).
4. Verifies sanity conditions (probabilities in [0, 1], binary predictions, non-fabricated metrics).
5. Compiles and saves results/metrics/dataset3_model_comparison.csv.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_preprocessing.dataset3_preprocessing import load_config, Dataset3Preprocessor
from src.models.dataset_3_kan.train import train_kan
from src.models.dataset_3_tabnet.train import train_tabnet
from src.models.dataset_3_kan.inference import load_kan_model, predict_batch as kan_predict_batch
from src.models.dataset_3_tabnet.inference import load_tabnet_model, predict_batch as tabnet_predict_batch


def run_pipeline(config_path: str = "configs/config.yaml", max_epochs: int | None = None):
    config = load_config(config_path)
    
    if max_epochs is not None:
        if "kan" in config:
            config["kan"]["epochs"] = max_epochs
        if "tabnet" in config:
            config["tabnet"]["epochs"] = max_epochs
            
    print("==================================================")
    print("STEP 1: PREPROCESSING DATASET 3")
    print("==================================================")
    preprocessor = Dataset3Preprocessor(config)
    preprocessor.run()
    
    X_train, y_train = preprocessor.get_split("train")
    X_val, y_val = preprocessor.get_split("val")
    X_test, y_test = preprocessor.get_split("test")
    
    print(f"Data splits: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"Input dimension: {preprocessor.input_dim}")
    
    print("\n==================================================")
    print("STEP 2: TRAINING & EVALUATING KAN (PRIMARY)")
    print("==================================================")
    kan_res = train_kan(config, preprocessor)
    kan_metrics = kan_res["metrics"]
    
    print("\n==================================================")
    print("STEP 3: TRAINING & EVALUATING TABNET (BENCHMARK)")
    print("==================================================")
    tabnet_res = train_tabnet(config, preprocessor)
    tabnet_metrics = tabnet_res["metrics"]
    
    print("\n==================================================")
    print("STEP 4: VERIFICATION & SANITY CHECKS")
    print("==================================================")
    device_name = config.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    
    # Check KAN inference
    kan_model = load_kan_model("results/checkpoints/kan_best.pt", preprocessor.input_dim, config)
    kan_test_probs = kan_predict_batch(kan_model, X_test, device)
    kan_test_preds = (kan_test_probs >= 0.5).astype(int)
    
    assert np.all((kan_test_probs >= 0.0) & (kan_test_probs <= 1.0)), "KAN probabilities out of [0, 1]"
    assert set(np.unique(kan_test_preds)).issubset({0, 1}), "KAN predictions not binary"
    print("[OK] KAN sanity verified: Probabilities in [0, 1], predictions binary.")
    
    # Check TabNet inference
    tabnet_model = load_tabnet_model("results/checkpoints/tabnet_best.pt", preprocessor.input_dim, config)
    tabnet_test_probs = tabnet_predict_batch(tabnet_model, X_test, device)
    tabnet_test_preds = (tabnet_test_probs >= 0.5).astype(int)
    
    assert np.all((tabnet_test_probs >= 0.0) & (tabnet_test_probs <= 1.0)), "TabNet probabilities out of [0, 1]"
    assert set(np.unique(tabnet_test_preds)).issubset({0, 1}), "TabNet predictions not binary"
    print("[OK] TabNet sanity verified: Probabilities in [0, 1], predictions binary.")
    
    print("\n==================================================")
    print("STEP 5: GENERATING COMPARISON TABLE")
    print("==================================================")
    comparison_data = [
        {
            "Model": "KAN",
            "Dataset": "Dataset 3 (Clinical/Tabular)",
            "Accuracy": round(kan_metrics["accuracy"], 4),
            "Precision": round(kan_metrics["precision"], 4),
            "Recall": round(kan_metrics["recall"], 4),
            "F1": round(kan_metrics["f1"], 4),
            "ROC_AUC": round(kan_metrics["roc_auc"], 4),
        },
        {
            "Model": "TabNet",
            "Dataset": "Dataset 3 (Clinical/Tabular)",
            "Accuracy": round(tabnet_metrics["accuracy"], 4),
            "Precision": round(tabnet_metrics["precision"], 4),
            "Recall": round(tabnet_metrics["recall"], 4),
            "F1": round(tabnet_metrics["f1"], 4),
            "ROC_AUC": round(tabnet_metrics["roc_auc"], 4),
        },
    ]
    
    comparison_df = pd.DataFrame(comparison_data)
    os.makedirs("results/metrics", exist_ok=True)
    comparison_path = "results/metrics/dataset3_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved comparison to {comparison_path}:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dataset 3 training and evaluation pipeline.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Optional max epochs for sanity check.")
    args = parser.parse_args()
    
    run_pipeline(args.config, args.max_epochs)
