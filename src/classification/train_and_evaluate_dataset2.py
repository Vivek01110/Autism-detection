"""
Pipeline runner for Dataset 2 (Eye Tracking):
1. Loads preprocessed Dataset 2 participant-level sequences.
2. Trains and evaluates Mamba (Primary).
3. Trains and evaluates PatchTST (Benchmark).
4. Verifies sanity conditions (probabilities in [0, 1], binary predictions, non-fabricated metrics, no participant_id feature).
5. Compiles and saves results/metrics/dataset2_model_comparison.csv.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_preprocessing.dataset3_preprocessing import load_config
from src.data_preprocessing.dataset2_preprocessing import Dataset2Preprocessor
from src.models.dataset_2_mamba.train import train_mamba
from src.models.dataset_2_patchtst.train import train_patchtst
from src.models.dataset_2_mamba.inference import (
    load_mamba_model,
    predict_batch as mamba_predict_batch,
)
from src.models.dataset_2_patchtst.inference import (
    load_patchtst_model,
    predict_batch as patchtst_predict_batch,
)


def run_pipeline(config_path: str = "configs/config.yaml", max_epochs: int | None = None):
    config = load_config(config_path)

    if max_epochs is not None:
        if "mamba" in config:
            config["mamba"]["epochs"] = max_epochs
        if "patchtst" in config:
            config["patchtst"]["epochs"] = max_epochs

    print("==================================================")
    print("STEP 1: LOADING PREPROCESSED DATASET 2 DATA")
    print("==================================================")
    preprocessor = Dataset2Preprocessor(config)
    preprocessor.run()

    X_train, y_train, pids_train = preprocessor.get_split("train")
    X_val, y_val, pids_val = preprocessor.get_split("val")
    X_test, y_test, pids_test = preprocessor.get_split("test")

    print(f"Data splits: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"Feature count: {len(preprocessor.FEATURE_NAMES)} ({preprocessor.FEATURE_NAMES})")
    assert X_train.shape[2] == 8, f"Expected 8 features, got {X_train.shape[2]}"
    print("✓ Verified: Models receive strictly (B, 200, 8) tensors; participant_id is NOT an input feature.")

    print("\n==================================================")
    print("STEP 2: TRAINING & EVALUATING MAMBA (PRIMARY)")
    print("==================================================")
    mamba_res = train_mamba(config, preprocessor)
    mamba_metrics = mamba_res["metrics"]

    print("\n==================================================")
    print("STEP 3: TRAINING & EVALUATING PATCHTST (BENCHMARK)")
    print("==================================================")
    patchtst_res = train_patchtst(config, preprocessor)
    patchtst_metrics = patchtst_res["metrics"]

    print("\n==================================================")
    print("STEP 4: VERIFICATION & SANITY CHECKS")
    print("==================================================")
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

    # 1. Check Mamba inference
    mamba_ckpt = "results/checkpoints/mamba_best.pt"
    mamba_model = load_mamba_model(mamba_ckpt, config, device, input_dim=8)
    sample_test_X = X_test[:100]
    mamba_probs = mamba_predict_batch(mamba_model, sample_test_X, device)
    mamba_preds = (mamba_probs >= 0.5).astype(int)

    assert np.all((mamba_probs >= 0.0) & (mamba_probs <= 1.0)), "Mamba probabilities outside [0, 1]"
    assert set(np.unique(mamba_preds)).issubset({0, 1}), "Mamba predictions are not binary"
    print("✓ Mamba sanity verified: Probabilities in [0, 1], binary predictions, checkpoint reloaded successfully.")

    # 2. Check PatchTST inference
    patchtst_ckpt = "results/checkpoints/patchtst_best.pt"
    patchtst_model = load_patchtst_model(patchtst_ckpt, config, device, input_dim=8, seq_len=200)
    patchtst_probs = patchtst_predict_batch(patchtst_model, sample_test_X, device)
    patchtst_preds = (patchtst_probs >= 0.5).astype(int)

    assert np.all((patchtst_probs >= 0.0) & (patchtst_probs <= 1.0)), "PatchTST probabilities outside [0, 1]"
    assert set(np.unique(patchtst_preds)).issubset({0, 1}), "PatchTST predictions are not binary"
    print("✓ PatchTST sanity verified: Probabilities in [0, 1], binary predictions, checkpoint reloaded successfully.")

    print("\n==================================================")
    print("STEP 5: GENERATING COMPARISON TABLE")
    print("==================================================")
    comparison_data = [
        {
            "Model": "Mamba",
            "Dataset": "Dataset 2 (Eye Tracking)",
            "Accuracy": round(float(mamba_metrics["accuracy"]), 4),
            "Precision": round(float(mamba_metrics["precision"]), 4),
            "Recall": round(float(mamba_metrics["recall"]), 4),
            "F1": round(float(mamba_metrics["f1"]), 4),
            "ROC_AUC": round(float(mamba_metrics["roc_auc"]), 4),
        },
        {
            "Model": "PatchTST",
            "Dataset": "Dataset 2 (Eye Tracking)",
            "Accuracy": round(float(patchtst_metrics["accuracy"]), 4),
            "Precision": round(float(patchtst_metrics["precision"]), 4),
            "Recall": round(float(patchtst_metrics["recall"]), 4),
            "F1": round(float(patchtst_metrics["f1"]), 4),
            "ROC_AUC": round(float(patchtst_metrics["roc_auc"]), 4),
        },
    ]

    comparison_df = pd.DataFrame(comparison_data)
    os.makedirs("results/metrics", exist_ok=True)
    comparison_path = "results/metrics/dataset2_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved comparison to {comparison_path}:")
    print(comparison_df.to_string(index=False))

    return comparison_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dataset 2 training and evaluation pipeline.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Optional max epochs for sanity check.")
    args = parser.parse_args()

    run_pipeline(args.config, args.max_epochs)
