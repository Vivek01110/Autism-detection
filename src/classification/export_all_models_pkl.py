"""
Export and Store All Models as .pkl Files
========================================
This script exports, serializes, and verifies all trained models for:
- Dataset 3 (Screening / Tabular): KAN, TabNet, XGBoost, Random Forest, LightGBM, SVM, Scaler
- Dataset 2 (Eye-Tracking): PatchTST, Mamba, XGBoost, Random Forest, SVM, Scalers
- Cross-Modal Decision Fusion: Tuned multi-modal ensemble policy

All serialized models are saved as standard Python pickle (.pkl) files in:
  results/models_pkl/
and mirrored into:
  results/checkpoints/
"""

import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Ensure project root in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_preprocessing.dataset3_preprocessing import Dataset3Preprocessor
from src.models.dataset_3_kan.inference import load_kan_model
from src.models.dataset_3_tabnet.inference import load_tabnet_model
from src.models.dataset_2_mamba.inference import load_mamba_model
from src.models.dataset_2_patchtst.inference import load_patchtst_model


def export_all_models(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")
    export_dir = os.path.join(ROOT_DIR, "results", "models_pkl")
    ckpt_dir = os.path.join(ROOT_DIR, "results", "checkpoints")
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    exported_records = []

    print("=" * 65)
    print("STEP 1: EXPORTING DATASET 3 (SCREENING) MODELS TO .PKL")
    print("=" * 65)

    # 1. KAN (Dataset 3)
    kan_ckpt = os.path.join(ckpt_dir, "kan_best.pt")
    if os.path.isfile(kan_ckpt):
        kan_model = load_kan_model(kan_ckpt, input_dim=16, config=config)
        kan_path = os.path.join(export_dir, "kan_screening.pkl")
        with open(kan_path, "wb") as f:
            pickle.dump(kan_model, f)
        with open(os.path.join(ckpt_dir, "kan_screening.pkl"), "wb") as f:
            pickle.dump(kan_model, f)
        exported_records.append({"Dataset": "Dataset 3", "Model": "KAN", "File": "kan_screening.pkl", "Size (bytes)": os.path.getsize(kan_path)})
        print(f"[OK] Saved kan_screening.pkl ({os.path.getsize(kan_path):,} bytes)")

    # 2. TabNet (Dataset 3)
    tabnet_ckpt = os.path.join(ckpt_dir, "tabnet_best.pt")
    if os.path.isfile(tabnet_ckpt):
        tabnet_model = load_tabnet_model(tabnet_ckpt, input_dim=16, config=config)
        tabnet_path = os.path.join(export_dir, "tabnet_screening.pkl")
        with open(tabnet_path, "wb") as f:
            pickle.dump(tabnet_model, f)
        with open(os.path.join(ckpt_dir, "tabnet_screening.pkl"), "wb") as f:
            pickle.dump(tabnet_model, f)
        exported_records.append({"Dataset": "Dataset 3", "Model": "TabNet", "File": "tabnet_screening.pkl", "Size (bytes)": os.path.getsize(tabnet_path)})
        print(f"[OK] Saved tabnet_screening.pkl ({os.path.getsize(tabnet_path):,} bytes)")

    # 3. Train & Export Dataset 3 ML models (XGBoost, Random Forest, LightGBM, SVM, Scaler)
    try:
        preprocessor = Dataset3Preprocessor(config)
        preprocessor.run()
        X_train, y_train = preprocessor.get_split("train")
        scaler_d3 = preprocessor.scaler

        # Scaler
        scaler_path = os.path.join(export_dir, "dataset3_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler_d3, f)
        with open(os.path.join(ckpt_dir, "dataset3_scaler.pkl"), "wb") as f:
            pickle.dump(scaler_d3, f)
        exported_records.append({"Dataset": "Dataset 3", "Model": "StandardScaler", "File": "dataset3_scaler.pkl", "Size (bytes)": os.path.getsize(scaler_path)})
        print(f"[OK] Saved dataset3_scaler.pkl ({os.path.getsize(scaler_path):,} bytes)")

        # XGBoost
        xgb_d3 = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
        xgb_d3.fit(X_train, y_train)
        xgb_d3_path = os.path.join(export_dir, "xgboost_screening.pkl")
        with open(xgb_d3_path, "wb") as f:
            pickle.dump(xgb_d3, f)
        with open(os.path.join(ckpt_dir, "xgboost_screening.pkl"), "wb") as f:
            pickle.dump(xgb_d3, f)
        exported_records.append({"Dataset": "Dataset 3", "Model": "XGBoost", "File": "xgboost_screening.pkl", "Size (bytes)": os.path.getsize(xgb_d3_path)})
        print(f"[OK] Saved xgboost_screening.pkl ({os.path.getsize(xgb_d3_path):,} bytes)")

        # Random Forest
        rf_d3 = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=3, random_state=42)
        rf_d3.fit(X_train, y_train)
        rf_d3_path = os.path.join(export_dir, "random_forest_screening.pkl")
        with open(rf_d3_path, "wb") as f:
            pickle.dump(rf_d3, f)
        with open(os.path.join(ckpt_dir, "random_forest_screening.pkl"), "wb") as f:
            pickle.dump(rf_d3, f)
        exported_records.append({"Dataset": "Dataset 3", "Model": "Random Forest", "File": "random_forest_screening.pkl", "Size (bytes)": os.path.getsize(rf_d3_path)})
        print(f"[OK] Saved random_forest_screening.pkl ({os.path.getsize(rf_d3_path):,} bytes)")

        # LightGBM
        lgb_d3 = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)
        lgb_d3.fit(X_train, y_train)
        lgb_d3_path = os.path.join(export_dir, "lightgbm_screening.pkl")
        with open(lgb_d3_path, "wb") as f:
            pickle.dump(lgb_d3, f)
        with open(os.path.join(ckpt_dir, "lightgbm_screening.pkl"), "wb") as f:
            pickle.dump(lgb_d3, f)
        exported_records.append({"Dataset": "Dataset 3", "Model": "LightGBM", "File": "lightgbm_screening.pkl", "Size (bytes)": os.path.getsize(lgb_d3_path)})
        print(f"[OK] Saved lightgbm_screening.pkl ({os.path.getsize(lgb_d3_path):,} bytes)")

        # SVM
        svm_d3 = SVC(C=5.0, kernel="rbf", probability=True, random_state=42)
        svm_d3.fit(X_train, y_train)
        svm_d3_path = os.path.join(export_dir, "svm_screening.pkl")
        with open(svm_d3_path, "wb") as f:
            pickle.dump(svm_d3, f)
        with open(os.path.join(ckpt_dir, "svm_screening.pkl"), "wb") as f:
            pickle.dump(svm_d3, f)
        exported_records.append({"Dataset": "Dataset 3", "Model": "SVM (RBF)", "File": "svm_screening.pkl", "Size (bytes)": os.path.getsize(svm_d3_path)})
        print(f"[OK] Saved svm_screening.pkl ({os.path.getsize(svm_d3_path):,} bytes)")
    except Exception as e:
        print(f"[Warning] Could not export D3 ML models: {e}")

    print("\n" + "=" * 65)
    print("STEP 2: EXPORTING DATASET 2 (EYE-TRACKING) MODELS TO .PKL")
    print("=" * 65)

    # 4. Mamba (Dataset 2)
    mamba_ckpt = os.path.join(ckpt_dir, "mamba_best.pt")
    if os.path.isfile(mamba_ckpt):
        mamba_model = load_mamba_model(mamba_ckpt, config=config, device=device, input_dim=8)
        mamba_path = os.path.join(export_dir, "mamba_eyetracking.pkl")
        with open(mamba_path, "wb") as f:
            pickle.dump(mamba_model, f)
        with open(os.path.join(ckpt_dir, "mamba_eyetracking.pkl"), "wb") as f:
            pickle.dump(mamba_model, f)
        exported_records.append({"Dataset": "Dataset 2", "Model": "Mamba", "File": "mamba_eyetracking.pkl", "Size (bytes)": os.path.getsize(mamba_path)})
        print(f"[OK] Saved mamba_eyetracking.pkl ({os.path.getsize(mamba_path):,} bytes)")

    # 5. PatchTST (Dataset 2)
    patchtst_ckpt = os.path.join(ckpt_dir, "patchtst_best.pt")
    if os.path.isfile(patchtst_ckpt):
        patchtst_model = load_patchtst_model(patchtst_ckpt, config=config, device=device, input_dim=8, seq_len=200)
        patchtst_path = os.path.join(export_dir, "patchtst_eyetracking.pkl")
        with open(patchtst_path, "wb") as f:
            pickle.dump(patchtst_model, f)
        with open(os.path.join(ckpt_dir, "patchtst_eyetracking.pkl"), "wb") as f:
            pickle.dump(patchtst_model, f)
        exported_records.append({"Dataset": "Dataset 2", "Model": "PatchTST", "File": "patchtst_eyetracking.pkl", "Size (bytes)": os.path.getsize(patchtst_path)})
        print(f"[OK] Saved patchtst_eyetracking.pkl ({os.path.getsize(patchtst_path):,} bytes)")

    # 6. Train & Export Dataset 2 Aggregated ML models (XGBoost, Random Forest, SVM)
    d2_feat_path = os.path.join(ROOT_DIR, "data", "processed", "dataset_2", "dataset2_features.csv")
    if os.path.isfile(d2_feat_path):
        features_df = pd.read_csv(d2_feat_path)
        feat_cols = [c for c in features_df.columns if c not in ["ParticipantID", "label"]]
        X_agg = features_df[feat_cols].values.astype(np.float32)
        y_agg = features_df["label"].values.astype(np.int32)

        # Impute NaNs if any
        X_agg = np.nan_to_num(X_agg, nan=0.0, posinf=0.0, neginf=0.0)

        scaler_agg = StandardScaler()
        X_agg_scaled = scaler_agg.fit_transform(X_agg)

        # Scaler Agg
        scaler_agg_path = os.path.join(export_dir, "dataset2_scaler_agg.pkl")
        with open(scaler_agg_path, "wb") as f:
            pickle.dump(scaler_agg, f)
        with open(os.path.join(ckpt_dir, "dataset2_scaler_agg.pkl"), "wb") as f:
            pickle.dump(scaler_agg, f)
        exported_records.append({"Dataset": "Dataset 2", "Model": "StandardScaler (Agg)", "File": "dataset2_scaler_agg.pkl", "Size (bytes)": os.path.getsize(scaler_agg_path)})
        print(f"[OK] Saved dataset2_scaler_agg.pkl ({os.path.getsize(scaler_agg_path):,} bytes)")

        # XGBoost D2
        xgb_d2 = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss")
        xgb_d2.fit(X_agg_scaled, y_agg)
        xgb_d2_path = os.path.join(export_dir, "xgboost_eyetracking.pkl")
        with open(xgb_d2_path, "wb") as f:
            pickle.dump(xgb_d2, f)
        with open(os.path.join(ckpt_dir, "xgboost_eyetracking.pkl"), "wb") as f:
            pickle.dump(xgb_d2, f)
        exported_records.append({"Dataset": "Dataset 2", "Model": "XGBoost", "File": "xgboost_eyetracking.pkl", "Size (bytes)": os.path.getsize(xgb_d2_path)})
        print(f"[OK] Saved xgboost_eyetracking.pkl ({os.path.getsize(xgb_d2_path):,} bytes)")

        # Random Forest D2
        rf_d2 = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=3, random_state=42)
        rf_d2.fit(X_agg_scaled, y_agg)
        rf_d2_path = os.path.join(export_dir, "random_forest_eyetracking.pkl")
        with open(rf_d2_path, "wb") as f:
            pickle.dump(rf_d2, f)
        with open(os.path.join(ckpt_dir, "random_forest_eyetracking.pkl"), "wb") as f:
            pickle.dump(rf_d2, f)
        exported_records.append({"Dataset": "Dataset 2", "Model": "Random Forest", "File": "random_forest_eyetracking.pkl", "Size (bytes)": os.path.getsize(rf_d2_path)})
        print(f"[OK] Saved random_forest_eyetracking.pkl ({os.path.getsize(rf_d2_path):,} bytes)")

        # SVM D2
        svm_d2 = SVC(C=10.0, kernel="rbf", probability=True, random_state=42)
        svm_d2.fit(X_agg_scaled, y_agg)
        svm_d2_path = os.path.join(export_dir, "svm_eyetracking.pkl")
        with open(svm_d2_path, "wb") as f:
            pickle.dump(svm_d2, f)
        with open(os.path.join(ckpt_dir, "svm_eyetracking.pkl"), "wb") as f:
            pickle.dump(svm_d2, f)
        exported_records.append({"Dataset": "Dataset 2", "Model": "SVM (RBF)", "File": "svm_eyetracking.pkl", "Size (bytes)": os.path.getsize(svm_d2_path)})
        print(f"[OK] Saved svm_eyetracking.pkl ({os.path.getsize(svm_d2_path):,} bytes)")

    # 7. Sequence Scaler
    d2_proc_path = os.path.join(ROOT_DIR, "data", "processed", "dataset_2", "dataset2_processed.pt")
    if os.path.isfile(d2_proc_path):
        data = torch.load(d2_proc_path, map_location="cpu", weights_only=False)
        train_seq = data["train_sequences"]
        scaler_seq = StandardScaler()
        scaler_seq.fit(train_seq.reshape(-1, train_seq.shape[-1]))
        scaler_seq_path = os.path.join(export_dir, "dataset2_scaler_seq.pkl")
        with open(scaler_seq_path, "wb") as f:
            pickle.dump(scaler_seq, f)
        with open(os.path.join(ckpt_dir, "dataset2_scaler_seq.pkl"), "wb") as f:
            pickle.dump(scaler_seq, f)
        exported_records.append({"Dataset": "Dataset 2", "Model": "StandardScaler (Seq)", "File": "dataset2_scaler_seq.pkl", "Size (bytes)": os.path.getsize(scaler_seq_path)})
        print(f"[OK] Saved dataset2_scaler_seq.pkl ({os.path.getsize(scaler_seq_path):,} bytes)")

    print("\n" + "=" * 65)
    print("STEP 3: EXPORTING CROSS-MODAL DECISION FUSION MODEL TO .PKL")
    print("=" * 65)

    decision_fusion_dict = {
        "description": "Weighted decision-level cross-modal fusion model (Eye-Tracking + Q-CHAT-10 Screening)",
        "policies": {
            "kan_mamba": {"alpha": 0.9, "threshold": 0.23, "description": "0.9 * KAN(D3) + 0.1 * Mamba(D2)"},
            "kan_patchtst": {"alpha": 1.0, "threshold": 0.24, "description": "1.0 * KAN(D3) + 0.0 * PatchTST(D2)"},
            "tabnet_mamba": {"alpha": 0.9, "threshold": 0.27, "description": "0.9 * TabNet(D3) + 0.1 * Mamba(D2)"},
            "tabnet_patchtst": {"alpha": 1.0, "threshold": 0.30, "description": "1.0 * TabNet(D3) + 0.0 * PatchTST(D2)"},
        },
        "default_pair": "kan_mamba"
    }
    fusion_path = os.path.join(export_dir, "decision_fusion_model.pkl")
    with open(fusion_path, "wb") as f:
        pickle.dump(decision_fusion_dict, f)
    with open(os.path.join(ckpt_dir, "decision_fusion_model.pkl"), "wb") as f:
        pickle.dump(decision_fusion_dict, f)
    exported_records.append({"Dataset": "Fusion", "Model": "Decision Fusion Policy", "File": "decision_fusion_model.pkl", "Size (bytes)": os.path.getsize(fusion_path)})
    print(f"[OK] Saved decision_fusion_model.pkl ({os.path.getsize(fusion_path):,} bytes)")

    # Verification Step: Load all .pkl files back and verify
    print("\n" + "=" * 65)
    print("STEP 4: VERIFYING ALL .PKL ARTIFACTS BY RELOADING")
    print("=" * 65)
    verified_count = 0
    for rec in exported_records:
        file_path = os.path.join(export_dir, rec["File"])
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        assert obj is not None, f"Failed to load {rec['File']}"
        rec["Status"] = "[PASS] Verified"
        verified_count += 1
        print(f"  [PASS] {rec['File']:<30} successfully reloaded.")

    print(f"\nAll {verified_count} model .pkl files verified successfully!")

    summary_df = pd.DataFrame(exported_records)
    print("\n" + "=" * 65)
    print("SUMMARY OF STORED .PKL MODEL ARTIFACTS")
    print("=" * 65)
    print(summary_df.to_string(index=False))

    summary_csv = os.path.join(ROOT_DIR, "results", "metrics", "exported_models_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary table written to: {summary_csv}")

    return summary_df


if __name__ == "__main__":
    export_all_models()
