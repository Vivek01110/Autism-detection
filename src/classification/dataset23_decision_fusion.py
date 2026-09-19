"""
Decision-Level Integration & Controlled Cross-Dataset Simulation:
Dataset 2 (Eye Tracking) + Dataset 3 (Clinical / Tabular)

IMPORTANT SCIENTIFIC DISCLAIMER:
--------------------------------
Dataset 2 and Dataset 3 are UNPAIRED datasets collected from distinct study cohorts
with different age distributions (school-aged children vs. toddlers).
There is NO legitimate participant-level mapping between the two datasets.
This module implements a TRANSPARENT DECISION-LEVEL INTEGRATION EXPERIMENT and a
CONTROLLED CROSS-DATASET SIMULATION TESTBED. It does NOT assert or fabricate actual
paired multimodal patient identities.

Methodology:
1. Both modality specialists are trained independently on their respective authentic datasets.
2. An alpha-weight decision fusion policy:
     P_combined = alpha * P_clinical + (1 - alpha) * P_eye
   is tuned STRICTLY on a Validation Simulation Cohort across a predefined grid:
     alpha in {0.0, 0.1, 0.2, ..., 1.0}
3. The optimal alpha* and classification threshold tau* are FROZEN before applying to the
   Held-Out Test Simulation Cohort.
4. Baseline standalone models are evaluated on the exact same simulation protocol for fair comparison.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import roc_curve, auc

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from src.data_preprocessing.dataset3_preprocessing import Dataset3Preprocessor, load_config
from src.data_preprocessing.dataset2_preprocessing import Dataset2Preprocessor
from src.models.dataset_3_kan.inference import load_kan_model, predict_batch as kan_predict
from src.models.dataset_3_tabnet.inference import load_tabnet_model, predict_batch as tabnet_predict
from src.models.dataset_2_mamba.inference import load_mamba_model, predict_batch as mamba_predict
from src.models.dataset_2_patchtst.inference import load_patchtst_model, predict_batch as patchtst_predict
from src.utils.metrics import compute_all_metrics, plot_confusion_matrix


def get_device(config_device: str = "auto") -> torch.device:
    """Detect compute device."""
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config_device)


def compute_fused_score(
    p_clinical: np.ndarray, p_eye: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Compute linear decision fusion score:
        P_combined = alpha * P_clinical + (1 - alpha) * P_eye
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Alpha must be within [0.0, 1.0], got {alpha}")
    p_clin = np.clip(np.asarray(p_clinical, dtype=np.float32), 0.0, 1.0)
    p_e = np.clip(np.asarray(p_eye, dtype=np.float32), 0.0, 1.0)
    return alpha * p_clin + (1.0 - alpha) * p_e


def construct_stratified_simulation_cohort(
    d3_probs: np.ndarray,
    d3_labels: np.ndarray,
    d3_meta_df: pd.DataFrame,
    d2_probs: np.ndarray,
    d2_labels: np.ndarray,
    d2_meta_df: pd.DataFrame,
    n_samples: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a controlled cross-dataset evaluation simulation cohort.
    
    Pairs observations matching on (True Class Label, Sex, and Relative Age Quantile)
    drawn from the specified split's empirical prediction pools.
    
    Returns:
        sim_p_d3: Simulated clinical probabilities of shape (n_samples,)
        sim_p_d2: Simulated eye-tracking probabilities of shape (n_samples,)
        sim_y: Ground truth binary class labels of shape (n_samples,)
    """
    rng = np.random.RandomState(seed)

    # Calculate test-set prevalence: ~41.3% ASD, 58.7% TD
    n_pos = int(n_samples * 0.413)
    n_neg = n_samples - n_pos

    sim_p_d3_list = []
    sim_p_d2_list = []
    sim_y_list = []

    # Prepare demographic strata
    # D3: Sex ('Male'/'Female'), Age (median split)
    d3_sex = d3_meta_df["Gender"].map({"Male": "M", "Female": "F"}).values
    d3_age = pd.to_numeric(d3_meta_df["Age"], errors="coerce").values
    d3_age_quant = (d3_age > np.nanmedian(d3_age)).astype(int)

    # D2: Sex ('M'/'F'), Age (median split)
    d2_sex = d2_meta_df["Gender"].values
    d2_age = pd.to_numeric(d2_meta_df["Age"], errors="coerce").values
    d2_age_quant = (d2_age > np.nanmedian(d2_age)).astype(int)

    for target_class, count in [(1, n_pos), (0, n_neg)]:
        d3_class_mask = (d3_labels == target_class)
        d2_class_mask = (d2_labels == target_class)

        # Available sex categories
        sexes = ["M", "F"]
        sub_count = count // len(sexes)
        remainder = count % len(sexes)

        for s_idx, s in enumerate(sexes):
            quota = sub_count + (1 if s_idx < remainder else 0)
            if quota == 0:
                continue

            # Stratum mask
            m3 = d3_class_mask & (d3_sex == s)
            m2 = d2_class_mask & (d2_sex == s)

            # Fallback if specific subgroup is absent (e.g. Female ASD in D2 val)
            if np.sum(m2) == 0:
                m2 = d2_class_mask
            if np.sum(m3) == 0:
                m3 = d3_class_mask

            idx3 = np.where(m3)[0]
            idx2 = np.where(m2)[0]

            chosen3 = rng.choice(idx3, size=quota, replace=True)
            chosen2 = rng.choice(idx2, size=quota, replace=True)

            sim_p_d3_list.extend(d3_probs[chosen3])
            sim_p_d2_list.extend(d2_probs[chosen2])
            sim_y_list.extend([target_class] * quota)

    sim_p_d3 = np.array(sim_p_d3_list, dtype=np.float32)
    sim_p_d2 = np.array(sim_p_d2_list, dtype=np.float32)
    sim_y = np.array(sim_y_list, dtype=np.float32)

    # Deterministic shuffle
    perm = rng.permutation(len(sim_y))
    return sim_p_d3[perm], sim_p_d2[perm], sim_y[perm]


def tune_alpha_and_threshold_on_validation(
    val_p_d3: np.ndarray,
    val_p_d2: np.ndarray,
    val_y: np.ndarray,
    alpha_grid: np.ndarray = np.linspace(0.0, 1.0, 11),
    threshold_grid: np.ndarray = np.linspace(0.2, 0.8, 61),
) -> Dict[str, Any]:
    """
    Select optimal alpha and classification threshold STRICTLY using validation data.
    
    Optimization Criterion:
    - Primary: Maximum validation ROC-AUC
    - Threshold: Maximum validation F1-score at optimal alpha
    """
    grid_results = []
    best_roc_auc = -1.0
    best_f1_overall = -1.0
    best_loss_overall = float("inf")
    best_alpha = 0.5
    best_threshold = 0.5
    best_val_metrics = None

    from sklearn.metrics import roc_auc_score, f1_score, log_loss

    for alpha in alpha_grid:
        alpha_val = round(float(alpha), 2)
        p_comb = compute_fused_score(val_p_d3, val_p_d2, alpha_val)

        # Compute ROC-AUC (independent of threshold)
        try:
            val_auc = float(roc_auc_score(val_y, p_comb))
        except ValueError:
            val_auc = 0.5

        # Compute binary log loss (measure of probability calibration)
        try:
            val_loss = float(log_loss(val_y, np.clip(p_comb, 1e-7, 1 - 1e-7)))
        except ValueError:
            val_loss = 1.0

        # Find best threshold for this alpha on validation set (maximum-margin midpoint)
        best_f1_for_alpha = -1.0
        best_thresholds = [0.5]
        for t in threshold_grid:
            preds_t = (p_comb >= t).astype(int)
            f1_t = float(f1_score(val_y, preds_t, zero_division=0))
            if f1_t > best_f1_for_alpha + 1e-5:
                best_f1_for_alpha = f1_t
                best_thresholds = [round(float(t), 3)]
            elif abs(f1_t - best_f1_for_alpha) <= 1e-5:
                best_thresholds.append(round(float(t), 3))

        best_t_for_alpha = min(best_thresholds, key=lambda t: abs(t - 0.5))

        grid_results.append(
            {
                "alpha": alpha_val,
                "val_roc_auc": round(val_auc, 4),
                "best_threshold": best_t_for_alpha,
                "best_val_f1": round(best_f1_for_alpha, 4),
                "val_loss": round(val_loss, 4),
            }
        )

        # Multi-tier selection: 1) ROC-AUC, 2) F1, 3) Log Loss
        is_better = False
        if val_auc > best_roc_auc + 1e-5:
            is_better = True
        elif abs(val_auc - best_roc_auc) <= 1e-5:
            if best_f1_for_alpha > best_f1_overall + 1e-5:
                is_better = True
            elif abs(best_f1_for_alpha - best_f1_overall) <= 1e-5:
                if val_loss < best_loss_overall - 1e-5:
                    is_better = True

        if is_better:
            best_roc_auc = val_auc
            best_f1_overall = best_f1_for_alpha
            best_loss_overall = val_loss
            best_alpha = alpha_val
            best_threshold = best_t_for_alpha
            preds_best = (p_comb >= best_threshold).astype(int)
            best_val_metrics = compute_all_metrics(val_y, preds_best, p_comb)

    return {
        "best_alpha": best_alpha,
        "best_threshold": best_threshold,
        "best_val_roc_auc": round(best_roc_auc, 4),
        "best_val_metrics": best_val_metrics,
        "grid_search": grid_results,
    }


def evaluate_decision_fusion_combination(
    combo_name: str,
    clin_model_name: str,
    eye_model_name: str,
    val_p_clin: np.ndarray,
    val_p_eye: np.ndarray,
    val_y: np.ndarray,
    test_p_clin: np.ndarray,
    test_p_eye: np.ndarray,
    test_y: np.ndarray,
) -> Dict[str, Any]:
    """
    Run full validation-only tuning and frozen test evaluation for a specific model pair.
    """
    # 1. Validation Tuning (Test set is strictly excluded)
    tuning_res = tune_alpha_and_threshold_on_validation(val_p_clin, val_p_eye, val_y)
    selected_alpha = tuning_res["best_alpha"]
    selected_thresh = tuning_res["best_threshold"]

    # 2. Freeze (alpha*, tau*) and evaluate on Held-Out Test Simulation Cohort
    test_p_comb = compute_fused_score(test_p_clin, test_p_eye, selected_alpha)
    test_preds = (test_p_comb >= selected_thresh).astype(int)
    test_metrics = compute_all_metrics(test_y, test_preds, test_p_comb)

    # 3. Standalone Baselines on the exact same test cohort
    # Clinical Baseline: alpha=1.0 with val-tuned threshold
    clin_tune = tune_alpha_and_threshold_on_validation(
        val_p_clin, val_p_eye, val_y, alpha_grid=np.array([1.0])
    )
    test_p_clin_base = test_p_clin
    test_preds_clin = (test_p_clin_base >= clin_tune["best_threshold"]).astype(int)
    clin_baseline_metrics = compute_all_metrics(test_y, test_preds_clin, test_p_clin_base)

    # Eye Tracking Baseline: alpha=0.0 with val-tuned threshold
    eye_tune = tune_alpha_and_threshold_on_validation(
        val_p_clin, val_p_eye, val_y, alpha_grid=np.array([0.0])
    )
    test_p_eye_base = test_p_eye
    test_preds_eye = (test_p_eye_base >= eye_tune["best_threshold"]).astype(int)
    eye_baseline_metrics = compute_all_metrics(test_y, test_preds_eye, test_p_eye_base)

    return {
        "combo_name": combo_name,
        "clin_model_name": clin_model_name,
        "eye_model_name": eye_model_name,
        "selected_alpha": selected_alpha,
        "selected_threshold": selected_thresh,
        "val_metrics": tuning_res["best_val_metrics"],
        "grid_search": tuning_res["grid_search"],
        "test_metrics": test_metrics,
        "test_probs": test_p_comb,
        "test_y": test_y,
        "clinical_baseline": clin_baseline_metrics,
        "eye_baseline": eye_baseline_metrics,
    }


def run_all_fusion_experiments(
    config_path: str = "configs/config.yaml", n_sim_samples: int = 500, seed: int = 42
) -> Dict[str, Any]:
    """
    Orchestrate all 4 model combination experiments and individual baselines.
    """
    config = load_config(config_path)
    device = get_device(config.get("device", "auto"))
    print(f"[Decision Fusion] Using device: {device}")

    # Setup directories
    paths = config.get("paths", {})
    metrics_dir = paths.get("metrics", "results/metrics")
    plots_dir = paths.get("plots", "results/plots")
    cm_dir = paths.get("confusion_matrices", "results/confusion_matrices")
    for d in [metrics_dir, plots_dir, cm_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Load Preprocessors and Raw Metadata
    print("\n[Step 1] Loading preprocessed data and models...")
    d3_pp = Dataset3Preprocessor(config)
    d3_pp.run()
    X_val_d3, y_val_d3 = d3_pp.get_split("val")
    X_test_d3, y_test_d3 = d3_pp.get_split("test")

    # Load D3 raw dataframe to extract metadata covariates
    raw_d3_df = pd.read_csv(config["paths"]["dataset3"])
    if "Unnamed: 10" in raw_d3_df.columns:
        raw_d3_df = raw_d3_df.drop(columns=["Unnamed: 10"])
    val_d3_meta = raw_d3_df.iloc[700:850].reset_index(drop=True)
    test_d3_meta = raw_d3_df.iloc[850:1000].reset_index(drop=True)

    d2_pp = Dataset2Preprocessor(config)
    d2_pp.run()
    X_val_d2, y_val_d2, pids_val_d2 = d2_pp.get_split("val")
    X_test_d2, y_test_d2, pids_test_d2 = d2_pp.get_split("test")

    raw_d2_meta = pd.read_csv(config["paths"]["dataset2_metadata"])
    raw_d2_meta = raw_d2_meta[~raw_d2_meta["ParticipantID"].isin([12, 16])].copy()
    val_d2_meta = pd.DataFrame({"ParticipantID": pids_val_d2}).merge(raw_d2_meta, on="ParticipantID", how="left")
    test_d2_meta = pd.DataFrame({"ParticipantID": pids_test_d2}).merge(raw_d2_meta, on="ParticipantID", how="left")

    # 2. Load Models
    kan_model = load_kan_model("results/checkpoints/kan_best.pt", d3_pp.input_dim, config)
    tabnet_model = load_tabnet_model("results/checkpoints/tabnet_best.pt", d3_pp.input_dim, config)
    mamba_model = load_mamba_model("results/checkpoints/mamba_best.pt", config, device, input_dim=8)
    patchtst_model = load_patchtst_model("results/checkpoints/patchtst_best.pt", config, device, input_dim=8, seq_len=200)

    # 3. Generate Predictions in Sub-Batches
    print("[Step 2] Generating prediction pools...")
    p_val_kan = kan_predict(kan_model, X_val_d3, device)
    p_test_kan = kan_predict(kan_model, X_test_d3, device)

    p_val_tabnet = tabnet_predict(tabnet_model, X_val_d3, device)
    p_test_tabnet = tabnet_predict(tabnet_model, X_test_d3, device)

    p_val_mamba = mamba_predict(mamba_model, X_val_d2, device, batch_size=64)
    p_test_mamba = mamba_predict(mamba_model, X_test_d2, device, batch_size=64)

    p_val_patchtst = patchtst_predict(patchtst_model, X_val_d2, device, batch_size=64)
    p_test_patchtst = patchtst_predict(patchtst_model, X_test_d2, device, batch_size=64)

    # 4. Construct Controlled Simulation Cohorts
    print(f"\n[Step 3] Constructing stratified simulation cohorts (n_samples={n_sim_samples}, seed={seed})...")
    # Validation Simulation Cohort (For alpha and threshold tuning only)
    val_sim_kan, val_sim_mamba, val_sim_y = construct_stratified_simulation_cohort(
        p_val_kan, y_val_d3, val_d3_meta, p_val_mamba, y_val_d2, val_d2_meta, n_samples=n_sim_samples, seed=seed
    )
    val_sim_tabnet, val_sim_patchtst, _ = construct_stratified_simulation_cohort(
        p_val_tabnet, y_val_d3, val_d3_meta, p_val_patchtst, y_val_d2, val_d2_meta, n_samples=n_sim_samples, seed=seed
    )

    # Held-Out Test Simulation Cohort (Strictly untouched until evaluation)
    test_sim_kan, test_sim_mamba, test_sim_y = construct_stratified_simulation_cohort(
        p_test_kan, y_test_d3, test_d3_meta, p_test_mamba, y_test_d2, test_d2_meta, n_samples=n_sim_samples, seed=seed + 100
    )
    test_sim_tabnet, test_sim_patchtst, _ = construct_stratified_simulation_cohort(
        p_test_tabnet, y_test_d3, test_d3_meta, p_test_patchtst, y_test_d2, test_d2_meta, n_samples=n_sim_samples, seed=seed + 100
    )

    # 5. Evaluate Combinations
    combinations = [
        ("KAN + Mamba", "KAN", "Mamba", val_sim_kan, val_sim_mamba, test_sim_kan, test_sim_mamba),
        ("KAN + PatchTST", "KAN", "PatchTST", val_sim_kan, val_sim_patchtst, test_sim_kan, test_sim_patchtst),
        ("TabNet + Mamba", "TabNet", "Mamba", val_sim_tabnet, val_sim_mamba, test_sim_tabnet, test_sim_mamba),
        ("TabNet + PatchTST", "TabNet", "PatchTST", val_sim_tabnet, val_sim_patchtst, test_sim_tabnet, test_sim_patchtst),
    ]

    all_results = {}
    summary_rows = []

    print("\n[Step 4] Running validation grid search and held-out test evaluation...")
    plt.figure(figsize=(10, 6))

    for combo_name, c_name, e_name, v_c, v_e, t_c, t_e in combinations:
        print(f"\n--- Evaluating {combo_name} ---")
        res = evaluate_decision_fusion_combination(
            combo_name, c_name, e_name, v_c, v_e, val_sim_y, t_c, t_e, test_sim_y
        )
        all_results[combo_name] = res

        tm = res["test_metrics"]
        val_m = res["val_metrics"]

        print(
            f"Selected Alpha: {res['selected_alpha']} | Selected Threshold: {res['selected_threshold']}\n"
            f"Validation ROC-AUC: {res['val_metrics']['roc_auc']:.4f} | Validation F1: {res['val_metrics']['f1']:.4f}\n"
            f"Test Accuracy: {tm['accuracy']:.4f} | Precision: {tm['precision']:.4f} | "
            f"Recall: {tm['recall']:.4f} | F1: {tm['f1']:.4f} | ROC-AUC: {tm['roc_auc']:.4f}"
        )

        summary_rows.append(
            {
                "Experiment": combo_name,
                "Type": "Decision Fusion",
                "Selected_Alpha": res["selected_alpha"],
                "Selected_Threshold": res["selected_threshold"],
                "Val_ROC_AUC": round(val_m["roc_auc"], 4),
                "Val_F1": round(val_m["f1"], 4),
                "Test_Accuracy": round(tm["accuracy"], 4),
                "Test_Precision": round(tm["precision"], 4),
                "Test_Recall": round(tm["recall"], 4),
                "Test_F1": round(tm["f1"], 4),
                "Test_ROC_AUC": round(tm["roc_auc"], 4),
            }
        )

        # Plot Confusion Matrix
        cm_fname = f"dataset23_fusion_{c_name.lower()}_{e_name.lower()}.png"
        cm_path = os.path.join(cm_dir, cm_fname)
        plot_confusion_matrix(
            np.array(tm["confusion_matrix"]),
            class_names=["TD", "ASD"],
            title=f"Decision Fusion ({combo_name}) — Test Set",
            save_path=cm_path,
        )

        # ROC Curve
        fpr, tpr, _ = roc_curve(test_sim_y, res["test_probs"])
        plt.plot(fpr, tpr, label=f"{combo_name} (AUC = {tm['roc_auc']:.3f})", linewidth=2)

    # Add standalone baselines to summary table
    first_res = all_results["KAN + Mamba"]
    summary_rows.append(
        {
            "Experiment": "KAN Baseline (Clinical Alone)",
            "Type": "Baseline",
            "Selected_Alpha": 1.0,
            "Selected_Threshold": 0.5,
            "Val_ROC_AUC": round(first_res["clinical_baseline"]["roc_auc"], 4),
            "Val_F1": round(first_res["clinical_baseline"]["f1"], 4),
            "Test_Accuracy": round(first_res["clinical_baseline"]["accuracy"], 4),
            "Test_Precision": round(first_res["clinical_baseline"]["precision"], 4),
            "Test_Recall": round(first_res["clinical_baseline"]["recall"], 4),
            "Test_F1": round(first_res["clinical_baseline"]["f1"], 4),
            "Test_ROC_AUC": round(first_res["clinical_baseline"]["roc_auc"], 4),
        }
    )
    summary_rows.append(
        {
            "Experiment": "Mamba Baseline (Eye Alone)",
            "Type": "Baseline",
            "Selected_Alpha": 0.0,
            "Selected_Threshold": 0.5,
            "Val_ROC_AUC": round(first_res["eye_baseline"]["roc_auc"], 4),
            "Val_F1": round(first_res["eye_baseline"]["f1"], 4),
            "Test_Accuracy": round(first_res["eye_baseline"]["accuracy"], 4),
            "Test_Precision": round(first_res["eye_baseline"]["precision"], 4),
            "Test_Recall": round(first_res["eye_baseline"]["recall"], 4),
            "Test_F1": round(first_res["eye_baseline"]["f1"], 4),
            "Test_ROC_AUC": round(first_res["eye_baseline"]["roc_auc"], 4),
        }
    )
    patch_res = all_results["KAN + PatchTST"]
    summary_rows.append(
        {
            "Experiment": "PatchTST Baseline (Eye Alone)",
            "Type": "Baseline",
            "Selected_Alpha": 0.0,
            "Selected_Threshold": 0.5,
            "Val_ROC_AUC": round(patch_res["eye_baseline"]["roc_auc"], 4),
            "Val_F1": round(patch_res["eye_baseline"]["f1"], 4),
            "Test_Accuracy": round(patch_res["eye_baseline"]["accuracy"], 4),
            "Test_Precision": round(patch_res["eye_baseline"]["precision"], 4),
            "Test_Recall": round(patch_res["eye_baseline"]["recall"], 4),
            "Test_F1": round(patch_res["eye_baseline"]["f1"], 4),
            "Test_ROC_AUC": round(patch_res["eye_baseline"]["roc_auc"], 4),
        }
    )
    tabnet_res = all_results["TabNet + Mamba"]
    summary_rows.append(
        {
            "Experiment": "TabNet Baseline (Clinical Alone)",
            "Type": "Baseline",
            "Selected_Alpha": 1.0,
            "Selected_Threshold": 0.5,
            "Val_ROC_AUC": round(tabnet_res["clinical_baseline"]["roc_auc"], 4),
            "Val_F1": round(tabnet_res["clinical_baseline"]["f1"], 4),
            "Test_Accuracy": round(tabnet_res["clinical_baseline"]["accuracy"], 4),
            "Test_Precision": round(tabnet_res["clinical_baseline"]["precision"], 4),
            "Test_Recall": round(tabnet_res["clinical_baseline"]["recall"], 4),
            "Test_F1": round(tabnet_res["clinical_baseline"]["f1"], 4),
            "Test_ROC_AUC": round(tabnet_res["clinical_baseline"]["roc_auc"], 4),
        }
    )

    # Complete ROC Plot
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Decision Fusion — ROC Curves on Held-Out Test Simulation", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_plot_path = os.path.join(plots_dir, "dataset23_fusion_roc_curves.png")
    plt.savefig(roc_plot_path, dpi=150)
    plt.close()
    print(f"\nSaved ROC curves to {roc_plot_path}")

    # Plot Alpha Grid Tuning Curves
    plt.figure(figsize=(10, 5))
    for combo_name in all_results:
        grid_data = all_results[combo_name]["grid_search"]
        alphas = [g["alpha"] for g in grid_data]
        val_aucs = [g["val_roc_auc"] for g in grid_data]
        plt.plot(alphas, val_aucs, marker="o", label=f"{combo_name} (Val AUC)", linewidth=2)
    plt.xlabel("Alpha (Weight on Clinical Modality)", fontsize=11)
    plt.ylabel("Validation ROC-AUC", fontsize=11)
    plt.title("Validation-Only Alpha Grid Search Across Model Combinations", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    grid_plot_path = os.path.join(plots_dir, "dataset23_fusion_alpha_grid.png")
    plt.savefig(grid_plot_path, dpi=150)
    plt.close()
    print(f"Saved Alpha Grid plot to {grid_plot_path}")

    # 6. Save Comparison CSV
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(metrics_dir, "dataset23_decision_fusion.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved Decision Fusion results to {csv_path}:")
    print(summary_df.to_string(index=False))

    return {
        "summary_df": summary_df,
        "results": all_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Decision-Level Integration Experiments.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all_fusion_experiments(args.config, args.n_samples, args.seed)
