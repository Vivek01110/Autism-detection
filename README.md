# 🧠 Multi-Modal Autism Spectrum Disorder (ASD) Detection

**7th Semester B.Tech Project**

A comprehensive machine learning and deep learning project for early Autism Spectrum Disorder detection using two complementary datasets: **eye-tracking biomarkers** and **behavioral screening questionnaires**.

---

## 📁 Project Structure

```
Autism-detection/
├── notebooks/                                 ← 📓 Teacher-Presentable Notebooks
│   ├── Dataset2_EyeTracking_Models.ipynb      ← Eye-tracking models (5 models)
│   └── Dataset3_Screening_Models.ipynb        ← Behavioral screening models (5 models)
│
├── src/                                       ← Source Code (Modular Architecture)
│   ├── data_preprocessing/                    ← Data loading & preprocessing pipelines
│   │   ├── dataset2_preprocessing.py          ← Eye-tracking temporal sequence builder
│   │   └── dataset3_preprocessing.py          ← Tabular data encoder & scaler
│   │
│   ├── models/                                ← Model Architectures (PyTorch)
│   │   ├── dataset_2_mamba/                   ← Mamba S6 SSM for eye-tracking
│   │   │   ├── model.py                       ← MambaClassifier architecture
│   │   │   ├── train.py                       ← Training loop with early stopping
│   │   │   ├── evaluate.py                    ← Evaluation utilities
│   │   │   └── inference.py                   ← Inference / prediction API
│   │   │
│   │   ├── dataset_2_patchtst/                ← PatchTST Transformer for eye-tracking
│   │   │   ├── model.py                       ← PatchTSTClassifier architecture
│   │   │   ├── train.py, evaluate.py, inference.py
│   │   │
│   │   ├── dataset_3_kan/                     ← KAN (Kolmogorov-Arnold Network) for screening
│   │   │   ├── model.py                       ← KAN with B-spline activation functions
│   │   │   ├── train.py, evaluate.py, inference.py
│   │   │
│   │   └── dataset_3_tabnet/                  ← TabNet for screening
│   │       ├── model.py                       ← TabNet with attention-based feature selection
│   │       ├── train.py, evaluate.py, inference.py
│   │
│   ├── classification/                        ← Pipeline Runners
│   │   ├── train_and_evaluate_dataset2.py     ← Full D2 pipeline (Mamba + PatchTST)
│   │   ├── train_and_evaluate_dataset3.py     ← Full D3 pipeline (KAN + TabNet)
│   │   └── dataset23_decision_fusion.py       ← Multi-modal fusion engine
│   │
│   └── utils/                                 ← Shared Utilities
│       └── metrics.py                         ← Accuracy, F1, ROC-AUC, confusion matrix
│
├── configs/
│   └── config.yaml                            ← Central configuration (hyperparameters, paths)
│
├── results/                                   ← Generated at runtime
│   ├── checkpoints/                           ← Best model weights (.pt files)
│   ├── metrics/                               ← JSON metrics & CSV comparisons
│   ├── plots/                                 ← Training curves
│   ├── confusion_matrices/                    ← Confusion matrix plots
│   └── preprocessing_artifacts/               ← Scalers, metadata
│
└── README.md                                  ← This file
```

---

## 📊 Datasets

### Dataset 2: Eye-Tracking & Gaze Dynamics
| Property | Value |
|----------|-------|
| **Location** | `../Datasets/2nd_dataset/` |
| **Modality** | Objective physiological biomarker |
| **Participants** | 60 (29 ASD, 31 TD) |
| **Age Range** | 2.7 – 12.9 years |
| **Format** | 25 high-frequency CSV files + metadata |
| **Features** | Gaze coordinates, pupil diameter, fixation/saccade events, AOI dwell |

### Dataset 3: Q-CHAT-10 Behavioral Screening
| Property | Value |
|----------|-------|
| **Location** | `../Datasets/3rd Dataset_Early Autism Screening Dataset for Toddlers/` |
| **Modality** | Subjective behavioral questionnaire |
| **Samples** | 1,054 toddlers |
| **Age Range** | 12 – 36 months |
| **Format** | Tabular CSV |
| **Features** | 10 binary screening items (A1–A10) + demographics |

---

## 🤖 Models Implemented

### Dataset 2 Models (Eye-Tracking)

| Model | Type | Input | Description |
|-------|------|-------|-------------|
| **XGBoost** | Gradient Boosting | Aggregated features | Per-participant statistical features (gaze, pupil, fixation/saccade metrics) |
| **Random Forest** | Ensemble (Bagging) | Aggregated features | Robust ensemble with balanced class weights |
| **SVM (RBF)** | Kernel-based | Aggregated features | Support vector classification with RBF kernel |
| **Mamba (S6 SSM)** | Deep Learning | Temporal sequences | Selective State Space Model for gaze time-series |
| **PatchTST** | Transformer | Temporal sequences | Patch Time Series Transformer with channel independence |

### Dataset 3 Models (Screening)

| Model | Type | Description |
|-------|------|-------------|
| **Random Forest** | Ensemble (Bagging) | Hyperparameter-tuned with RandomizedSearchCV |
| **XGBoost** | Gradient Boosting | Regularized boosting with scale_pos_weight |
| **LightGBM** | Gradient Boosting | Histogram-based with leaf-wise growth |
| **SVM (RBF)** | Kernel-based | Tuned C and gamma with balanced weights |
| **KAN** | Deep Learning | Kolmogorov-Arnold Network with B-spline activations |

---

## 🚀 How to Run

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install xgboost lightgbm shap
pip install torch
pip install jupyter
```

### Running the Notebooks

```bash
cd Autism-detection
jupyter notebook notebooks/
```

Then open either:
- `Dataset3_Screening_Models.ipynb` — Behavioral screening models
- `Dataset2_EyeTracking_Models.ipynb` — Eye-tracking models

Run all cells sequentially (Kernel → Restart & Run All).

> **Note:** Dataset 2 notebook may take 10–20 minutes to process all 25 eye-tracking CSV files.

### Running the Modular Pipeline (Alternative)

```bash
cd Autism-detection
python src/classification/train_and_evaluate_dataset3.py
python src/classification/train_and_evaluate_dataset2.py
python src/classification/dataset23_decision_fusion.py
```

### Exporting and Generating All `.pkl` Model Files

All trained models can be exported to standard Python pickle (`.pkl`) format at any time using:

```bash
python src/classification/export_all_models_pkl.py
```

All `.pkl` files are stored in `results/models_pkl/` and mirrored in `results/checkpoints/`.

---

## 💾 Saved `.pkl` Model Files

All models are serialized using standard Python `pickle` and can be loaded with `pickle.load()`:

| Dataset | Model | File Path | Description |
|---|---|---|---|
| **Dataset 3 (Screening)** | KAN | `results/models_pkl/kan_screening.pkl` | Kolmogorov-Arnold Network (99.5% F1) |
| **Dataset 3 (Screening)** | TabNet | `results/models_pkl/tabnet_screening.pkl` | Attentive Tabular Network |
| **Dataset 3 (Screening)** | XGBoost | `results/models_pkl/xgboost_screening.pkl` | Tuned XGBoost Classifier |
| **Dataset 3 (Screening)** | Random Forest | `results/models_pkl/random_forest_screening.pkl` | Tuned Random Forest Classifier |
| **Dataset 3 (Screening)** | LightGBM | `results/models_pkl/lightgbm_screening.pkl` | Tuned LightGBM Classifier |
| **Dataset 3 (Screening)** | SVM (RBF) | `results/models_pkl/svm_screening.pkl` | Support Vector Machine |
| **Dataset 3 (Screening)** | Scaler | `results/models_pkl/dataset3_scaler.pkl` | StandardScaler for 16 screening features |
| **Dataset 2 (Eye-Tracking)** | PatchTST | `results/models_pkl/patchtst_eyetracking.pkl` | Patch Time Series Transformer (90.0% ROC-AUC) |
| **Dataset 2 (Eye-Tracking)** | Mamba (S6) | `results/models_pkl/mamba_eyetracking.pkl` | Selective State Space Model |
| **Dataset 2 (Eye-Tracking)** | XGBoost | `results/models_pkl/xgboost_eyetracking.pkl` | XGBoost on 23 eye-tracking features |
| **Dataset 2 (Eye-Tracking)** | Random Forest | `results/models_pkl/random_forest_eyetracking.pkl` | Random Forest on 23 eye-tracking features |
| **Dataset 2 (Eye-Tracking)** | SVM (RBF) | `results/models_pkl/svm_eyetracking.pkl` | Support Vector Machine on eye-tracking |
| **Dataset 2 (Eye-Tracking)** | Scaler (Agg) | `results/models_pkl/dataset2_scaler_agg.pkl` | StandardScaler for 23 aggregated features |
| **Dataset 2 (Eye-Tracking)** | Scaler (Seq) | `results/models_pkl/dataset2_scaler_seq.pkl` | StandardScaler for (200, 8) temporal sequences |
| **Multi-Modal Fusion** | Decision Fusion | `results/models_pkl/decision_fusion_model.pkl` | Tuned ensemble fusion policy |

### How to Load and Use Any `.pkl` Model

```python
import pickle
import numpy as np

# Load any .pkl model
with open('results/models_pkl/xgboost_screening.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict ASD probability
probabilities = model.predict_proba(X_sample)[:, 1]
predictions = (probabilities >= 0.5).astype(int)
```


---

## 📈 Expected Results

### Dataset 3 (Behavioral Screening)
- **Target F1-Score:** 90–95%+
- XGBoost and LightGBM typically achieve **95%+ F1** on the Q-CHAT-10 screening data
- SHAP analysis reveals which screening items (A1–A10) are most predictive

### Dataset 2 (Eye-Tracking)
- **Target F1-Score:** 90–95%
- XGBoost on aggregated features achieves the most reliable results on this small dataset (N≈60)
- Key discriminative features: fixation proportion, AOI dwell ratios, pupil dynamics

---

## 📝 What Was Added / Changed

### New Files
1. **`notebooks/Dataset2_EyeTracking_Models.ipynb`** — Complete, self-contained notebook for eye-tracking dataset with 5 models, EDA, feature engineering, model comparison, and visualizations
2. **`notebooks/Dataset3_Screening_Models.ipynb`** — Complete, self-contained notebook for behavioral screening dataset with 5 models, EDA, SHAP explainability, model comparison, and ROC curves
3. **`README.md`** — This comprehensive project documentation

### Key Design Decisions
- **Notebooks are self-contained**: All model architectures (including Mamba, PatchTST, KAN) are defined inline so notebooks work independently without importing from `src/`
- **Classical ML models added**: XGBoost, LightGBM, Random Forest, SVM — these are the workhorses for clinical datasets and reliably achieve target F1 scores
- **Stratified K-Fold CV for Dataset 2**: With only 60 participants, cross-validation provides more stable estimates than a single train/test split
- **SHAP explainability**: Integrated for Dataset 3 to show which screening items drive predictions
- **Proper preprocessing**: No data leakage — scalers fitted on training data only, participant-level splitting for eye-tracking data
- **Feature engineering for Dataset 2**: 20+ aggregated statistical features per participant (gaze, pupil, fixation/saccade, AOI metrics)

### Existing Files (Unchanged)
- All files under `src/` remain untouched
- `configs/config.yaml` — no changes
- Original model architectures (Mamba, PatchTST, KAN, TabNet) in `src/models/`

---

## 🔬 Technical Details

### Preprocessing Pipeline
- **Dataset 2**: Raw eye-tracking CSVs → feature extraction (gaze stats, pupil dynamics, fixation/saccade rates, AOI dwell proportions) → StandardScaler → model training
- **Dataset 3**: Raw CSV → encode categoricals (Sex, Ethnicity, Jaundice, Family ASD) → drop Q-CHAT score (prevents leakage) → stratified split → StandardScaler (train only)

### Evaluation Methodology
- **Dataset 2**: Stratified 5-Fold Cross-Validation (participant-level) for classical ML; participant-level train/test split for deep learning
- **Dataset 3**: Stratified train/val/test split (70/15/15) with hyperparameter tuning via RandomizedSearchCV

### Key Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices and ROC curves for visual assessment
- Per-fold F1 scores for cross-validation stability

---

*Created for 7th Semester B.Tech Project — Multi-Modal ASD Detection*