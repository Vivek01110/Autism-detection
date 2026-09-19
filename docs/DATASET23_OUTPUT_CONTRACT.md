# Standardized ASD Classification Output Contract (Dataset 2 + Dataset 3)

**Target Audience:** Teammates developing downstream modules (e.g., Dataset 1 / video / multimodal integration).  
**Status:** Frozen & Verified  
**Scope Covered:** Dataset 2 (Eye Tracking) and Dataset 3 (Clinical / Tabular) ASD Classification.  
**Strict Scope Boundary:** Dataset 1 (Video / MMASD / SkeletonMAE / PoseFormerV2) and Severity Assessment are handled separately by teammates and are outside this interface.

---

## 1. Overview & Architecture

The Dataset 2 + Dataset 3 subsystem provides binary Autism Spectrum Disorder (ASD) classification outputs:
- **0 = TD / Non-ASD** (Typically Developing)
- **1 = ASD** (Autism Spectrum Disorder)

Together with calibrated posterior probabilities $P(\text{ASD} \mid \mathbf{x}) \in [0.0, 1.0]$.

Downstream consumers can query:
1. **Dataset 2 Specialists** (Eye-tracking scanpaths & pupil dynamics)
2. **Dataset 3 Specialists** (Clinical screening survey scores & history)
3. **Decision-Level Fusion Policy** (Statistically calibrated weighted integration of modality specialist probabilities)

```
             Dataset 3 (Clinical/Tabular)                Dataset 2 (Eye-Tracking Sequences)
              [Age, Scores, Demographics]                       [(200, 8) Gaze Tensors]
                          │                                                │
                 ┌────────┴────────┐                              ┌────────┴────────┐
                 ▼                 ▼                              ▼                 ▼
             KAN Model        TabNet Model                   Mamba Model      PatchTST Model
            (Primary)         (Benchmark)                     (Primary)        (Benchmark)
                 │                 │                              │                 │
                 └────────┬────────┘                              └────────┬────────┘
                          │                                                │
                          ▼                                                ▼
                   P(ASD|Clinical)                                    P(ASD|Eye)
                          │                                                │
                          └───────────────────────┬────────────────────────┘
                                                  ▼
                                     Decision-Level Integration
                                 P_comb = α·P_clin + (1-α)·P_eye
                                                  │
                                                  ▼
                                    Standardized Output Contract
                               {predicted_label, asd_probability, ...}
```

---

## 2. Models Provided & Checkpoints

### A. Dataset 2 Models (Eye-Tracking Sequence Specialists)
Input tensor shape: `(B, 200, 8)` representing 200 temporal gaze steps sampled at 60 Hz across 8 calibrated channels:
1. `Point of Regard Right X [px]`
2. `Point of Regard Right Y [px]`
3. `Point of Regard Left X [px]`
4. `Point of Regard Left Y [px]`
5. `Pupil Diameter Right [mm]`
6. `Pupil Diameter Left [mm]`
7. `Gaze_Velocity`
8. `Fixation_Flag`

* **Primary: Mamba (`MambaClassifier`)**
  - Architecture: Selective State Space Model with bidirectional block structure, local 1D convolution (`d_conv=4`), expansion factor $E=2$, State dimension $N=16$, Model dimension $D=64$.
  - Checkpoint: `results/checkpoints/mamba_best.pt` (302 KB)
  - Test Metrics: Accuracy: **72.99%**, Precision: **63.05%**, Recall: **87.10%**, F1: **73.15%**, ROC-AUC: **0.8121**
* **Benchmark: PatchTST (`PatchTSTClassifier`)**
  - Architecture: Patch-based Channel-Independent Transformer with patching (`patch_len=16`, `stride=8`), multi-head attention (`d_model=64`, 4 heads, 2 layers).
  - Checkpoint: `results/checkpoints/patchtst_best.pt` (353 KB)
  - Test Metrics: Accuracy: **79.73%**, Precision: **71.96%**, Recall: **85.25%**, F1: **78.04%**, ROC-AUC: **0.8910**

### B. Dataset 3 Models (Clinical / Tabular Screening Specialists)
Input feature vector: `(1, 8)` or `(B, 8)` containing 4 z-score normalized continuous metrics and 4 binary flags:
1. `Age` (months)
2. `Social_Interaction_Score` (0–10)
3. `Communication_Score` (0–10)
4. `Repetitive_Behavior_Score` (0–10)
5. `Gender` (`Male`=1, `Female`=0)
6. `Jaundice` (`Yes`=1, `No`=0)
7. `Family_ASD_History` (`Yes`=1, `No`=0)
8. `Language_Delay` (`Yes`=1, `No`=0)

* **Primary: KAN (`Kolmogorov-Arnold Network`)**
  - Architecture: Learnable B-spline activation functions on network edges, grid size $G=5$, spline order $k=3$, hidden dimensions `[64, 32]`, layer normalization, dropout.
  - Checkpoint: `results/checkpoints/kan_best.pt` (102 KB)
  - Test Metrics: Accuracy: **58.67%**, Precision: **50.00%**, Recall: **25.81%**, F1: **34.04%**, ROC-AUC: **0.5409**
* **Benchmark: TabNet (`TabNet`)**
  - Architecture: Attentive sequential multi-step feature selection ($N_{\text{steps}}=3$) with Sparsemax attention masks and Ghost Batch Normalization.
  - Checkpoint: `results/checkpoints/tabnet_best.pt` (167 KB)
  - Test Metrics: Accuracy: **56.67%**, Precision: **38.46%**, Recall: **8.06%**, F1: **13.33%**, ROC-AUC: **0.4665**

---

## 3. Standardized Output Schema Contract

Every inference query produces an `ASDClassificationResult` instance or a standard JSON payload adhering to the following schema:

```json
{
  "predicted_label": 1,
  "asd_probability": 0.8245,
  "model": "mamba",
  "dataset": "dataset_2",
  "split": "inference",
  "metadata": {
    "participant_id": "17",
    "decision_threshold": 0.5,
    "confidence": "high"
  }
}
```

### Exact Field Specifications:

| Field Name | Type | Range / Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `predicted_label` | `int` | `0` or `1` | Binary ASD classification decision (`0` = TD / Non-ASD, `1` = ASD). |
| `asd_probability` | `float` | `[0.0, 1.0]` | Calibrated model posterior probability for ASD class $P(\text{ASD} \mid \mathbf{x})$. |
| `model` | `str` | `"kan"`, `"tabnet"`, `"mamba"`, `"patchtst"`, `"<clin>_<eye>_fusion"` | Name of the producing neural network or fusion ensemble. |
| `dataset` | `str` | `"dataset_2"`, `"dataset_3"`, `"decision_fusion"` | Underlying modality domain source. |
| `split` | `str` | `"inference"`, `"test"`, `"val"`, `"train"` | Data partition or operational context. |
| `metadata` | `dict` | Key-value pairs | Optional diagnostic/operational context. Note: `participant_id` is strictly stored here and is NEVER passed into the models. |

---

## 4. Decision-Level Integration Output

For combining predictions from Dataset 2 and Dataset 3, the linear decision-fusion rule is:
$$P_{\text{combined}} = \alpha \cdot P_{\text{clinical}} + (1 - \alpha) \cdot P_{\text{eye}}$$

Tuned on the validation simulation cohort and frozen before test evaluation:

| Combination | Selected $\alpha^*$ | Optimal Threshold $\tau^*$ | Val ROC-AUC | Test Accuracy | Test F1 | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KAN + Mamba** | **0.9** | **0.23** | **0.6286** | **42.40%** | **0.5886** | **0.6197** |
| **KAN + PatchTST** | **1.0** | **0.24** | **0.6278** | **42.60%** | **0.5894** | **0.5319** |
| **TabNet + Mamba** | **0.9** | **0.27** | **0.5804** | **43.40%** | **0.5757** | **0.5252** |
| **TabNet + PatchTST**| **1.0** | **0.30** | **0.5688** | **37.80%** | **0.4808** | **0.4333** |

Example Decision Fusion Payload:
```json
{
  "predicted_label": 1,
  "asd_probability": 0.6520,
  "model": "kan_mamba_fusion",
  "dataset": "decision_fusion",
  "split": "inference",
  "metadata": {
    "clinical_model": "kan",
    "eye_tracking_model": "mamba",
    "p_clinical": 0.68,
    "p_eye": 0.40,
    "fusion_alpha": 0.9,
    "decision_threshold": 0.23,
    "scientific_note": "Decision-level fusion from independent modality specialist scores."
  }
}
```

---

## 5. Critical Constraints & Limitations

> [!IMPORTANT]
> **Unpaired Dataset Cohorts:**
> Dataset 2 and Dataset 3 comprise completely distinct cohorts of children:
> - Dataset 2: 57 school-aged children (ages 2.7–12.9 years, median 8.1y) from a Turkish clinical eye-tracking study with integer IDs.
> - Dataset 3: 1,000 toddler screening records (ages 12–36 months, median 24m) collected via mobile Q-CHAT screening with UUID strings.
> 
> Consequently:
> 1. **No Patient-Level Concatenation:** Do NOT attempt row-level or feature-level concatenation of raw data between Dataset 2 and Dataset 3.
> 2. **Controlled Simulation Testbed:** Decision fusion metrics reflect a controlled simulation testbed matching demographic marginals, NOT ground-truth multimodal recordings of the same individuals.
> 3. **Non-Autonomous Diagnostic System:** Outputs are investigative screening indicators and do not constitute clinical validation or an autonomous diagnostic verdict.

---

## 6. How Teammates Can Consume the Output

Downstream teammates do NOT need to modify any of the Dataset 2 or Dataset 3 models. The interface can be imported directly in Python or called from the command line.

### Method 1: Direct Python Import (Recommended)

```python
from src.classification import StandardizedASDClassifier

# Initialize classifier (loads configs and metadata automatically)
classifier = StandardizedASDClassifier(config_path="configs/config.yaml")

# 1. Classify a Dataset 3 Clinical Record
# Accepts either a dictionary with feature names or a preprocessed numpy vector (1, 8)
clinical_sample = {
    "Child_ID": "CH-1042",  # Will be safely isolated to metadata; NOT fed to model
    "Age": 28,
    "Social_Interaction_Score": 6.5,
    "Communication_Score": 5.0,
    "Repetitive_Behavior_Score": 4.0,
    "Gender": "Male",
    "Jaundice": "No",
    "Family_ASD_History": "Yes",
    "Language_Delay": "Yes",
}
clinical_result = classifier.classify_dataset3_sample(clinical_sample, model_name="kan")
print(clinical_result.to_dict())
# -> {'predicted_label': 1, 'asd_probability': 0.7412, 'model': 'kan', ...}

# 2. Classify a Dataset 2 Eye-Tracking Sequence
# Accepts a numpy array or torch tensor of shape (200, 8) or (1, 200, 8)
import numpy as np
eye_sequence = np.random.randn(200, 8)
eye_result = classifier.classify_dataset2_sample(
    eye_sequence, model_name="mamba", participant_id=14
)
print(eye_result.to_dict())
# -> {'predicted_label': 1, 'asd_probability': 0.8105, 'model': 'mamba', ...}

# 3. Decision-Level Fusion
fused_result = classifier.classify_decision_fusion(
    p_clinical=clinical_result.asd_probability,
    p_eye=eye_result.asd_probability,
    clin_model="kan",
    eye_model="mamba",
)
print(fused_result.to_dict())
# -> {'predicted_label': 1, 'asd_probability': 0.7481, 'model': 'kan_mamba_fusion', ...}
```

### Method 2: Batch Processing
```python
# Batch clinical records (N, 8)
batch_x = np.random.randn(10, 8)
results_d3 = classifier.classify_dataset3_batch(batch_x, model_name="kan")

# Batch eye sequences (N, 200, 8)
batch_seq = np.random.randn(5, 200, 8)
results_d2 = classifier.classify_dataset2_batch(batch_seq, model_name="mamba")
```

### Method 3: Standalone Factory
If generating outputs from external model pipelines, use `standardize_classification_output`:
```python
from src.classification import standardize_classification_output

result = standardize_classification_output(
    predicted_label=1,
    asd_probability=0.78,
    model="custom_model",
    dataset="dataset_1",
    split="inference",
    metadata={"framerate": 30}
)
```
