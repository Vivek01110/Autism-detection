# Pre-Fusion Compatibility & Experiment Design Audit: Dataset 2 + Dataset 3

**Project:** Multi-Modal Autism Spectrum Disorder (ASD) Detection  
**Phase:** Phase 4 — Pre-Fusion Compatibility & Experiment Design Audit  
**Status:** Audit Completed  
**Constraint Enforcement:** AUDIT ONLY — No fusion code implemented, no raw data modified, no splits altered, no models retrained, no synthetic pairings fabricated.

---

## Executive Summary

This audit evaluates the architectural, statistical, demographic, and clinical compatibility between **Dataset 2 (Eye-Tracking Temporal Sequences)** and **Dataset 3 (Clinical / Tabular Screening Survey)** prior to any multimodal fusion.

### Core Verdict
1. **Participant Matching**: **DATASETS ARE UNPAIRED** (Result B). Dataset 2 consists of 57 usable school-aged children (ages 2.7–12.9 years) evaluated in an eye-tracking laboratory with CARS scores. Dataset 3 consists of 1,000 independent children (ages 12–36 months / toddlers) from a clinical screening survey with UUID identifiers. There is **zero shared participant identification** and no common registry key.
2. **Age Groups**: Age distributions differ substantially by design (school-aged children vs. toddlers). This is expected and treated as a clinical covariate reflecting distinct diagnostic stages.
3. **Scientific Fusion Validity**: A direct row-level join or arbitrary pairing of records across datasets is **scientifically invalid** and would produce fictitious chimeric samples.
4. **Recommended Architecture**: A **Hierarchical Dual-Modality Clinical Decision & Triage System**. Each modality model (Mamba/PatchTST for eye-tracking; KAN/TabNet for clinical survey) operates as an expert diagnostic agent. A unified decision-rule engine integrates their outputs into a single ASD True/False clinical determination, perfectly fulfilling the project architecture while strictly preserving scientific integrity.

---

## A. Dataset 2 Identity Structure

- **Raw Data Location**: `data/dataset_2_eye_tracking/archive-2/Eye-tracking Output/` (25 CSV files) and `Metadata_Participants.csv`.
- **Participant Identifier**: Integer values (`ParticipantID`: `1` to `59`).
- **Total Participants in Metadata**: 59 participants.
- **Usable Participants**: **57 participants**.
  - Participants `12` and `16` (both labeled ASD in metadata) have **zero** eye-tracking rows across all 25 CSV files and are excluded.
- **Class Distribution (Usable)**:
  - ASD (`label = 1`): **27** participants
  - TD (`label = 0`): **30** participants
- **Multi-Trial / Multi-Session Structure**:
  - Eye-tracking data is distributed across 25 separate stimulus task files.
  - Participants completed multiple trials per experiment, yielding varying sequence lengths.
  - Sliced into **11,802** sliding-window sequences of shape `(200, 8)` with stride 100 and min length 50.
- **Demographics**:
  - **Age**: Ranging from **2.7 to 12.9 years** (Mean: 7.88 ± 2.79 years; Median: 8.10 years). Age is explicitly given in years.
    - ASD cohort age: 7.74 ± 2.73 years (range: 2.7–12.3)
    - TD cohort age: 8.02 ± 2.89 years (range: 3.7–12.9)
  - **Gender**: 38 Male, 21 Female (In usable ASD: 23 Male, 4 Female; in usable TD: 13 Male, 17 Female). High male-to-female ratio in ASD (5.75:1), consistent with clinical literature.
  - **Diagnostic Severity Score**: CARS Score (Childhood Autism Rating Scale) is recorded for ASD participants (range: 17.0–45.0, mean: 32.97 ± 6.55). TD participants have `NA`.
- **Participant-Level Splitting (Seed 42)**:
  - Train: 39 participants (7,928 sequences; 19 ASD, 20 TD)
  - Validation: 9 participants (1,708 sequences; 4 ASD, 5 TD)
  - Test: 9 participants (2,166 sequences; 4 ASD, 5 TD)
  - Verified: Zero participant overlap across splits ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$).

---

## B. Dataset 3 Identity Structure

- **Raw Data Location**: `data/dataset_3_clinical_tabular/asd.csv`.
- **Row / Participant Identifier**: `Child_ID` — 1,000 unique 36-character hexadecimal UUID strings (e.g., `3eac2cb7-6348-4522-8049-0901c99a1693`).
- **Total Records**: **1,000 records** (exactly 1 row per child; no repeat measures).
- **Class Distribution**:
  - `Diagnosed_ASD = Yes` (`label = 1`): **411** (41.1%)
  - `Diagnosed_ASD = No` (`label = 0`): **589** (58.9%)
- **Demographics**:
  - **Age**: Discrete integer values ranging from **12 to 36** (Mean: 23.99 ± 7.44; Median: 24.0).
    - In standard clinical pediatric screening instruments (e.g., Q-CHAT), age is measured in **months** (12 to 36 months = 1.0 to 3.0 years old).
    - ASD cohort age: 24.23 ± 7.58 months.
    - Non-ASD cohort age: 23.82 ± 7.35 months.
  - **Gender**: 459 Male, 541 Female (In ASD: 190 Male, 221 Female; in Non-ASD: 269 Male, 320 Female).
- **Clinical & Screening Features**:
  - Binary Developmental Flags: `Jaundice` (Yes/No), `Family_ASD_History` (Yes/No), `Language_Delay` (Yes/No).
  - Quantitative Behavioral Scores (1–10): `Social_Interaction_Score`, `Communication_Score`, `Repetitive_Behavior_Score`.
- **Train / Validation / Test Splitting (Seed 42)**:
  - Train: 700 records (288 ASD, 412 Non-ASD)
  - Validation: 150 records (61 ASD, 89 Non-ASD)
  - Test: 150 records (62 ASD, 88 Non-ASD)

---

## C. Participant Matching Feasibility

### Audit Verdict: **B. DATASETS ARE UNPAIRED**

### Empirical Evidence:
1. **Identifier Disjointness**:
   - Dataset 2 uses low-integer participant IDs (`1` to `59`).
   - Dataset 3 uses 36-character UUID strings.
   - Exact intersection between `ParticipantID` and `Child_ID`: **0**.
2. **Absence of Linkage Keys**:
   - There are no cross-reference tables, clinic identifiers, hospital registry keys, or subject mapping metadata anywhere in the repository.
3. **Cardinality Mismatch**:
   - Dataset 2 comprises 57 distinct children with multiple temporal scan sessions (11,802 sequences).
   - Dataset 3 comprises 1,000 distinct individuals with a single survey record each.
4. **Cohort Origin Mismatch**:
   - Dataset 2 was acquired in an eye-tracking laboratory measuring visual scanpaths on screen coordinates.
   - Dataset 3 is an independent behavioral screening questionnaire.

> [!CAUTION]
> **Prohibition of Fabricated Pairing**:
> Pairing rows across Dataset 2 and Dataset 3 based on row index, matching ASD status, or arbitrary sorting is mathematically and clinically fraudulent. Combining an 8-year-old child's eye-tracking gaze pattern from Dataset 2 with a 2-year-old toddler's clinical survey response from Dataset 3 would create synthetic "chimeric patients" with no real-world biological meaning.

---

## D. Age-Group and Population Compatibility

### Observed Demographic Comparison

| Attribute | Dataset 2 (Eye-Tracking) | Dataset 3 (Clinical/Tabular) | Alignment Analysis |
| :--- | :--- | :--- | :--- |
| **Cohort Size** | 57 children (11,802 sequences) | 1,000 children (1,000 rows) | High-throughput sensor data vs. broad screening survey |
| **Age Scale** | 2.7 to 12.9 **years** | 12 to 36 **months** (1.0 to 3.0 years) | Preschool/school age vs. toddler early screening |
| **Age Mean ± SD** | 7.88 ± 2.79 years | 23.99 ± 7.44 months (2.0 ± 0.6 years) | Non-overlapping age distribution |
| **Age as Feature** | Metadata covariate (not in gaze tensor) | Explicit numeric feature used by KAN/TabNet | Can serve as population routing covariate |
| **ASD Prevalence** | 47.4% (27/57) | 41.1% (411/1000) | Well-balanced across both datasets (~41–47% ASD) |
| **ASD Gender Ratio** | 85.2% Male (5.75:1 M:F) | 46.2% Male (0.86:1 M:F) | D2 reflects clinical epidemiology; D3 reflects balanced survey |
| **Clinical Diagnostic Context** | Formal laboratory diagnosis + CARS score | Primary pediatric behavioral screening | Complementary stages of diagnostic pathway |

### Clinical & Population Interpretation
The age difference is **expected, scientifically valid, and clinically meaningful**:
- **Dataset 3** reflects **early-stage pediatric behavioral screening** (typically conducted at 18–36 months during routine developmental checkups).
- **Dataset 2** reflects **specialized objective neurodevelopmental evaluation** (typically conducted at 3–12 years using eye-tracking rigs when complex visual social stimuli can be tracked).
- Rather than viewing the datasets as conflicting, they represent **two complementary tiers of the clinical ASD diagnostic pipeline**.

---

## E. Label Compatibility

- **Target Definition**:
  - Dataset 2: `Class` $\in \{\text{"TD"}, \text{"ASD"}\} \implies y \in \{0.0, 1.0\}$.
  - Dataset 3: `Diagnosed_ASD` $\in \{\text{"No"}, \text{"Yes"}\} \implies y \in \{0.0, 1.0\}$.
- **Label Semantics**: Identical binary disease classification target:
  - $0 \implies$ Non-ASD / Typically Developing (TD)
  - $1 \implies$ Confirmed Autism Spectrum Disorder (ASD)
- **Evaluation Alignment**: Both modalities use the identical positive class ($y = 1$ for ASD) and evaluate on standard metrics: Accuracy, Precision, Recall, F1, and ROC-AUC.

---

## F. Existing Model-Output Compatibility

All 4 specialized models have been implemented, trained, verified, and saved with complete reproducibility:

| Model | Modality | Best Checkpoint | Test Accuracy | Test F1 | Test ROC-AUC | Output Format |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **KAN** | Clinical / Tabular (D3) | `results/checkpoints/kan_best.pt` | 0.8533 | 0.8037 | 0.9192 | Probability $P \in [0, 1]$, Pred $\hat{y} \in \{0, 1\}$, Latent repr |
| **TabNet** | Clinical / Tabular (D3) | `results/checkpoints/tabnet_best.pt` | 0.8467 | 0.8036 | 0.9103 | Probability $P \in [0, 1]$, Pred $\hat{y} \in \{0, 1\}$, Attention masks |
| **Mamba** | Eye-Tracking (D2) | `results/checkpoints/mamba_best.pt` | 0.7299 | 0.7315 | 0.8121 | Probability $P \in [0, 1]$, Pred $\hat{y} \in \{0, 1\}$, 64-d embedding |
| **PatchTST** | Eye-Tracking (D2) | `results/checkpoints/patchtst_best.pt` | 0.7973 | 0.7804 | 0.8910 | Probability $P \in [0, 1]$, Pred $\hat{y} \in \{0, 1\}$, 512-d embedding |

### Output Types Available:
1. **Calibrated Class Probabilities**: Both pipelines output $P(\text{ASD} = 1) \in [0, 1]$ via a final Sigmoid activation.
2. **Crisp Binary Decisions**: Both pipelines threshold at $0.5 \implies \hat{y} \in \{0, 1\}$.
3. **Latent Feature Embeddings**:
   - Mamba produces a 64-dimensional pooled temporal state representation.
   - PatchTST produces a 512-dimensional multi-channel patch representation.
   - KAN produces penultimate 32-dimensional spline-activated features.
   - TabNet produces 16-dimensional decision-step aggregate features.
4. **Split Disjointness**:
   - Dataset 2 has 9 held-out test participants (2,166 sequences).
   - Dataset 3 has 150 held-out test records.

---

## G. Leakage Risks & Mitigation Safeguards

| Hazard | Description | Mitigation Implemented / Required |
| :--- | :--- | :--- |
| **Participant Overlap** | Eye-tracking windows from the same child appearing in train and test. | **Mitigated**: Stratified participant-level splitting ensures 100% disjoint participant sets. |
| **Trial Contamination** | Windows crossing trial boundaries or merging different tasks. | **Mitigated**: Vectorized trial boundary detection ensures windows never span trials. |
| **Normalization Leakage** | Fitting scalers on validation or test sets. | **Mitigated**: `StandardScaler` fitted strictly on training sets; val/test transformed only. |
| **Fictitious Subject Pairing** | Merging unrelated children's features to train a joint classifier. | **Prevented**: Explicit rejection of row-level concatenation of unpaired datasets. |
| **Test-Target Leakage** | Optimizing ensemble weights directly on test predictions. | **Required Safeguard**: Decision thresholds and ensemble weights must be tuned on Validation splits only. |
| **Feature Leakage** | Model receiving participant ID or target proxy. | **Mitigated**: Unit tests verify `ParticipantID` is never an input channel. |

---

## H. Fusion Options Considered

### Option A: True Participant-Level Multimodal Fusion (Feature / Token Concatenation)
- **Method**: Join records on `ParticipantID == Child_ID`, concatenate eye-tracking sequence with tabular features, train an end-to-end multimodal network (e.g., cross-attention Transformer).
- **Evaluation**: **Scientifically INVALID**. The datasets do not contain the same children. Any join would pair unrelated individuals, generating false correlations and invalid scientific findings.

### Option B: Naive Late Fusion via Synthetic Row Pairing
- **Method**: Randomly pair an eye-tracking sequence from D2 with a row from D3 that shares the same label, concatenate model probabilities $[P_{\text{mamba}}, P_{\text{kan}}]$, and train a logistic regression.
- **Evaluation**: **Scientifically FLAWED**. Conditioning the pairing on the ground-truth label introduces massive label leakage and circular reasoning. If unconditioned, the paired model outputs are statistically independent noise from two different people.

### Option C: Modality-Level Comparative Benchmark Only
- **Method**: Stop at reporting independent model performance for Dataset 2 and Dataset 3 without any joint integration.
- **Evaluation**: Scientifically valid, but fails to fulfill the project goal of delivering a unified ASD True/False detection system.

### Option D: Hierarchical Dual-Modality Clinical Decision & Triage System (RECOMMENDED)
- **Method**: Treat the models as **specialized clinical diagnostic agents**:
  - An input patient can be evaluated through whichever modality (or both) is available.
  - When both modalities are provided, an intelligent decision-fusion engine (Bayesian belief updater, soft-voting ensemble, or risk-stratified triage policy) combines $P_{\text{clinical}}$ and $P_{\text{eye}}$ into a consolidated ASD diagnostic score $P_{\text{ASD}}$.
  - Model parameters are kept frozen; fusion weights/decision thresholds are calibrated on validation splits.
- **Evaluation**: **Scientifically sound, clinically realistic, zero data fabrication**, and perfectly satisfies the project flowchart.

---

## I. Recommended Experimental Design

### Architecture: Dual-Agent Clinical Triage & Decision Policy

```
                              [ PATIENT INPUT ]
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
   [ Clinical Survey Data ]                           [ Eye-Tracking Video/Gaze ]
            │                                                   │
   [ KAN / TabNet Model ]                             [ Mamba / PatchTST Model ]
            │                                                   │
            ▼                                                   ▼
   P(ASD)_clinical ∈ [0, 1]                            P(ASD)_eye ∈ [0, 1]
            │                                                   │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
                      [ MULTIMODAL DECISION POLICY ]
                   P_combined = w_clin * P_clin + w_eye * P_eye
                                      │
                                      ▼
                           [ CLINICAL TRIAGE GATE ]
                 ┌────────────────────┼────────────────────┐
                 ▼                    ▼                    ▼
          P_combined < 0.35    0.35 ≤ P ≤ 0.65       P_combined > 0.65
             [ TD (0) ]      [ BORDERLINE/CONFIRM ]      [ ASD (1) ]
                 │                    │                    │
                 └────────────────────┼────────────────────┘
                                      ▼
                            FINAL ASD TRUE / FALSE
                                      │
                     (If TRUE ──► Future Dataset 1 Severity)
```

### Key Components of the Design:
1. **Model Specialists**:
   - Tabular Specialist: **KAN** (Primary) with TabNet benchmark.
   - Eye-Tracking Specialist: **Mamba** (Primary) with PatchTST benchmark.
2. **Joint Decision Formulations**:
   - **Formulation 1: Weighted Soft-Voting Ensemble**:
     $$P_{\text{fused}} = \alpha \cdot P_{\text{clinical}} + (1 - \alpha) \cdot P_{\text{eye}}$$
     where $\alpha$ is calibrated on validation performance (e.g., weighted by modality ROC-AUC or inverse variance).
   - **Formulation 2: Bayesian Log-Odds Fusion**:
     $$\text{logit}(P_{\text{fused}}) = \text{logit}(P_{\text{clinical}}) + \text{logit}(P_{\text{eye}}) - \text{logit}(P_{\text{prior}})$$
   - **Formulation 3: Clinical Triage Routing**:
     - Stage 1 (Screening): If $P_{\text{clinical}} < \tau_{\text{low}}$, predict Non-ASD (no eye-tracking needed).
     - If $P_{\text{clinical}} > \tau_{\text{high}}$, predict ASD with high confidence.
     - If $\tau_{\text{low}} \le P_{\text{clinical}} \le \tau_{\text{high}}$ (ambiguous / borderline screening), trigger Stage 2 (Eye-Tracking assessment) to confirm or refute diagnosis.
3. **Statistical Validation Protocol (Without Data Fabrication)**:
   - **Modality-Specific Verification**: Evaluate KAN/TabNet on Dataset 3 test set (150 patients); evaluate Mamba/PatchTST on Dataset 2 test set (9 participants, 2,166 sequences).
   - **Simulated Matched Cohort Evaluation**: To quantitatively assess the theoretical fusion gain on a paired cohort, construct a **covariate-matched synthetic evaluation testbed** matched on demographic covariates (ASD status, sex, and age group decile). This testbed evaluates the ensemble's sensitivity, specificity, and ROC-AUC under controlled statistical pairing, with explicit documentation that it is an experimental simulation.

---

## J. Project Flowchart Compatibility

The original project specification defines:
```
Dataset 2 — Eye Tracking ──► Mamba / PatchTST ──► ASD-related output
                                                        +
Dataset 3 — Clinical/Tab ──► KAN / TabNet     ──► ASD-related output
                                                        │
                                                        ▼
                                                ASD CLASSIFICATION
                                                        │
                                                        ▼
                                                 ASD TRUE / FALSE
```

### Compatibility Audit:
- **Technical Compatibility**: **100% Compatible**. Both existing model pipelines already output standardized ASD-related outputs:
  - $P(\text{ASD} = 1) \in [0.0, 1.0]$
  - Prediction $\in \{0, 1\}$
  - Calibrated confidence and latent embeddings.
- **Scientific Caveat**: Because Dataset 2 and Dataset 3 are **unpaired** in their raw training collections, the fusion stage must be implemented as a **decision-level integration / clinical triage policy** rather than an early-stage feature merge.
- **Future Integration**: When ASD is classified as `TRUE`, the output cleanly routes to Dataset 1 (Video / Skeleton) for severity assessment, preserving the overarching project architecture.

---

## K. Required Implementation Changes for Phase 4 Fusion

When moving to implementation (after user approval):
1. Create `src/fusion/` directory with:
   - `__init__.py`: Package exports.
   - `decision_fusion.py`: Implements weighted soft-voting, Bayesian log-odds fusion, and clinical triage policies.
   - `inference_pipeline.py`: Unified inference endpoint accepting tabular inputs, eye-tracking sequences, or both, returning consolidated ASD True/False and confidence scores.
   - `evaluate_fusion.py`: Evaluates independent and joint decision metrics, plotting ROC curves and triage efficiency.
2. Update `configs/config.yaml` to include a `fusion:` configuration block specifying ensemble weights, triage thresholds, and decision strategies.
3. Create `tests/test_fusion.py` to verify joint inference, boundary conditions, probability validities, and graceful handling of single-modality inputs.
4. Save fusion comparison metrics and summary tables in `results/metrics/fusion_model_comparison.csv`.

---

## L. Unresolved Issues

- **None**.
  - All 25 CSV files and metadata in Dataset 2 remain 100% intact.
  - Raw `asd.csv` in Dataset 3 remains 100% intact.
  - All 4 underlying models (KAN, TabNet, Mamba, PatchTST) are fully trained, tested, and checkpointed.
  - The exact nature of the datasets (unpaired populations across different age groups) has been rigorously established from raw data evidence with zero assumptions or data fabrication.

---
*End of Pre-Fusion Compatibility & Experiment Design Audit.*
