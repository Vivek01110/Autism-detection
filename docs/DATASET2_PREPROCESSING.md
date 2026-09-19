# Dataset 2 (Eye Tracking) — Preprocessing & Sequence Construction Specification

## 1. Overview & Architecture Role

Dataset 2 comprises raw high-frequency eye-tracking recordings from **57 pediatric participants** viewing visual stimuli. This dataset serves as the temporal multi-channel input for:
- **Primary Model**: Mamba (State Space Model)
- **Benchmark Model**: PatchTST (Patch Time Series Transformer)

---

## 2. Raw Data Structure & Integrity Rules

- **Location**: `data/dataset_2_eye_tracking/archive-2/Eye-tracking Output/`
- **Metadata**: `data/dataset_2_eye_tracking/archive-2/Metadata_Participants.csv`
- **Total Files**: 25 CSV files (`1.csv` through `25.csv`) + 1 Metadata CSV
- **Integrity Rule**: All raw data files are treated as strictly **READ-ONLY** and untouched. No raw file was edited, deleted, renamed, or overwritten.

---

## 3. Participant Structure & Mapping

### 3.1 Non-Equivalence of Files and Participants
- A single CSV file does **NOT** equal a single participant. Each file corresponds to an experimental recording session containing **multiple participants** (ranging from 3 to 39 participants per session).
- A single participant appears across multiple recording sessions and trials.
- Therefore, participant identity is determined strictly by the `Participant` column, never by filename.

### 3.2 Participant Filtering & Metadata Mapping
- **Total Metadata Participants**: 59 (`ParticipantID` 1 to 59).
- **Usable Participants**: **57 participants**.
  - `ParticipantID` 12 (ASD) and `ParticipantID` 16 (ASD) have **0 rows of eye-tracking data** across all 25 files and are excluded.
  - Non-numeric entries (`Unidentified(Neg)`, `Unidentified(Pos)`) are excluded.
- **Diagnosis Label Mapping**:
  - Class `ASD` $\rightarrow$ `1.0` (27 usable participants: IDs 1–29 excluding 12, 16)
  - Class `TD` (Typically Developing) $\rightarrow$ `0.0` (30 usable participants: IDs 30–59)

---

## 4. Schema Variations & Feature Selection

### 4.1 Schema Variation Across the 25 Files
- Column counts vary from 37 to 58 columns across files.
- Common core columns exist across all files: `Participant`, `Trial`, `RecordingTime [ms]`, `Category Group`, `Category Right`, `Point of Regard Right X/Y [px]`.
- Files `8.csv` and `13.csv` are **monocular (Right eye only)**, lacking Left eye channels.
- Missing values in SMI eye-tracking outputs are encoded as string `'-'` (hyphen), which is mapped to `NaN`.

### 4.2 Gaze Feature Validity Assessment
- Inspection confirmed that `Gaze Vector Right/Left X/Y/Z` columns are **100% all zeros** in files `1.csv`, `2.csv`, `13.csv`, `25.csv` and several others.
- Consequently, 3D `Gaze Vector` is **invalid/unreliable** across the recording corpus.
- In contrast, screen-plane `Point of Regard` (gaze screen coordinates in pixels) is 100% populated and calibrated across all 25 files.
- Therefore, `Point of Regard` is designated as the reliable gaze measurement.

### 4.3 Selected 8 Feature Channels

| Channel Index | Feature Name | Description | Missing / Monocular Handling |
| :---: | :--- | :--- | :--- |
| **0** | `Point of Regard Right X [px]` | Right eye gaze horizontal coordinate | Linear interpolation $\rightarrow$ forward/back fill |
| **1** | `Point of Regard Right Y [px]` | Right eye gaze vertical coordinate | Linear interpolation $\rightarrow$ forward/back fill |
| **2** | `Point of Regard Left X [px]` | Left eye gaze horizontal coordinate | Mirrored from Right eye in monocular files |
| **3** | `Point of Regard Left Y [px]` | Left eye gaze vertical coordinate | Mirrored from Right eye in monocular files |
| **4** | `Pupil Diameter Right [mm]` | Right pupil diameter in mm | Imputed via linear interpolation + train median |
| **5** | `Pupil Diameter Left [mm]` | Left pupil diameter in mm | Mirrored from Right or imputed |
| **6** | `Gaze_Velocity` | $\sqrt{\Delta x^2 + \Delta y^2} / \Delta t$ [px/ms] | Derived instantaneous eye motion dynamics |
| **7** | `Fixation_Flag` | Eye event classification | Binary: 1.0 if `Category Right == 'Fixation'`, 0.0 otherwise |

---

## 5. Temporal Sequences & Windowing Strategy

1. **Filtering**: Rows with `Category Group != 'Eye'` (e.g., separator and information lines) are removed.
2. **Temporal Ordering**: Within each participant's trial, rows are strictly sorted by `RecordingTime [ms]`.
3. **Window Length**: `sequence_length = 200` timesteps (~4.8 to 5.0 seconds of continuous gaze tracking at SMI ~41 Hz).
4. **Stride**: `stride = 100` timesteps (50% overlap for data augmentation and temporal continuity).
5. **Boundary Constraint**: **Windows NEVER cross trial boundaries and NEVER cross participant boundaries.** Slicing is performed strictly per (participant, trial).
6. **Padding**: Trials between 50 and 199 samples are edge-padded to 200. Trials with fewer than 50 samples are discarded.

---

## 6. Train / Validation / Test Splitting Strategy

### 6.1 Strict Participant-Level Split
To prevent data leakage, splitting is conducted at the **participant level** with `random_seed = 42` before any feature scaling or sequence generation:

$$\text{Train Participants} \cap \text{Validation Participants} = \emptyset$$
$$\text{Train Participants} \cap \text{Test Participants} = \emptyset$$
$$\text{Validation Participants} \cap \text{Test Participants} = \emptyset$$

### 6.2 Split Distribution
- **Total Usable Participants**: 57
  - **Train (70%)**: **39 participants** (19 ASD, 20 TD — 48.7% ASD)
  - **Validation (15%)**: **9 participants** (4 ASD, 5 TD — 44.4% ASD)
  - **Test (15%)**: **9 participants** (4 ASD, 5 TD — 44.4% ASD)

### 6.3 Data Leakage Prevention
- `StandardScaler` is fitted **ONLY on training participants' sequence timesteps**.
- Validation and test sequences are transformed strictly using the training-fitted scaler parameters.
- Scaler artifact is saved to `results/preprocessing_artifacts/dataset2_scaler.joblib`.

---

## 7. Resulting Tensor Shapes & PyTorch Dataset

- **Sequence Tensor Shape**: `(batch_size, sequence_length, num_features)` $\rightarrow$ `(B, 200, 8)`
- **Label Shape**: `(batch_size,)` (binary `0.0` or `1.0`)
- **Participant ID Shape**: `(batch_size,)` (int, available for audit and evaluation, excluded from model inputs)
- **PyTorch Classes**:
  - `EyeTrackingDataset`: Custom PyTorch `Dataset` yielding `(sequence, label, participant_id)`
  - `Dataset2Preprocessor.get_dataloader(split, batch_size)`: Standard PyTorch `DataLoader`
- **Cached Dataset**: Persisted to `data/processed/dataset_2/dataset2_processed.pt` for sub-second ingestion.
