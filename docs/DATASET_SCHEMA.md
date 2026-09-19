# Dataset Schema — Multi-Modal ASD Detection Project

> **Document generated from exhaustive inspection of all raw data files.**
> No columns were assumed. No labels were invented. No data was modified.

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Dataset 3 — Clinical / Tabular (`asd.csv`)](#2-dataset-3--clinical--tabular-asdcsv)
3. [Dataset 2 — Eye Tracking (25 CSV files + Metadata)](#3-dataset-2--eye-tracking-25-csv-files--metadata)
4. [Temporal Sequence Conversion Strategy for Mamba / PatchTST](#4-temporal-sequence-conversion-strategy-for-mamba--patchtst)
5. [Train / Validation / Test Splitting Strategy](#5-train--validation--test-splitting-strategy)
6. [Critical Observations & Warnings](#6-critical-observations--warnings)

---

## 1. Repository Structure

```
Autism-detection/
├── data/
│   ├── dataset_1_MMASD/                   # (out of scope for now)
│   ├── dataset_2_eye_tracking/
│   │   └── archive-2/
│   │       ├── Eye-tracking Output/
│   │       │   ├── 1.csv  ... 25.csv       # 25 eye-tracking recording files
│   │       └── Metadata_Participants.csv    # Participant demographics + labels
│   ├── dataset_3_clinical_tabular/
│   │   └── asd.csv                          # 1,000-row clinical survey
│   └── processed/                           # (empty — not yet generated)
├── src/
│   └── models/
│       ├── dataset_2_mamba/                 # (empty — to be implemented)
│       ├── dataset_2_patchtst/              # (empty — to be implemented)
│       ├── dataset_3_kan/                   # (empty — to be implemented)
│       └── dataset_3_tabnet/                # (empty — to be implemented)
├── configs/                                 # (empty)
├── docs/                                    # This schema document
├── notebooks/                               # (empty)
├── results/                                 # (empty)
└── tests/                                   # (empty)
```

---

## 2. Dataset 3 — Clinical / Tabular (`asd.csv`)

**Path:** `data/dataset_3_clinical_tabular/asd.csv`
**Size:** ~68 KB

### 2.1 Dimensions

| Metric | Value |
|--------|-------|
| Rows | **1,000** |
| Columns | **11** (10 meaningful + 1 artifact) |

### 2.2 Exact Column Names (in order)

| # | Column Name | Data Type | Role |
|---|-------------|-----------|------|
| 1 | `Child_ID` | `object` (UUID string) | Identifier — **exclude from features** |
| 2 | `Age` | `int64` | Numerical feature |
| 3 | `Gender` | `object` | Categorical feature |
| 4 | `Jaundice` | `object` | Categorical feature |
| 5 | `Family_ASD_History` | `object` | Categorical feature |
| 6 | `Language_Delay` | `object` | Categorical feature |
| 7 | `Social_Interaction_Score` | `int64` | Numerical feature |
| 8 | `Communication_Score` | `int64` | Numerical feature |
| 9 | `Repetitive_Behavior_Score` | `int64` | Numerical feature |
| 10 | `Diagnosed_ASD` | `object` | **TARGET / LABEL** |
| 11 | `Unnamed: 10` | `float64` | Artifact (100% NaN) — **drop this column** |

### 2.3 Missing Values

| Column | Missing Count | % Missing |
|--------|--------------|-----------|
| `Child_ID` | 0 | 0% |
| `Age` | 0 | 0% |
| `Gender` | 0 | 0% |
| `Jaundice` | 0 | 0% |
| `Family_ASD_History` | 0 | 0% |
| `Language_Delay` | 0 | 0% |
| `Social_Interaction_Score` | 0 | 0% |
| `Communication_Score` | 0 | 0% |
| `Repetitive_Behavior_Score` | 0 | 0% |
| `Diagnosed_ASD` | 0 | 0% |
| `Unnamed: 10` | **1,000** | **100%** |

> [!NOTE]
> Zero missing values in all 10 meaningful columns. `Unnamed: 10` is a CSV parsing artifact from trailing commas and should be dropped.

### 2.4 Target Column: `Diagnosed_ASD`

| Class | Count | Percentage |
|-------|-------|------------|
| `No` (non-ASD) | 589 | 58.9% |
| `Yes` (ASD) | 411 | 41.1% |
| **Total** | **1,000** | 100% |

**Relatively balanced** — no extreme class imbalance.

### 2.5 Numerical Features — Statistics

| Statistic | `Age` | `Social_Interaction_Score` | `Communication_Score` | `Repetitive_Behavior_Score` |
|-----------|-------|---------------------------|----------------------|---------------------------|
| Min | 12 | 1 | 1 | 1 |
| Max | 36 | 10 | 10 | 10 |
| Mean | 23.99 | 5.50 | 5.47 | 5.19 |
| Std | 7.44 | 2.80 | 2.87 | 2.65 |
| Median | 24 | 6 | 5 | 5 |
| Unique | 25 | 10 | 10 | 10 |

- `Age` ranges 12–36 (likely months, representing toddler age).
- Scores are on a 1–10 integer scale.

### 2.6 Categorical Features — Value Counts

| Column | Values | Distribution |
|--------|--------|-------------|
| `Gender` | `Female`: 541 (54.1%), `Male`: 459 (45.9%) | ~balanced |
| `Jaundice` | `Yes`: 520 (52.0%), `No`: 480 (48.0%) | ~balanced |
| `Family_ASD_History` | `No`: 650 (65.0%), `Yes`: 350 (35.0%) | slight skew |
| `Language_Delay` | `No`: 609 (60.9%), `Yes`: 391 (39.1%) | slight skew |

### 2.7 Feature Summary for Modeling

**For KAN (primary) and TabNet (benchmark):**

| Feature Category | Columns | Preprocessing |
|-----------------|---------|--------------|
| **Numerical** (4) | `Age`, `Social_Interaction_Score`, `Communication_Score`, `Repetitive_Behavior_Score` | Standardize / normalize |
| **Binary Categorical** (4) | `Gender`, `Jaundice`, `Family_ASD_History`, `Language_Delay` | Encode: Yes=1/No=0, Male=1/Female=0 |
| **Target** (1) | `Diagnosed_ASD` | Encode: Yes=1, No=0 |
| **Drop** (2) | `Child_ID`, `Unnamed: 10` | Identifier + artifact |

**Total usable features: 8** (4 numerical + 4 binary categorical)

### 2.8 Sample Rows

```
Child_ID                              Age Gender Jaundice Family_ASD_History Language_Delay Social_Interaction_Score Communication_Score Repetitive_Behavior_Score Diagnosed_ASD
3eac2cb7-6348-4522-8049-0901c99a1693   18   Male       No                 No             No                        5                   4                         8            No
7c9b5683-0bf6-49c4-948b-cc6e5afd8000   31 Female      Yes                 No             No                        7                   6                         4            No
aafa5ad4-f77a-45c8-b9f2-310392773362   26   Male      Yes                Yes             No                        4                   5                         5           Yes
```

---

## 3. Dataset 2 — Eye Tracking (25 CSV files + Metadata)

### 3.1 Metadata File: `Metadata_Participants.csv`

**Path:** `data/dataset_2_eye_tracking/archive-2/Metadata_Participants.csv`
**Size:** ~1 KB

#### 3.1.1 Dimensions

| Metric | Value |
|--------|-------|
| Rows | **59** |
| Columns | **5** |

#### 3.1.2 Column Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `ParticipantID` | `int64` | Sequential integer 1–59 |
| `Gender` | `object` | `'M'` or `'F'` |
| `Age` | `float64` | Child's age in years (range: 2.7–12.9) |
| `Class` | `object` | **ASD label**: `'ASD'` or `'TD'` (Typically Developing) |
| `CARS Score` | `float64` | Childhood Autism Rating Scale score (ASD only, range: 17.0–45.0) |

#### 3.1.3 Class Distribution

| Class | Count | Percentage | ParticipantID Range |
|-------|-------|------------|-------------------|
| **ASD** | 29 | 49.15% | 1–29 |
| **TD** | 30 | 50.85% | 30–59 |
| **Total** | **59** | 100% | |

#### 3.1.4 Missing Values

| Column | Missing | Note |
|--------|---------|------|
| `ParticipantID` | 0 | — |
| `Gender` | 0 | — |
| `Age` | 0 | — |
| `Class` | 0 | — |
| `CARS Score` | **30** (50.85%) | All 30 TD participants have NaN (expected — CARS is only scored for ASD) |

#### 3.1.5 Gender Breakdown

| Group | Male | Female |
|-------|------|--------|
| ASD (n=29) | 25 | 4 |
| TD (n=30) | 13 | 17 |
| Total (n=59) | 38 | 21 |

> [!IMPORTANT]
> Strong gender imbalance within the ASD group (86% male). This reflects real-world ASD prevalence patterns.

---

### 3.2 Eye-Tracking Data Files: `1.csv` through `25.csv`

**Path:** `data/dataset_2_eye_tracking/archive-2/Eye-tracking Output/`
**Total size:** ~600 MB
**Total data rows:** **2,252,943** (across all 25 files)

#### 3.2.1 File-Level Summary

| File | Rows | Columns | # Participants | # Trials | Sample Stimulus |
|------|------|---------|---------------|----------|-----------------|
| 1.csv | 34,630 | 37 | 18 | 1 | eye tracking (ballon droite).avi |
| 2.csv | 16,687 | 51 | 7 | 1 | Eye Tracking (ballon droite).avi |
| 3.csv | 124,658 | 52 | 14 | 1 | Eye Tracking (ballon gauche).avi |
| 4.csv | 23,136 | 51 | 12 | 1 | eye tracking (ballon gauche).avi |
| 5.csv | 17,769 | 51 | 9 | 1 | Eye Tracking (ballon gauche).avi |
| 6.csv | 62,648 | 37 | 26 | 18 | coucou d.jpg, devant.jpg, ... |
| 7.csv | 61,583 | 58 | 10 | 1 | VNVD151207.avi |
| 8.csv | 114,285 | 39 | 14 | 1 | VNVD151207.avi |
| 9.csv | 74,479 | 51 | 25 | 1 | FEDE Drte.avi |
| 10.csv | 51,829 | 37 | 26 | 1 | 01vnvg151201b1.avi |
| 11.csv | 114,285 | 52 | 14 | 1 | VNVD151207.avi |
| 12.csv | 73,440 | 51 | 24 | 1 | Federica Final_WMV_3000Kbps_720p.avi |
| 13.csv | 21,947 | 38 | 3 | 14 | neutre2.avi, bonbons triste vs joie.avi, ... |
| 14.csv | 28,374 | 51 | 5 | 14 | neutre4.avi, neutre visage gris.jpg, ... |
| 15.csv | 29,699 | 51 | 3 | 14 | neutre3.avi, vole triste vs joie1.avi, ... |
| 16.csv | 110,967 | 37 | 18 | 15 | eye tracking (ballon droite).avi, neutre5.avi, ... |
| 17.csv | 396,298 | 52 | 14 | 14 | neutre22.avi, Triste joie.jpg, ... |
| 18.csv | 70,797 | 51 | 25 | 18 | yeux chat D.jpg, tete chat droite.jpg, ... |
| 19.csv | 71,234 | 51 | 24 | 18 | coucou D.jpg, yeux chat gauche.jpg, ... |
| 20.csv | 123,702 | 52 | 14 | 17 | coucou D.png, yeux chat G.png, ... |
| 21.csv | 284,931 | 52 | 35 | 34 | coucou D.png, yeux chat G.png, ... |
| 22.csv | 56,929 | 51 | 26 | 17 | coucou g.jpg, devant.jpg, ... |
| 23.csv | 130,909 | 52 | 14 | 1 | Eye Tracking (ballon droite).avi |
| 24.csv | 38,663 | 51 | 21 | 1 | fede invisible d avi mpeg4-pcm.avi |
| 25.csv | 119,064 | **50** | 36 | 1 | 01vnvg151201b1.avi, VNVG151201b.avi |

> [!WARNING]
> **Each CSV file is NOT one participant.** Files represent **recording sessions/experiments** containing data for **multiple participants** viewing a specific set of stimuli (trials). Files are organized by **experiment/stimulus type**, not by participant.

#### 3.2.2 Column Schema Variations

The 25 files have **6 different column schemas** (37, 38, 39, 50, 51, 52, or 58 columns). All files share a **common core** of columns, with variations in optional columns.

##### Common Core Columns (present in ALL files)

| Category | Column Name | Description |
|----------|------------|-------------|
| **Index** | *(unnamed, col 0)* | Row index within the file |
| **Temporal** | `RecordingTime [ms]` | Absolute recording timestamp in milliseconds |
| **Temporal** | `Time of Day [h:m:s:ms]` | Wall-clock time of recording |
| **Trial** | `Trial` | Trial identifier (e.g., `Trial020`, `Trial001`) |
| **Trial** | `Stimulus` | Stimulus file being shown (e.g., `eye tracking (ballon droite).avi`) |
| **Trial** | `Export Start Trial Time [ms]` | Trial start time |
| **Trial** | `Export End Trial Time [ms]` | Trial end time |
| **Participant** | `Participant` | Participant ID (numeric or `Unidentified(Neg)`/`Unidentified(Pos)`) |
| **Metadata** | `Color` | Display color code for the participant |
| **Metadata** | `Tracking Ratio [%]` | Overall eye-tracking quality ratio |
| **Eye Event** | `Category Group` | `Eye` or `Information` |
| **Eye Event** | `Category Right` | Eye event type: `Fixation`, `Saccade`, `Blink`, `Separator`, `-` |
| **Eye Event** | `Index Right` | Event index for right eye |
| **Pupil** | `Pupil Diameter Right [mm]` | Right pupil diameter |
| **Gaze** | `Point of Regard Right X [px]` | Right eye gaze X coordinate (pixels) |
| **Gaze** | `Point of Regard Right Y [px]` | Right eye gaze Y coordinate (pixels) |
| **Gaze** | `Gaze Vector Right X` | Right eye gaze vector X component |
| **Gaze** | `Gaze Vector Right Y` | Right eye gaze vector Y component |
| **Gaze** | `Gaze Vector Right Z` | Right eye gaze vector Z component |
| **AOI** | `AOI Name Right` | Area of Interest the right eye is looking at |
| **Annotation** | `Annotation Name` | Manual annotation marker |
| **Annotation** | `Annotation Description` | Annotation details |
| **Annotation** | `Annotation Tags` | Annotation tags |
| **Mouse** | `Mouse Position X [px]` | Mouse X position |
| **Mouse** | `Mouse Position Y [px]` | Mouse Y position |
| **Scroll** | `Scroll Direction X` | Horizontal scroll |
| **Scroll** | `Scroll Direction Y` | Vertical scroll |
| **Content** | `Content` | Currently displayed content |

##### Optional Left-Eye Columns (present in most files, absent from monocular files 8, 13)

| Column Name | Description |
|-------------|-------------|
| `Category Left` | Eye event type for left eye |
| `Index Left` | Event index for left eye |
| `Pupil Diameter Left [mm]` | Left pupil diameter |
| `Point of Regard Left X [px]` | Left eye gaze X coordinate |
| `Point of Regard Left Y [px]` | Left eye gaze Y coordinate |
| `AOI Name Left` | Area of Interest for left eye |
| `Gaze Vector Left X/Y/Z` | Left eye gaze vector |

##### Optional Extended Columns (present in some files)

| Column Name | Files Present |
|-------------|--------------|
| `Pupil Size Right X [px]` | 2, 3, 4, 5, 7, 9, 11, 12, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25 |
| `Pupil Size Right Y [px]` | Same as above |
| `Pupil Size Left X [px]` | Same as above (binocular files only) |
| `Pupil Size Left Y [px]` | Same as above (binocular files only) |
| `Eye Position Right X/Y/Z [mm]` | 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25 |
| `Eye Position Left X/Y/Z [mm]` | Binocular files with Eye Position |
| `Pupil Position Right X/Y [px]` | Same as Eye Position files |
| `Pupil Position Left X/Y [px]` | Binocular files with Pupil Position |
| `AOI Group Right/Left` | 3, 7, 8, 13, 17, 20, 21, 23 |
| `AOI Scope Right/Left` | Same as AOI Group |
| `AOI Order Right` | 7, 8, 13 |
| `AOI Order Binocular` | 7 |
| `Port Status` | 3, 7, 8, 13 |
| `groupe d'enfants` | **25 only** (French: "group of children", value: `TC`) |

#### 3.2.3 File-to-Participant Mapping

> [!CAUTION]
> **The filename number (1–25) does NOT correspond to ParticipantID.** Each file contains data for multiple participants from the same recording session. The `Participant` column within each CSV identifies which participant generated each row.

**Participant ID values found in data:**
- **Numeric IDs:** 1–11, 13–15, 17–59 (57 out of 59 metadata participants)
- **Non-numeric IDs:** `Unidentified(Neg)`, `Unidentified(Pos)` — rows where the eye tracker could not identify the participant
- **Missing participants:** **ParticipantID 12 and 16** have no eye-tracking data in any of the 25 files

**Participant-to-Class mapping (via Metadata):**
- IDs 1–29 → **ASD** (29 participants, IDs 12 and 16 have no eye-tracking data → **27 usable ASD**)
- IDs 30–59 → **TD** (30 participants, all present → **30 usable TD**)
- **Total usable participants: 57** (27 ASD + 30 TD)

#### 3.2.4 Data Characteristics

| Property | Value |
|----------|-------|
| Approximate sampling rate | ~41 Hz (≈24 ms between samples) |
| Time representation | Absolute `RecordingTime [ms]` |
| Eye event types | `Fixation`, `Saccade`, `Blink`, `Separator` (in `Category Right/Left`) |
| Category groups | `Eye` (actual gaze data), `Information` (metadata/separator rows) |
| AOI values (example) | `corps` (body), `BallonInvisible`, `-` (no AOI hit) |
| Pupil diameter unit | millimeters (mm) |
| Gaze coordinates unit | pixels (px) |
| Missing gaze/pupil values | Represented as `'-'` (string dash, not NaN) |
| Annotation/Mouse/Scroll cols | 100% dash (`'-'`) in most files — effectively empty |
| AOI Name missing rate | ~41% dash values (when gaze is outside defined AOIs) |

> [!IMPORTANT]
> **Missing value encoding:** The SMI eye-tracker uses the string `'-'` (not `NaN`) to encode missing/invalid data. When loading with pandas, use `na_values=['-']` to properly parse these as `NaN`. A naive `df.isna().sum()` will report **zero** missing values without this.
>
> Example missing rates from `1.csv` (34,630 rows):
> - `Pupil Diameter Right/Left [mm]`: 21 rows (0.06%) — during blinks
> - `Point of Regard Right/Left X/Y [px]`: 21 rows (0.06%)
> - `Category Right/Left`: 1,693 / 1,631 rows (~5%) — Information/separator rows
> - `AOI Name Right/Left`: 14,380 rows (41.5%) — gaze outside defined AOIs
> - `Annotation/Mouse/Scroll` columns: 34,630 rows (100%) — unpopulated

#### 3.2.5 How Each CSV Represents a Temporal Sequence

Each CSV file contains **time-series eye-tracking data** from one or more recording sessions:

1. **Each row** = one eye-tracking sample at a specific `RecordingTime [ms]`
2. **Within a file**, data is organized by `Participant` → `Trial` → time-ordered samples
3. **A sequence for one participant** = all rows where `Participant == <ID>`, filtered by `Category Group == 'Eye'`, ordered by `RecordingTime [ms]`
4. **The same participant may appear in multiple files** (different experiments/stimuli)
5. **`Information` rows** (where `Category Group == 'Information'`) are separator/metadata rows and should be filtered out before sequence extraction

#### 3.2.6 How the 25 Files Relate to Each Other

The 25 CSV files represent **different experimental tasks/stimuli sessions**. They do NOT partition participants — instead:

- **Files 1, 4**: Balloon tracking (right/left variants), same participant groups
- **Files 2, 5**: Balloon tracking, different participant batch
- **Files 3, 23**: Balloon tracking (left/right), same participant batch
- **Files 7, 8, 11**: VNVD video stimulus, overlapping participant groups
- **Files 9, 12**: FEDE / Federica videos, similar participant groups
- **Files 10, 25**: Visual preference (vnvg), same participants
- **Files 6, 22**: Social images (coucou, devant, regard chien), similar groups
- **Files 13–15**: Emotion preference tasks (neutre, bonbons, joie), smaller groups
- **Files 16**: Mixed stimuli session, large group
- **Files 17, 20**: Emotion/preference tasks, specific group
- **Files 18, 19**: Social attention tasks, large groups
- **Files 21**: Large combined session with many participants and trials
- **File 24**: Single stimulus (fede invisible)

#### 3.2.7 How ASD/Non-ASD Labels are Obtained

1. Extract the `Participant` column value from each eye-tracking row
2. Parse it as an integer (discard `Unidentified(Neg)` / `Unidentified(Pos)` rows)
3. Join with `Metadata_Participants.csv` on `ParticipantID`
4. Use the `Class` column: `'ASD'` → positive, `'TD'` → negative

---

## 4. Temporal Sequence Conversion Strategy for Mamba / PatchTST

### 4.1 Sequence Definition

For each participant, we need to construct temporal sequences suitable for Mamba (SSM) and PatchTST (Transformer):

```
Sequence = [x_1, x_2, ..., x_T]  where x_t ∈ ℝ^d
```

### 4.2 Recommended Feature Channels (d dimensions)

Select the **common core** numerical columns present across all files:

| Channel | Column | Unit | Notes |
|---------|--------|------|-------|
| 1 | `Pupil Diameter Right [mm]` | mm | Primary pupil feature |
| 2 | `Pupil Diameter Left [mm]` | mm | When available; impute/mirror when missing |
| 3 | `Point of Regard Right X [px]` | px | Horizontal gaze position |
| 4 | `Point of Regard Right Y [px]` | px | Vertical gaze position |
| 5 | `Point of Regard Left X [px]` | px | When available |
| 6 | `Point of Regard Left Y [px]` | px | When available |
| 7 | `Gaze Vector Right X` | unitless | 3D gaze direction |
| 8 | `Gaze Vector Right Y` | unitless | |
| 9 | `Gaze Vector Right Z` | unitless | |
| 10 | `Gaze Vector Left X` | unitless | When available |
| 11 | `Gaze Vector Left Y` | unitless | |
| 12 | `Gaze Vector Left Z` | unitless | |

**Derived features** (compute during preprocessing):
| Channel | Feature | Formula |
|---------|---------|---------|
| 13 | Gaze velocity | $\sqrt{(\Delta x)^2 + (\Delta y)^2} / \Delta t$ |
| 14 | Pupil dilation rate | $\Delta \text{diameter} / \Delta t$ |
| 15 | Binocular disparity | $\lvert \text{PoR\_Right} - \text{PoR\_Left} \rvert$ |

### 4.3 Preprocessing Pipeline

```
For each participant P (with valid ID in metadata):
  1. Collect ALL rows across all 25 files where Participant == P
  2. Filter: Keep only rows where Category Group == 'Eye'
  3. Filter: Remove rows where Category Right == 'Separator'
  4. Sort by RecordingTime [ms]
  5. Group by (Trial, Stimulus) to get per-trial sequences
  6. Handle missing values:
     - Replace '-' strings with NaN
     - For monocular files: use right-eye data only, or mirror R→L
     - Interpolate short gaps (< 75ms / ~3 samples)
     - Mark longer gaps as invalid
  7. Normalize numeric features (z-score per participant or global)
  8. Segment into fixed-length windows OR pad/truncate to max_len
```

### 4.4 Sequence Granularity Options

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Per-participant** | Concatenate all trials → one long sequence per participant | Maximizes data per participant | Very unequal lengths, mixed stimuli |
| **Per-trial** (recommended) | One sequence per (participant, trial) pair | Natural segmentation, comparable lengths | Multiple sequences per participant — need aggregation |
| **Fixed-window** | Sliding windows of N samples across concatenated data | Uniform input size | Window boundaries are arbitrary |

**Recommended approach:** Per-trial sequences, then aggregate trial-level predictions to participant-level using majority vote or mean probability.

### 4.5 Handling Schema Variations

Since files have different column sets:

1. **Define a canonical feature set** using the 12 core channels above
2. **For monocular files** (8, 13): Set left-eye channels to NaN → handle via:
   - Option A: Mirror right-eye values
   - Option B: Use a mask indicating available channels
   - Option C: Use right-eye only (6 channels) with a separate model head
3. **Ignore non-core columns** (AOI Group/Scope/Order, Port Status, Mouse, Scroll, Annotations)

### 4.6 Input Format for Mamba vs PatchTST

| Model | Expected Input | Notes |
|-------|---------------|-------|
| **Mamba** | `(batch, seq_len, d_model)` | Processes sequences causally; handles variable lengths well |
| **PatchTST** | `(batch, n_channels, seq_len)` → patched | Divides each channel into patches; needs fixed seq_len |

---

## 5. Train / Validation / Test Splitting Strategy

### 5.1 Dataset 3 (Clinical/Tabular)

**Simple stratified split** — each row is an independent patient:

| Split | Ratio | ~Rows | Strategy |
|-------|-------|-------|----------|
| Train | 70% | 700 | Stratified by `Diagnosed_ASD` |
| Validation | 15% | 150 | Stratified by `Diagnosed_ASD` |
| Test | 15% | 150 | Stratified by `Diagnosed_ASD` |

Use `sklearn.model_selection.train_test_split` with `stratify=y`.

### 5.2 Dataset 2 (Eye Tracking)

> [!CAUTION]
> **Participant-level splitting is MANDATORY to prevent data leakage.** Each participant has data in multiple files and multiple trials. If the same participant's trials appear in both train and test, the model learns participant-specific patterns, not ASD-discriminative patterns.

**Strategy: GroupKFold or manual participant-level split**

```
Split by ParticipantID, NOT by file or row:
  - All trials for participant P go into the SAME split
  - Stratify by Class (ASD/TD) to maintain class balance
```

| Split | # Participants | ASD | TD | Strategy |
|-------|---------------|-----|-----|----------|
| Train | ~39 (68%) | ~19 | ~20 | Stratified group split |
| Validation | ~9 (16%) | ~4 | ~5 | Stratified group split |
| Test | ~9 (16%) | ~4 | ~5 | Stratified group split |

**Implementation:**
```python
from sklearn.model_selection import StratifiedGroupKFold

# groups = participant IDs (ensures all trials of one participant stay together)
# y = ASD/TD labels
# Use StratifiedGroupKFold with n_splits=5 for cross-validation
# Or manual stratified group split for fixed train/val/test
```

### 5.3 Combined Fusion Classifier

When fusing Dataset 2 + Dataset 3 model outputs:

- **Dataset 3** has 1,000 independent patients (no overlap with Dataset 2)
- **Dataset 2** has 57 participants with eye-tracking data
- These are **separate patient populations** — fusion will combine model predictions, not raw data
- Train each model on its respective dataset with proper splits
- For the final ASD classifier, hold out a common test protocol

---

## 6. Critical Observations & Warnings

### 6.1 Dataset 3

> [!NOTE]
> **Likely synthetic data.** The dataset has perfect completeness (zero missing values), uniform feature distributions, UUID identifiers, and round numbers that suggest it was programmatically generated rather than collected from real clinical records. This affects generalizability claims but is fine for model benchmarking.

### 6.2 Dataset 2

> [!WARNING]
> **Missing participants:** ParticipantIDs 12 and 16 (both ASD) have no eye-tracking data. This reduces usable ASD participants from 29 to 27.

> [!WARNING]
> **`Unidentified(Neg/Pos)` rows** must be excluded — these are samples where the eye tracker lost track of which participant was being recorded. They cannot be assigned a label.

> [!WARNING]
> **Schema inconsistency:** Column sets vary across files (37 to 58 columns). Any preprocessing pipeline must handle this by selecting only the common core columns or by explicitly handling missing columns per file.

> [!IMPORTANT]
> **Missing value encoding:** Gaze and pupil values use the string `'-'` (dash) rather than NaN to indicate missing/invalid samples (e.g., during blinks). These must be converted to NaN before numeric processing.

> [!IMPORTANT]
> **Monocular vs binocular files:** Files 8 and 13 contain only right-eye data. The preprocessing pipeline must handle missing left-eye columns gracefully.

> [!WARNING]
> **Highly unequal sequence lengths:** Participant row counts range from hundreds to tens of thousands depending on how many experiments they participated in. Padding/truncation strategy is critical.

> [!NOTE]
> **Gaze Vector columns are all zeros in some files** (confirmed in `1.csv`). The `Point of Regard` (screen pixel coordinates) is the reliably populated gaze measurement. When `Gaze Vector` is all zeros, use `Point of Regard` as the primary gaze feature. The `Gaze Vector` 3D direction data appears populated only in files from certain recording setups.

### 6.3 General

- **No data was modified** during this inspection
- **No files were merged** or created (except this document)
- **All column names, values, and statistics** were extracted directly from the raw CSV files
