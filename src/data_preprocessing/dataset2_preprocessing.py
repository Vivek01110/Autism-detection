"""
Dataset 2 (Eye Tracking) — Preprocessing and Temporal Sequence Construction Pipeline

Loads raw SMI eye-tracking recordings (1.csv through 25.csv) and Metadata_Participants.csv,
handles schema variations across files, maps participant IDs to binary ASD/TD labels,
imputes missing values without leakage, extracts validated gaze and pupil features,
constructs participant-level temporal sequences suitable for Mamba and PatchTST,
executes a leak-free participant-level train/validation/test split,
fits StandardScaler strictly on training participants, and
persists processed datasets and artifacts for model training and evaluation.
"""

import os
import sys
import glob
import json
import joblib
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_preprocessing.dataset3_preprocessing import load_config


# ─────────────────────────────────────────────────────────────
# PyTorch Dataset for Eye-Tracking Temporal Sequences
# ─────────────────────────────────────────────────────────────
class EyeTrackingDataset(Dataset):
    """
    PyTorch Dataset for participant-level eye-tracking temporal sequences.
    Yields:
        sequence: Tensor of shape (sequence_length, num_features)
        label: Tensor (scalar float) 1.0 for ASD, 0.0 for TD
        participant_id: Tensor (scalar int)
    """

    def __init__(self, sequences: np.ndarray, labels: np.ndarray, participant_ids: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.participant_ids = torch.tensor(participant_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx], self.participant_ids[idx]


# ─────────────────────────────────────────────────────────────
# Main Preprocessor for Dataset 2
# ─────────────────────────────────────────────────────────────
class Dataset2Preprocessor:
    """
    End-to-end participant-level preprocessor for Dataset 2 (Eye Tracking).
    """

    # Validated feature channels (8 features)
    FEATURE_NAMES = [
        "Point of Regard Right X [px]",
        "Point of Regard Right Y [px]",
        "Point of Regard Left X [px]",
        "Point of Regard Left Y [px]",
        "Pupil Diameter Right [mm]",
        "Pupil Diameter Left [mm]",
        "Gaze_Velocity",
        "Fixation_Flag",
    ]

    def __init__(self, config: Dict[str, Any], project_root: str = "."):
        self.cfg = config
        self.ds_cfg = config.get("dataset2", {})
        self.project_root = project_root
        self.seed = config.get("random_seed", 42)

        # Paths
        self.raw_dir = self._resolve_raw_dir()
        self.metadata_path = self._resolve_metadata_path()
        self.processed_dir = self._resolve_path(
            self.ds_cfg.get("processed_dir", "data/processed/dataset_2")
        )
        self.artifacts_dir = self._resolve_path(
            self.cfg.get("paths", {}).get("preprocessing_artifacts", "results/preprocessing_artifacts")
        )

        # Hyperparameters
        self.seq_len = int(self.ds_cfg.get("sequence_length", 200))
        self.stride = int(self.ds_cfg.get("stride", 100))
        self.min_seq_len = int(self.ds_cfg.get("min_sequence_length", 50))
        self.test_size = float(self.ds_cfg.get("test_size", 0.15))
        self.val_size = float(self.ds_cfg.get("val_size", 0.15))

        # Populated after run()
        self.scaler: Optional[StandardScaler] = None
        self.split_pids: Dict[str, List[int]] = {}
        self.splits: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.metadata_df: Optional[pd.DataFrame] = None

    # ── Path Resolution ──────────────────────────────────────

    def _resolve_path(self, rel: str) -> str:
        if os.path.isabs(rel):
            return rel
        return os.path.join(self.project_root, rel)

    def _resolve_raw_dir(self) -> str:
        candidates = [
            self._resolve_path(self.ds_cfg.get("raw_dir", "data/dataset_2_eye_tracking/archive-2/Eye-tracking Output")),
            self._resolve_path("../Datasets/2nd_dataset/Eye-tracking Output"),
            self._resolve_path("data/dataset_2_eye_tracking/archive-2/Eye-tracking Output"),
            self._resolve_path("data/dataset_2_eye_tracking/Eye-tracking Output"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        raise FileNotFoundError(f"Eye-tracking raw directory not found. Checked: {candidates}")

    def _resolve_metadata_path(self) -> str:
        candidates = [
            self._resolve_path(self.ds_cfg.get("metadata_path", "data/dataset_2_eye_tracking/archive-2/Metadata_Participants.csv")),
            self._resolve_path("../Datasets/2nd_dataset/Metadata_Participants.csv"),
            self._resolve_path("data/dataset_2_eye_tracking/archive-2/Metadata_Participants.csv"),
            self._resolve_path("data/dataset_2_eye_tracking/Metadata_Participants.csv"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        raise FileNotFoundError(f"Metadata file not found. Checked: {candidates}")

    # ── Public API ──────────────────────────────────────────

    def run(self, force_recompute: bool = False) -> "Dataset2Preprocessor":
        """
        Run preprocessing or load cached preprocessed tensors.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        cache_path = os.path.join(self.processed_dir, "dataset2_processed.pt")

        if os.path.isfile(cache_path) and not force_recompute:
            print(f"[Dataset2] Loading preprocessed data from cache: {cache_path}")
            cached = torch.load(cache_path, weights_only=False)
            self.splits = {
                "train": (cached["train_sequences"], cached["train_labels"], cached["train_pids"]),
                "val": (cached["val_sequences"], cached["val_labels"], cached["val_pids"]),
                "test": (cached["test_sequences"], cached["test_labels"], cached["test_pids"]),
            }
            self.split_pids = cached["split_pids"]
            scaler_path = os.path.join(self.artifacts_dir, "dataset2_scaler.joblib")
            if os.path.isfile(scaler_path):
                self.scaler = joblib.load(scaler_path)
            return self

        print("[Dataset2] Starting end-to-end preprocessing...")
        meta_df = self._load_and_filter_metadata()
        self.metadata_df = meta_df

        # 1. Participant-level stratified split FIRST (prevent leakage)
        train_pids, val_pids, test_pids = self._split_participants(meta_df)
        self.split_pids = {
            "train": sorted(list(train_pids)),
            "val": sorted(list(val_pids)),
            "test": sorted(list(test_pids)),
        }
        self._verify_disjointness()

        # 2. Extract participant recordings and construct sequences
        participant_sequences = self._extract_all_sequences(meta_df)

        # 3. Assemble splits by participant ID
        train_seqs, train_labels, train_pids_arr = self._collect_split_data(participant_sequences, train_pids)
        val_seqs, val_labels, val_pids_arr = self._collect_split_data(participant_sequences, val_pids)
        test_seqs, test_labels, test_pids_arr = self._collect_split_data(participant_sequences, test_pids)

        print(f"[Dataset2] Raw sequences assembled:")
        print(f"  Train: {train_seqs.shape[0]} sequences from {len(train_pids)} participants")
        print(f"  Val:   {val_seqs.shape[0]} sequences from {len(val_pids)} participants")
        print(f"  Test:  {test_seqs.shape[0]} sequences from {len(test_pids)} participants")

        # 4. Fit StandardScaler ONLY on train sequences
        self._scale_splits(train_seqs, val_seqs, test_seqs)

        self.splits = {
            "train": (train_seqs, train_labels, train_pids_arr),
            "val": (val_seqs, val_labels, val_pids_arr),
            "test": (test_seqs, test_labels, test_pids_arr),
        }

        # 5. Save cached tensors and metadata
        self._save_artifacts(cache_path)
        return self

    def get_split(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (sequences, labels, participant_ids) for the split."""
        assert split in self.splits, f"Unknown split: {split}. Expected one of {list(self.splits.keys())}"
        return self.splits[split]

    def get_dataloader(
        self, split: str, batch_size: int, shuffle: Optional[bool] = None
    ) -> DataLoader:
        """Return a PyTorch DataLoader for the split."""
        seqs, labels, pids = self.get_split(split)
        ds = EyeTrackingDataset(seqs, labels, pids)
        if shuffle is None:
            shuffle = (split == "train")
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def get_dataloaders(
        self, batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Return (train_loader, val_loader, test_loader)."""
        train_loader = self.get_dataloader("train", batch_size=batch_size, shuffle=True)
        val_loader = self.get_dataloader("val", batch_size=batch_size, shuffle=False)
        test_loader = self.get_dataloader("test", batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    @property
    def train_loader(self) -> DataLoader:
        batch_size = int(self.ds_cfg.get("batch_size", 32))
        return self.get_dataloader("train", batch_size=batch_size, shuffle=True)

    @property
    def val_loader(self) -> DataLoader:
        batch_size = int(self.ds_cfg.get("batch_size", 32))
        return self.get_dataloader("val", batch_size=batch_size, shuffle=False)

    @property
    def test_loader(self) -> DataLoader:
        batch_size = int(self.ds_cfg.get("batch_size", 32))
        return self.get_dataloader("test", batch_size=batch_size, shuffle=False)

    @property
    def input_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    @property
    def sequence_length(self) -> int:
        return self.seq_len

    # ── Internal Implementation Steps ───────────────────────

    def _load_and_filter_metadata(self) -> pd.DataFrame:
        """Load Metadata_Participants.csv and filter out missing participants (12, 16)."""
        meta = pd.read_csv(self.metadata_path)
        print(f"[Dataset2] Loaded metadata from {self.metadata_path}: {len(meta)} participants")

        # IDs 12 and 16 have zero eye-tracking data in all 25 files
        meta = meta[~meta["ParticipantID"].isin([12, 16])].copy()
        meta["label"] = (meta["Class"] == "ASD").astype(int)

        print(f"[Dataset2] Filtered usable participants: {len(meta)} (ASD: {(meta['label']==1).sum()}, TD: {(meta['label']==0).sum()})")
        return meta

    def _split_participants(self, meta: pd.DataFrame) -> Tuple[set, set, set]:
        """Stratified participant-level train/validation/test split."""
        trainval_meta, test_meta = train_test_split(
            meta,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=meta["label"],
        )
        val_frac = self.val_size / (1.0 - self.test_size)
        train_meta, val_meta = train_test_split(
            trainval_meta,
            test_size=val_frac,
            random_state=self.seed,
            stratify=trainval_meta["label"],
        )

        train_pids = set(train_meta["ParticipantID"].astype(int))
        val_pids = set(val_meta["ParticipantID"].astype(int))
        test_pids = set(test_meta["ParticipantID"].astype(int))

        print(f"[Dataset2] Participant split:")
        print(f"  Train: {len(train_pids)} (ASD: {(train_meta['label']==1).sum()}, TD: {(train_meta['label']==0).sum()})")
        print(f"  Val:   {len(val_pids)} (ASD: {(val_meta['label']==1).sum()}, TD: {(val_meta['label']==0).sum()})")
        print(f"  Test:  {len(test_pids)} (ASD: {(test_meta['label']==1).sum()}, TD: {(test_meta['label']==0).sum()})")

        return train_pids, val_pids, test_pids

    def _verify_disjointness(self) -> None:
        """Verify zero participant overlap across splits."""
        t_p = set(self.split_pids["train"])
        v_p = set(self.split_pids["val"])
        te_p = set(self.split_pids["test"])

        assert len(t_p & v_p) == 0, f"Leakage: Train & Val overlap: {t_p & v_p}"
        assert len(t_p & te_p) == 0, f"Leakage: Train & Test overlap: {t_p & te_p}"
        assert len(v_p & te_p) == 0, f"Leakage: Val & Test overlap: {v_p & te_p}"
        print("[Dataset2] [OK] Disjointness verified: Zero participant leakage across splits.")

    def _extract_all_sequences(
        self, meta: pd.DataFrame
    ) -> Dict[int, List[np.ndarray]]:
        """
        Iterate through all 25 CSV files, extract participant trials,
        sort temporally, compute features vectorized per file, and generate sliding window sequences.
        """
        pid_to_label = dict(zip(meta["ParticipantID"].astype(int), meta["label"]))
        valid_pids = set(pid_to_label.keys())
        participant_sequences: Dict[int, List[np.ndarray]] = {p: [] for p in valid_pids}

        files = sorted(
            glob.glob(os.path.join(self.raw_dir, "*.csv")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
            if os.path.splitext(os.path.basename(x))[0].isdigit()
            else 999,
        )
        print(f"[Dataset2] Scanning {len(files)} raw CSV files from {self.raw_dir}...")
        total_windows = 0

        for file_idx, fpath in enumerate(files, 1):
            with open(fpath, "r") as fp:
                header = [c.strip() for c in fp.readline().strip().split(",")]

            load_cols = [c for c in [
                "Participant", "RecordingTime [ms]", "Trial", "Category Group", "Category Right",
                "Point of Regard Right X [px]", "Point of Regard Right Y [px]",
                "Point of Regard Left X [px]", "Point of Regard Left Y [px]",
                "Pupil Diameter Right [mm]", "Pupil Diameter Left [mm]",
            ] if c in header]

            df = pd.read_csv(fpath, usecols=load_cols, na_values=["-"], low_memory=False)

            # Filter valid numeric participants
            df = df[df["Participant"].astype(str).str.isdigit()]
            if df.empty:
                continue
            df["Participant"] = df["Participant"].astype(int)
            df = df[df["Participant"].isin(valid_pids)]

            # Filter for Eye rows (discard separator/information rows)
            if "Category Group" in df.columns:
                df = df[df["Category Group"] == "Eye"]

            if df.empty:
                continue

            # Sort strictly by Participant, Trial, and RecordingTime [ms]
            df = df.sort_values(by=["Participant", "Trial", "RecordingTime [ms]"])

            # Vectorized feature extraction for the entire file
            feats = self._extract_features_vectorized(df)

            # Detect boundaries where participant or trial changes
            pids = df["Participant"].values
            trials = df["Trial"].values
            change = (pids[1:] != pids[:-1]) | (trials[1:] != trials[:-1])
            splits = np.concatenate([[0], np.where(change)[0] + 1, [len(pids)]])

            # Reset velocity at boundaries to prevent cross-trial derivative artifact
            boundary_idx = np.concatenate([[0], np.where(change)[0] + 1])
            feats[boundary_idx, 6] = 0.0

            # Slice windows within each trial segment
            file_windows = 0
            for s, e in zip(splits[:-1], splits[1:]):
                pid = int(pids[s])
                trial_matrix = feats[s:e]
                windows = self._slice_into_windows(trial_matrix)
                if windows:
                    participant_sequences[pid].extend(windows)
                    file_windows += len(windows)
                    total_windows += len(windows)

            print(f"[Dataset2] File {file_idx:2d}.csv: extracted {file_windows:5d} sequences from {len(splits)-1:4d} trials")

        print(f"[Dataset2] Generated {total_windows:,} temporal sequences of shape ({self.seq_len}, {len(self.FEATURE_NAMES)}).")
        return participant_sequences

    def _extract_features_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized extraction of the 8 validated features from a sorted file DataFrame.
        """
        n = len(df)
        feats = np.zeros((n, len(self.FEATURE_NAMES)), dtype=np.float32)

        # 1 & 2: Point of Regard Right X/Y
        rx = pd.to_numeric(df.get("Point of Regard Right X [px]", np.nan), errors="coerce")
        ry = pd.to_numeric(df.get("Point of Regard Right Y [px]", np.nan), errors="coerce")
        feats[:, 0] = rx.ffill().bfill().fillna(0.0).values
        feats[:, 1] = ry.ffill().bfill().fillna(0.0).values

        # 3 & 4: Point of Regard Left X/Y (if absent, copy Right eye)
        if "Point of Regard Left X [px]" in df.columns:
            lx = pd.to_numeric(df["Point of Regard Left X [px]"], errors="coerce")
            lx_interp = lx.ffill().bfill().values
            feats[:, 2] = np.where(np.isnan(lx_interp), feats[:, 0], lx_interp)
        else:
            feats[:, 2] = feats[:, 0]

        if "Point of Regard Left Y [px]" in df.columns:
            ly = pd.to_numeric(df["Point of Regard Left Y [px]"], errors="coerce")
            ly_interp = ly.ffill().bfill().values
            feats[:, 3] = np.where(np.isnan(ly_interp), feats[:, 1], ly_interp)
        else:
            feats[:, 3] = feats[:, 1]

        # 5 & 6: Pupil Diameter Right / Left [mm]
        if "Pupil Diameter Right [mm]" in df.columns:
            pr = pd.to_numeric(df["Pupil Diameter Right [mm]"], errors="coerce")
            feats[:, 4] = pr.ffill().bfill().fillna(3.5).values
        else:
            feats[:, 4] = 3.5

        if "Pupil Diameter Left [mm]" in df.columns:
            pl = pd.to_numeric(df["Pupil Diameter Left [mm]"], errors="coerce")
            pl_interp = pl.ffill().bfill().values
            feats[:, 5] = np.where(np.isnan(pl_interp), feats[:, 4], pl_interp)
        else:
            feats[:, 5] = feats[:, 4]

        # 7: Gaze Velocity (pixels / ms)
        time_vals = pd.to_numeric(df["RecordingTime [ms]"], errors="coerce").values
        dx = np.diff(feats[:, 0], prepend=feats[0, 0])
        dy = np.diff(feats[:, 1], prepend=feats[0, 1])
        dt = np.diff(time_vals, prepend=time_vals[0] if len(time_vals) > 0 else 0)
        dt = np.where(dt <= 0, 24.0, dt)
        velocity = np.sqrt(dx**2 + dy**2) / dt
        feats[:, 6] = np.nan_to_num(velocity, nan=0.0)

        # 8: Fixation Flag (1.0 if Fixation, else 0.0)
        if "Category Right" in df.columns:
            feats[:, 7] = (df["Category Right"] == "Fixation").astype(float).values
        else:
            feats[:, 7] = 0.0

        return feats

    def _slice_into_windows(self, matrix: np.ndarray) -> List[np.ndarray]:
        """
        Slice trial matrix of shape (N, D) into windows of length seq_len.
        Never crosses trial boundary.
        """
        n, d = matrix.shape
        if n < self.min_seq_len:
            return []

        if n < self.seq_len:
            # Pad to seq_len
            pad = np.zeros((self.seq_len - n, d), dtype=np.float32)
            # Edge-pad with last sample
            pad[:] = matrix[-1]
            return [np.vstack([matrix, pad])]

        windows = []
        for start in range(0, n - self.seq_len + 1, self.stride):
            windows.append(matrix[start : start + self.seq_len].copy())

        # Include remainder tail if sufficiently long and not redundant
        remainder = n - (start + self.seq_len)
        if remainder >= self.min_seq_len:
            windows.append(matrix[-self.seq_len :].copy())

        return windows

    def _collect_split_data(
        self, participant_sequences: Dict[int, List[np.ndarray]], pids: set
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assemble all windowed sequences for the given participant IDs."""
        seq_list = []
        label_list = []
        pid_list = []

        for pid in sorted(list(pids)):
            label = 1.0 if pid in self.metadata_df.loc[self.metadata_df["label"] == 1, "ParticipantID"].values else 0.0
            windows = participant_sequences.get(pid, [])
            for w in windows:
                seq_list.append(w)
                label_list.append(label)
                pid_list.append(pid)

        if not seq_list:
            return (
                np.empty((0, self.seq_len, len(self.FEATURE_NAMES)), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        return (
            np.array(seq_list, dtype=np.float32),
            np.array(label_list, dtype=np.float32),
            np.array(pid_list, dtype=np.int64),
        )

    def _scale_splits(
        self, train_seqs: np.ndarray, val_seqs: np.ndarray, test_seqs: np.ndarray
    ) -> None:
        """
        Fit StandardScaler STRICTLY on the training participants' sequences.
        Transform validation and test sequences without leakage.
        """
        num_features = len(self.FEATURE_NAMES)
        train_flat = train_seqs.reshape(-1, num_features)

        self.scaler = StandardScaler()
        self.scaler.fit(train_flat)
        print(f"[Dataset2] StandardScaler fitted on {train_flat.shape[0]:,} training timesteps.")

        # Transform in-place
        train_seqs[:] = self.scaler.transform(train_flat).reshape(train_seqs.shape)
        val_seqs[:] = self.scaler.transform(val_seqs.reshape(-1, num_features)).reshape(val_seqs.shape)
        test_seqs[:] = self.scaler.transform(test_seqs.reshape(-1, num_features)).reshape(test_seqs.shape)

    def _save_artifacts(self, cache_path: str) -> None:
        """Save preprocessed PyTorch tensors, scaler, and metadata JSON."""
        # 1. Save tensor cache
        cache_data = {
            "train_sequences": self.splits["train"][0],
            "train_labels": self.splits["train"][1],
            "train_pids": self.splits["train"][2],
            "val_sequences": self.splits["val"][0],
            "val_labels": self.splits["val"][1],
            "val_pids": self.splits["val"][2],
            "test_sequences": self.splits["test"][0],
            "test_labels": self.splits["test"][1],
            "test_pids": self.splits["test"][2],
            "split_pids": self.split_pids,
            "feature_names": self.FEATURE_NAMES,
            "sequence_length": self.seq_len,
        }
        torch.save(cache_data, cache_path)
        print(f"[Dataset2] Processed data saved to {cache_path}")

        # 2. Save scaler
        scaler_path = os.path.join(self.artifacts_dir, "dataset2_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)

        # 3. Save metadata JSON
        meta = {
            "feature_names": self.FEATURE_NAMES,
            "sequence_length": self.seq_len,
            "stride": self.stride,
            "num_features": len(self.FEATURE_NAMES),
            "split_participants": {k: [int(x) for x in v] for k, v in self.split_pids.items()},
            "split_counts": {
                "train_sequences": int(len(self.splits["train"][1])),
                "val_sequences": int(len(self.splits["val"][1])),
                "test_sequences": int(len(self.splits["test"][1])),
            },
            "split_class_distribution": {
                "train": {
                    "ASD": int((self.splits["train"][1] == 1).sum()),
                    "TD": int((self.splits["train"][1] == 0).sum()),
                },
                "val": {
                    "ASD": int((self.splits["val"][1] == 1).sum()),
                    "TD": int((self.splits["val"][1] == 0).sum()),
                },
                "test": {
                    "ASD": int((self.splits["test"][1] == 1).sum()),
                    "TD": int((self.splits["test"][1] == 0).sum()),
                },
            },
        }
        meta_path = os.path.join(self.artifacts_dir, "dataset2_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Dataset2] Artifacts and metadata saved to {self.artifacts_dir}")


# ─────────────────────────────────────────────────────────────
# Standalone execution
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    pp = Dataset2Preprocessor(cfg)
    pp.run(force_recompute=True)
