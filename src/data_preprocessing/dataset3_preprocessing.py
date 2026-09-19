"""
Dataset 3 (Clinical / Tabular) — Preprocessing Pipeline

Loads asd.csv, cleans, encodes, splits, and scales.
Transformations are fitted ONLY on the training set to prevent leakage.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────
# Helper: load config
# ─────────────────────────────────────────────────────────────
def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────
class TabularDataset(Dataset):
    """PyTorch Dataset for tabular ASD data (works for both KAN and TabNet)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────
# Main Preprocessor
# ─────────────────────────────────────────────────────────────
class Dataset3Preprocessor:
    """
    End-to-end preprocessor for Dataset 3 (clinical / tabular).

    Usage:
        pp = Dataset3Preprocessor(config)
        pp.run()
        X_train, y_train = pp.get_split("train")
    """

    def __init__(self, config: dict, project_root: str = "."):
        self.cfg = config
        self.ds_cfg = config["dataset3"]
        self.project_root = project_root
        self.seed = config["random_seed"]

        # Will be populated by run()
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        self.splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ── public API ──────────────────────────────────────────

    def run(self) -> "Dataset3Preprocessor":
        """Execute the full preprocessing pipeline."""
        df = self._load()
        df = self._clean(df)
        df = self._encode(df)
        X, y = self._separate_features_target(df)
        self._split_and_scale(X, y)
        self._save_artifacts()
        return self

    def get_split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) numpy arrays for the requested split."""
        assert split in self.splits, f"Unknown split: {split}"
        return self.splits[split]

    def get_dataloader(
        self, split: str, batch_size: int, shuffle: bool | None = None
    ) -> DataLoader:
        """Return a PyTorch DataLoader for the requested split."""
        X, y = self.get_split(split)
        ds = TabularDataset(X, y)
        if shuffle is None:
            shuffle = split == "train"
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def prepare_data(self) -> "Dataset3Preprocessor":
        """Alias for run()."""
        return self.run()

    def process(self) -> "Dataset3Preprocessor":
        """Alias for run()."""
        return self.run()

    def get_dataloaders(
        self, batch_size: int = 64
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Return (train_loader, val_loader, test_loader)."""
        train_loader = self.get_dataloader("train", batch_size=batch_size, shuffle=True)
        val_loader = self.get_dataloader("val", batch_size=batch_size, shuffle=False)
        test_loader = self.get_dataloader("test", batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    @property
    def train_loader(self) -> DataLoader:
        batch_size = self.cfg.get("kan", {}).get("batch_size", 64)
        return self.get_dataloader("train", batch_size=batch_size, shuffle=True)

    @property
    def val_loader(self) -> DataLoader:
        batch_size = self.cfg.get("kan", {}).get("batch_size", 64)
        return self.get_dataloader("val", batch_size=batch_size, shuffle=False)

    @property
    def test_loader(self) -> DataLoader:
        batch_size = self.cfg.get("kan", {}).get("batch_size", 64)
        return self.get_dataloader("test", batch_size=batch_size, shuffle=False)

    @property
    def input_dim(self) -> int:
        """Number of input features after preprocessing."""
        return len(self.feature_names)

    # ── internal steps ──────────────────────────────────────

    def _resolve_path(self, rel: str) -> str:
        return os.path.join(self.project_root, rel)

    def _load(self) -> pd.DataFrame:
        csv_path = self._resolve_path(self.cfg["paths"]["dataset3"])
        df = pd.read_csv(csv_path)
        print(f"[Preprocessing] Loaded {csv_path}: {df.shape[0]} rows, {df.shape[1]} cols")
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop artifact columns."""
        drop_cols = self.ds_cfg["drop_columns"]
        existing = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing)
        print(f"[Preprocessing] Dropped columns: {existing}")
        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features and target."""
        # Encode binary categoricals
        for col, mapping in self.ds_cfg["categorical_mappings"].items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                assert df[col].isna().sum() == 0, f"Unmapped values in {col}"

        # Encode target
        target = self.ds_cfg["target_column"]
        df[target] = df[target].map(self.ds_cfg["target_mapping"])
        assert df[target].isna().sum() == 0, "Unmapped values in target"

        print(f"[Preprocessing] Encoded categoricals and target")
        return df

    def _separate_features_target(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split dataframe into feature matrix X and target vector y."""
        target = self.ds_cfg["target_column"]
        feature_cols = (
            self.ds_cfg["numerical_features"] + self.ds_cfg["categorical_features"]
        )
        self.feature_names = feature_cols

        X = df[feature_cols].values.astype(np.float32)
        y = df[target].values.astype(np.float32)

        assert np.isnan(X).sum() == 0, "NaN found in features"
        assert np.isnan(y).sum() == 0, "NaN found in target"

        print(f"[Preprocessing] Features: {len(feature_cols)} cols, Target: {target}")
        return X, y

    def _split_and_scale(self, X: np.ndarray, y: np.ndarray) -> None:
        """Stratified split + StandardScaler fitted on train only."""
        test_size = self.ds_cfg["test_size"]
        val_size = self.ds_cfg["val_size"]

        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.seed,
            stratify=y,
        )
        # Second split: train vs val (from the remaining data)
        # val_size is fraction of total, convert to fraction of trainval
        val_frac_of_trainval = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_frac_of_trainval,
            random_state=self.seed,
            stratify=y_trainval,
        )

        print(f"[Preprocessing] Split — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
        print(f"[Preprocessing] Train pos rate: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")

        # Fit scaler on TRAIN only — scale numerical features
        num_features = self.ds_cfg["numerical_features"]
        num_indices = [self.feature_names.index(f) for f in num_features]

        self.scaler = StandardScaler()
        X_train[:, num_indices] = self.scaler.fit_transform(X_train[:, num_indices])
        X_val[:, num_indices] = self.scaler.transform(X_val[:, num_indices])
        X_test[:, num_indices] = self.scaler.transform(X_test[:, num_indices])

        print(f"[Preprocessing] StandardScaler fitted on train ({len(num_indices)} numerical features)")

        self.splits = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    def _save_artifacts(self) -> None:
        """Save scaler and metadata for reproducible inference."""
        artifacts_dir = self._resolve_path(self.cfg["paths"]["preprocessing_artifacts"])
        os.makedirs(artifacts_dir, exist_ok=True)

        # Save scaler
        scaler_path = os.path.join(artifacts_dir, "dataset3_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)

        # Save metadata
        meta = {
            "feature_names": self.feature_names,
            "numerical_features": self.ds_cfg["numerical_features"],
            "categorical_features": self.ds_cfg["categorical_features"],
            "categorical_mappings": self.ds_cfg["categorical_mappings"],
            "target_mapping": self.ds_cfg["target_mapping"],
            "input_dim": self.input_dim,
            "split_sizes": {
                k: len(v[1]) for k, v in self.splits.items()
            },
        }
        meta_path = os.path.join(artifacts_dir, "dataset3_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[Preprocessing] Artifacts saved to {artifacts_dir}")


# ─────────────────────────────────────────────────────────────
# Standalone execution
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    pp = Dataset3Preprocessor(config)
    pp.run()

    # Quick summary
    for split_name in ["train", "val", "test"]:
        X, y = pp.get_split(split_name)
        print(f"  {split_name}: X={X.shape}, y={y.shape}, pos_rate={y.mean():.3f}")
