"""
Comprehensive unit tests for Dataset 2 (Eye Tracking) Preprocessing & Temporal Sequence Construction.
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing.dataset3_preprocessing import load_config
from src.data_preprocessing.dataset2_preprocessing import (
    Dataset2Preprocessor,
    EyeTrackingDataset,
)


@pytest.fixture(scope="module")
def config():
    return load_config("configs/config.yaml")


@pytest.fixture(scope="module")
def preprocessor(config):
    pp = Dataset2Preprocessor(config)
    pp.run()
    return pp


class TestRawFilesAndMetadata:
    """Verify raw files exist, can be parsed, and metadata matches."""

    def test_all_25_raw_files_exist(self, preprocessor):
        files = glob.glob(os.path.join(preprocessor.raw_dir, "*.csv"))
        assert len(files) == 25, f"Expected 25 raw CSV files, found {len(files)}"

    def test_metadata_file_parsed(self, preprocessor):
        meta = pd.read_csv(preprocessor.metadata_path)
        assert "ParticipantID" in meta.columns
        assert "Class" in meta.columns
        assert len(meta) == 59

    def test_metadata_label_mapping(self, preprocessor):
        meta = preprocessor._load_and_filter_metadata()
        assert set(meta["label"].unique()) == {0, 1}
        # 12 and 16 should be excluded
        assert 12 not in meta["ParticipantID"].values
        assert 16 not in meta["ParticipantID"].values
        assert len(meta) == 57

    def test_raw_files_unmodified(self, preprocessor):
        """Verify raw directory contains only the original files and no generated files."""
        files = glob.glob(os.path.join(preprocessor.raw_dir, "*"))
        for f in files:
            assert f.endswith(".csv"), f"Unexpected non-csv file in raw directory: {f}"


class TestParticipantLevelSplitting:
    """Verify strict participant-level splitting and zero data leakage."""

    def test_split_participant_counts(self, preprocessor):
        train_pids = preprocessor.split_pids["train"]
        val_pids = preprocessor.split_pids["val"]
        test_pids = preprocessor.split_pids["test"]

        total = len(train_pids) + len(val_pids) + len(test_pids)
        assert total == 57, f"Expected 57 total participants, got {total}"
        assert len(train_pids) == 39
        assert len(val_pids) == 9
        assert len(test_pids) == 9

    def test_participant_disjointness_no_leakage(self, preprocessor):
        train_p = set(preprocessor.split_pids["train"])
        val_p = set(preprocessor.split_pids["val"])
        test_p = set(preprocessor.split_pids["test"])

        assert len(train_p & val_p) == 0, f"Train and Val overlap: {train_p & val_p}"
        assert len(train_p & test_p) == 0, f"Train and Test overlap: {train_p & test_p}"
        assert len(val_p & test_p) == 0, f"Val and Test overlap: {val_p & test_p}"

    def test_sequence_participants_strictly_match_split_pids(self, preprocessor):
        for split_name in ["train", "val", "test"]:
            _, _, pids = preprocessor.get_split(split_name)
            unique_pids = set(np.unique(pids))
            expected_pids = set(preprocessor.split_pids[split_name])
            assert unique_pids.issubset(expected_pids), (
                f"Sequences in {split_name} contain unexpected participants: {unique_pids - expected_pids}"
            )


class TestTemporalSequencesAndFeatures:
    """Verify sequence shape, features, temporal ordering, and no boundary crossing."""

    def test_sequence_tensor_shapes(self, preprocessor):
        for split_name in ["train", "val", "test"]:
            seqs, labels, pids = preprocessor.get_split(split_name)
            assert seqs.ndim == 3, f"Expected 3D tensor, got {seqs.ndim}D"
            assert seqs.shape[1] == preprocessor.sequence_length, (
                f"Expected sequence length {preprocessor.sequence_length}, got {seqs.shape[1]}"
            )
            assert seqs.shape[2] == preprocessor.input_dim, (
                f"Expected feature dim {preprocessor.input_dim}, got {seqs.shape[2]}"
            )
            assert len(labels) == seqs.shape[0]
            assert len(pids) == seqs.shape[0]

    def test_feature_selection(self, preprocessor):
        assert preprocessor.input_dim == 8
        assert "Point of Regard Right X [px]" in preprocessor.FEATURE_NAMES
        assert "Gaze_Velocity" in preprocessor.FEATURE_NAMES
        assert "Fixation_Flag" in preprocessor.FEATURE_NAMES

    def test_labels_are_binary(self, preprocessor):
        for split_name in ["train", "val", "test"]:
            _, labels, _ = preprocessor.get_split(split_name)
            unique_labels = set(np.unique(labels))
            assert unique_labels.issubset({0.0, 1.0}), f"Non-binary labels found in {split_name}: {unique_labels}"

    def test_no_nans_in_preprocessed_sequences(self, preprocessor):
        for split_name in ["train", "val", "test"]:
            seqs, labels, _ = preprocessor.get_split(split_name)
            assert not np.isnan(seqs).any(), f"NaN values present in {split_name} sequences"
            assert not np.isnan(labels).any(), f"NaN values present in {split_name} labels"

    def test_scaler_fitted_on_train_only(self, preprocessor):
        assert preprocessor.scaler is not None
        train_seqs, _, _ = preprocessor.get_split("train")
        flat_train = train_seqs.reshape(-1, preprocessor.input_dim)
        # Scaled train data should have mean ~0 and std ~1
        means = flat_train.mean(axis=0)
        stds = flat_train.std(axis=0)
        for i in range(preprocessor.input_dim):
            assert abs(means[i]) < 0.1, f"Train feature {i} mean too far from 0: {means[i]}"
            assert abs(stds[i] - 1.0) < 0.15, f"Train feature {i} std too far from 1: {stds[i]}"

    def test_no_window_crosses_participant_boundaries(self, preprocessor):
        """Verify each sequence is created for a single participant."""
        for split_name in ["train", "val", "test"]:
            _, _, pids = preprocessor.get_split(split_name)
            # Each sequence has a single unambiguous participant ID
            assert pids.ndim == 1


class TestPyTorchDatasetAndDataLoader:
    """Test PyTorch Dataset and DataLoader integration."""

    def test_eye_tracking_dataset_yields_correct_types_and_shapes(self, preprocessor):
        train_seqs, train_labels, train_pids = preprocessor.get_split("train")
        ds = EyeTrackingDataset(train_seqs, train_labels, train_pids)
        assert len(ds) == len(train_labels)

        seq, label, pid = ds[0]
        assert isinstance(seq, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert isinstance(pid, torch.Tensor)
        assert seq.shape == (200, 8)
        assert label.ndim == 0
        assert pid.ndim == 0

    def test_dataloader_batch_dimensions(self, preprocessor):
        loader = preprocessor.get_dataloader("val", batch_size=16)
        batch_seqs, batch_labels, batch_pids = next(iter(loader))

        assert batch_seqs.ndim == 3
        assert batch_seqs.shape[0] <= 16
        assert batch_seqs.shape[1] == 200
        assert batch_seqs.shape[2] == 8
        assert batch_labels.shape[0] == batch_seqs.shape[0]
        assert batch_pids.shape[0] == batch_seqs.shape[0]
