"""Tests for Dataset 3 preprocessing pipeline."""

import sys
import os
import numpy as np
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing.dataset3_preprocessing import (
    Dataset3Preprocessor,
    TabularDataset,
    load_config,
)


@pytest.fixture(scope="module")
def preprocessor():
    """Run preprocessing once for all tests in this module."""
    config = load_config("configs/config.yaml")
    pp = Dataset3Preprocessor(config, project_root=".")
    pp.run()
    return pp


class TestDataLoading:
    """Verify the raw data is loaded and cleaned correctly."""

    def test_split_sizes_sum(self, preprocessor):
        """Train + val + test rows must equal 1000."""
        total = sum(len(preprocessor.get_split(s)[1]) for s in ["train", "val", "test"])
        assert total == 1000, f"Expected 1000 total rows, got {total}"

    def test_no_nans_in_features(self, preprocessor):
        """No NaN values allowed in any split."""
        for split in ["train", "val", "test"]:
            X, y = preprocessor.get_split(split)
            assert np.isnan(X).sum() == 0, f"NaN in {split} features"
            assert np.isnan(y).sum() == 0, f"NaN in {split} target"

    def test_feature_dimensions(self, preprocessor):
        """Should have 8 features (4 numerical + 4 categorical)."""
        X, _ = preprocessor.get_split("train")
        assert X.shape[1] == 8, f"Expected 8 features, got {X.shape[1]}"
        assert preprocessor.input_dim == 8


class TestTargetEncoding:
    """Verify target encoding is correct."""

    def test_target_is_binary(self, preprocessor):
        """Target must be 0 or 1 only."""
        for split in ["train", "val", "test"]:
            _, y = preprocessor.get_split(split)
            unique = set(np.unique(y))
            assert unique.issubset({0.0, 1.0}), f"Non-binary target in {split}: {unique}"

    def test_class_balance_preserved(self, preprocessor):
        """Stratification should keep ~41% ASD in each split."""
        for split in ["train", "val", "test"]:
            _, y = preprocessor.get_split(split)
            pos_rate = y.mean()
            assert 0.35 < pos_rate < 0.47, (
                f"{split} positive rate {pos_rate:.3f} outside expected range"
            )


class TestCategoricalEncoding:
    """Verify categorical features are properly encoded."""

    def test_categoricals_are_binary(self, preprocessor):
        """Categorical columns must be 0/1 after encoding."""
        X, _ = preprocessor.get_split("train")
        cat_features = preprocessor.cfg["dataset3"]["categorical_features"]
        cat_indices = [
            preprocessor.feature_names.index(f) for f in cat_features
        ]
        for idx in cat_indices:
            unique = set(np.unique(X[:, idx]))
            assert unique.issubset({0.0, 1.0}), f"Feature idx {idx} not binary: {unique}"


class TestDataLeakage:
    """Verify no data leakage from test into training."""

    def test_scaler_fitted_on_train_only(self, preprocessor):
        """Scaler mean should match train data statistics (not global)."""
        X_train, _ = preprocessor.get_split("train")
        num_features = preprocessor.cfg["dataset3"]["numerical_features"]
        num_indices = [preprocessor.feature_names.index(f) for f in num_features]

        # After scaling, train numerical features should have ~mean=0, ~std=1
        for idx in num_indices:
            col = X_train[:, idx]
            assert abs(col.mean()) < 0.1, f"Train col {idx} mean too far from 0: {col.mean()}"
            assert abs(col.std() - 1.0) < 0.15, f"Train col {idx} std too far from 1: {col.std()}"


class TestTabularDataset:
    """Test PyTorch Dataset wrapper."""

    def test_dataset_length(self, preprocessor):
        X, y = preprocessor.get_split("train")
        ds = TabularDataset(X, y)
        assert len(ds) == len(y)

    def test_dataset_returns_tensors(self, preprocessor):
        import torch

        X, y = preprocessor.get_split("train")
        ds = TabularDataset(X, y)
        x_sample, y_sample = ds[0]
        assert isinstance(x_sample, torch.Tensor)
        assert isinstance(y_sample, torch.Tensor)
        assert x_sample.shape == (8,)

    def test_dataloader_batching(self, preprocessor):
        loader = preprocessor.get_dataloader("train", batch_size=32)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape[0] <= 32
        assert batch_x.shape[1] == 8
