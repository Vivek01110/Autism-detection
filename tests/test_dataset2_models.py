"""
Unit tests for Dataset 2 Models: Mamba (Primary) and PatchTST (Benchmark).
"""

import sys
import os
import pytest
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dataset_2_mamba.model import MambaClassifier
from src.models.dataset_2_mamba.inference import (
    predict_batch as mamba_predict_batch,
    predict_asd_probability as mamba_predict_prob,
)
from src.models.dataset_2_patchtst.model import PatchTSTClassifier
from src.models.dataset_2_patchtst.inference import (
    predict_batch as patchtst_predict_batch,
    predict_asd_probability as patchtst_predict_prob,
)
from src.data_preprocessing.dataset3_preprocessing import load_config
from src.data_preprocessing.dataset2_preprocessing import Dataset2Preprocessor


@pytest.fixture(scope="module")
def sample_batch():
    """Generates synthetic batch tensor (B=4, L=200, D=8)."""
    return torch.randn(4, 200, 8)


class TestMambaModel:
    """Test suite for Mamba (Selective State Space Model)."""

    def test_mamba_forward_pass(self, sample_batch):
        model = MambaClassifier(input_dim=8, d_model=32, d_state=8, d_conv=4, n_layers=2)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch)
        assert isinstance(out, torch.Tensor), "Mamba output should be a torch.Tensor"
        assert out.shape == (4,), f"Expected shape (4,), got {out.shape}"

    def test_mamba_output_shape(self):
        model = MambaClassifier(input_dim=8, d_model=32, d_state=8, n_layers=1)
        for b in [1, 2, 8, 16]:
            x = torch.randn(b, 200, 8)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (b,), f"Expected shape ({b},), got {out.shape}"

    def test_mamba_probability_range(self):
        model = MambaClassifier(input_dim=8, d_model=32, d_state=8, n_layers=2)
        model.eval()
        x = torch.randn(20, 200, 8)
        with torch.no_grad():
            probs = model(x)
        assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Mamba probabilities outside [0, 1]"

    def test_mamba_different_batch_sizes(self):
        model = MambaClassifier(input_dim=8, d_model=32, d_state=8, n_layers=1)
        for bs in [1, 3, 7, 32]:
            x = torch.randn(bs, 200, 8)
            with torch.no_grad():
                out = model(x)
            assert len(out) == bs

    def test_mamba_inference_functions(self):
        model = MambaClassifier(input_dim=8, d_model=32, d_state=8, n_layers=1)
        device = torch.device("cpu")
        model.to(device)

        # predict_batch
        X = np.random.randn(5, 200, 8).astype(np.float32)
        batch_probs = mamba_predict_batch(model, X, device)
        assert isinstance(batch_probs, np.ndarray)
        assert batch_probs.shape == (5,)
        assert np.all((batch_probs >= 0.0) & (batch_probs <= 1.0))

        # predict_asd_probability
        single_seq = np.random.randn(200, 8).astype(np.float32)
        prob = mamba_predict_prob(model, single_seq, scaler=None, device=device)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0


class TestPatchTSTModel:
    """Test suite for PatchTST (Patch Time Series Transformer)."""

    def test_patchtst_forward_pass(self, sample_batch):
        model = PatchTSTClassifier(
            input_dim=8, seq_len=200, patch_len=16, stride=8, d_model=32, n_heads=2, n_layers=1
        )
        model.eval()
        with torch.no_grad():
            out = model(sample_batch)
        assert isinstance(out, torch.Tensor), "PatchTST output should be a torch.Tensor"
        assert out.shape == (4,), f"Expected shape (4,), got {out.shape}"

    def test_patchtst_output_shape(self):
        model = PatchTSTClassifier(
            input_dim=8, seq_len=200, patch_len=16, stride=8, d_model=32, n_heads=2, n_layers=1
        )
        for b in [1, 2, 8, 16]:
            x = torch.randn(b, 200, 8)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (b,), f"Expected shape ({b},), got {out.shape}"

    def test_patchtst_probability_range(self):
        model = PatchTSTClassifier(
            input_dim=8, seq_len=200, patch_len=16, stride=8, d_model=32, n_heads=2, n_layers=1
        )
        model.eval()
        x = torch.randn(20, 200, 8)
        with torch.no_grad():
            probs = model(x)
        assert (probs >= 0.0).all() and (probs <= 1.0).all(), "PatchTST probabilities outside [0, 1]"

    def test_patchtst_different_batch_sizes(self):
        model = PatchTSTClassifier(
            input_dim=8, seq_len=200, patch_len=16, stride=8, d_model=32, n_heads=2, n_layers=1
        )
        for bs in [1, 3, 7, 32]:
            x = torch.randn(bs, 200, 8)
            with torch.no_grad():
                out = model(x)
            assert len(out) == bs

    def test_patchtst_inference_functions(self):
        model = PatchTSTClassifier(
            input_dim=8, seq_len=200, patch_len=16, stride=8, d_model=32, n_heads=2, n_layers=1
        )
        device = torch.device("cpu")
        model.to(device)

        # predict_batch
        X = np.random.randn(5, 200, 8).astype(np.float32)
        batch_probs = patchtst_predict_batch(model, X, device)
        assert isinstance(batch_probs, np.ndarray)
        assert batch_probs.shape == (5,)
        assert np.all((batch_probs >= 0.0) & (batch_probs <= 1.0))

        # predict_asd_probability
        single_seq = np.random.randn(200, 8).astype(np.float32)
        prob = patchtst_predict_prob(model, single_seq, scaler=None, device=device)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0


class TestModelCompatibilityAndNoLeakage:
    """Verify sequence length, feature dimensions, and absence of participant ID leakage."""

    def test_feature_dimension_compatibility(self):
        """Ensure models support varying feature dimensions."""
        for feat_dim in [4, 8, 12]:
            mamba = MambaClassifier(input_dim=feat_dim, d_model=16, d_state=4, n_layers=1)
            patchtst = PatchTSTClassifier(
                input_dim=feat_dim, seq_len=200, patch_len=16, stride=8, d_model=16, n_heads=2, n_layers=1
            )
            x = torch.randn(2, 200, feat_dim)
            with torch.no_grad():
                assert mamba(x).shape == (2,)
                assert patchtst(x).shape == (2,)

    def test_sequence_length_compatibility(self):
        """Ensure models support sequence length 200 as specified."""
        x = torch.randn(2, 200, 8)
        mamba = MambaClassifier(input_dim=8, d_model=16, d_state=4, n_layers=1)
        patchtst = PatchTSTClassifier(
            input_dim=8, seq_len=200, patch_len=16, stride=8, d_model=16, n_heads=2, n_layers=1
        )
        with torch.no_grad():
            assert mamba(x).shape == (2,)
            assert patchtst(x).shape == (2,)

    def test_neither_model_receives_participant_id(self):
        """
        Verify that preprocessed dataset tensors feed only 8 features to models
        and never include participant_id as an input channel.
        """
        config = load_config("configs/config.yaml")
        preprocessor = Dataset2Preprocessor(config)
        preprocessor.run()

        # Check feature names: exactly 8, none of them is ParticipantID
        feature_names = preprocessor.FEATURE_NAMES
        assert len(feature_names) == 8
        assert "ParticipantID" not in feature_names
        assert "participant_id" not in feature_names
        assert "id" not in [f.lower() for f in feature_names]

        # Check train sequence tensor dimensions
        train_seqs, train_labels, train_pids = preprocessor.get_split("train")
        assert train_seqs.shape[2] == 8
        assert train_pids.ndim == 1, "Participant IDs must be separate 1D array"
        assert train_pids.shape[0] == train_seqs.shape[0]
