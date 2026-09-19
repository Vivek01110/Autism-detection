"""Tests for KAN and TabNet model forward passes."""

import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestKANModel:
    """Test KAN model architecture and forward pass."""

    def test_kan_forward_pass(self):
        from src.models.dataset_3_kan.model import KAN

        model = KAN(input_dim=8, hidden_layers=[32, 16], output_dim=1)
        x = torch.randn(4, 8)
        out = model(x)
        assert out.shape == (4,), f"Expected shape (4,), got {out.shape}"

    def test_kan_output_range(self):
        """Output probabilities must be in [0, 1]."""
        from src.models.dataset_3_kan.model import KAN

        model = KAN(input_dim=8, hidden_layers=[32, 16], output_dim=1)
        model.eval()
        x = torch.randn(100, 8)
        with torch.no_grad():
            out = model(x)
        assert (out >= 0).all() and (out <= 1).all(), "Probabilities outside [0,1]"

    def test_kan_different_input_dims(self):
        """Model should work with any input dimension."""
        from src.models.dataset_3_kan.model import KAN

        for dim in [4, 8, 16, 32]:
            model = KAN(input_dim=dim, hidden_layers=[16], output_dim=1)
            x = torch.randn(2, dim)
            out = model(x)
            assert out.shape == (2,)

    def test_kan_single_sample(self):
        """Model should handle batch size = 1."""
        from src.models.dataset_3_kan.model import KAN

        model = KAN(input_dim=8, hidden_layers=[16], output_dim=1)
        x = torch.randn(1, 8)
        out = model(x)
        assert out.shape == (1,)


class TestTabNetModel:
    """Test TabNet model architecture and forward pass."""

    def test_tabnet_forward_pass(self):
        from src.models.dataset_3_tabnet.model import TabNet

        model = TabNet(input_dim=8, n_d=8, n_a=8, n_steps=3)
        x = torch.randn(4, 8)
        probs, sparsity_loss = model(x)
        assert probs.shape == (4,), f"Expected shape (4,), got {probs.shape}"
        assert isinstance(sparsity_loss, torch.Tensor)

    def test_tabnet_output_range(self):
        """Output probabilities must be in [0, 1]."""
        from src.models.dataset_3_tabnet.model import TabNet

        model = TabNet(input_dim=8, n_d=8, n_a=8, n_steps=3)
        model.eval()
        x = torch.randn(100, 8)
        with torch.no_grad():
            probs, _ = model(x)
        assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities outside [0,1]"

    def test_tabnet_sparsity_loss_nonneg(self):
        """Sparsity loss should be non-negative."""
        from src.models.dataset_3_tabnet.model import TabNet

        model = TabNet(input_dim=8, n_d=8, n_a=8, n_steps=3)
        x = torch.randn(16, 8)
        _, sparsity_loss = model(x)
        assert sparsity_loss.item() >= 0, "Sparsity loss is negative"

    def test_tabnet_different_input_dims(self):
        from src.models.dataset_3_tabnet.model import TabNet

        for dim in [4, 8, 16]:
            model = TabNet(input_dim=dim, n_d=8, n_a=8, n_steps=2)
            x = torch.randn(2, dim)
            probs, _ = model(x)
            assert probs.shape == (2,)

    def test_tabnet_inference_function(self):
        from src.models.dataset_3_tabnet.model import TabNet
        from src.models.dataset_3_tabnet.inference import predict_batch

        model = TabNet(input_dim=8, n_d=8, n_a=8, n_steps=2)
        X = np.random.randn(10, 8).astype(np.float32)
        probs = predict_batch(model, X, device="cpu")
        assert probs.shape == (10,)
        assert (probs >= 0).all() and (probs <= 1).all()
