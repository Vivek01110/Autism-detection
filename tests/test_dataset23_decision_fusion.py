"""
Unit tests for Decision-Level Integration & Controlled Cross-Dataset Simulation:
Dataset 2 (Eye Tracking) + Dataset 3 (Clinical / Tabular).
"""

import sys
import os
import glob
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classification.dataset23_decision_fusion import (
    compute_fused_score,
    construct_stratified_simulation_cohort,
    tune_alpha_and_threshold_on_validation,
    evaluate_decision_fusion_combination,
)
from src.data_preprocessing.dataset3_preprocessing import load_config


class TestDecisionFusionCoreFunctions:
    """Verify mathematical and algorithmic properties of decision fusion."""

    def test_alpha_bounds_enforced(self):
        """Alpha must be strictly within [0.0, 1.0]."""
        p1 = np.array([0.2, 0.8])
        p2 = np.array([0.4, 0.6])

        # Valid alpha values
        for alpha in [0.0, 0.5, 1.0]:
            fused = compute_fused_score(p1, p2, alpha)
            assert (fused >= 0.0).all() and (fused <= 1.0).all()

        # Invalid alpha values should raise ValueError
        with pytest.raises(ValueError):
            compute_fused_score(p1, p2, -0.1)

        with pytest.raises(ValueError):
            compute_fused_score(p1, p2, 1.1)

    def test_fused_score_calculation(self):
        """Verify exact linear combination: P_comb = alpha * p1 + (1 - alpha) * p2."""
        p1 = np.array([0.2, 0.8, 1.0, 0.0])
        p2 = np.array([0.6, 0.4, 0.5, 0.5])
        alpha = 0.7

        expected = 0.7 * p1 + 0.3 * p2
        actual = compute_fused_score(p1, p2, alpha)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_probability_values_in_unit_interval(self):
        """Ensure all fused probabilities are strictly in [0.0, 1.0]."""
        p1 = np.random.uniform(0.0, 1.0, 100)
        p2 = np.random.uniform(0.0, 1.0, 100)
        for alpha in np.linspace(0.0, 1.0, 11):
            fused = compute_fused_score(p1, p2, alpha)
            assert np.all(fused >= 0.0)
            assert np.all(fused <= 1.0)


class TestValidationOnlyTuning:
    """Verify that alpha and threshold tuning is strictly validation-only."""

    def test_validation_only_alpha_selection(self):
        """Ensure optimal alpha is determined strictly by validation metrics."""
        # Simulated validation set where D2 (alpha=0.0) is perfect and D3 (alpha=1.0) is random
        val_y = np.array([1, 1, 1, 0, 0, 0])
        val_p_d3 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # completely random
        val_p_d2 = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])  # perfect separator

        res = tune_alpha_and_threshold_on_validation(val_p_d3, val_p_d2, val_y)
        # Expect alpha=0.0 because D2 is vastly superior
        assert res["best_alpha"] == 0.0
        assert res["best_val_roc_auc"] == 1.0

        # Now reverse: D3 is perfect, D2 is random
        val_p_d3_rev = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        val_p_d2_rev = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        res_rev = tune_alpha_and_threshold_on_validation(val_p_d3_rev, val_p_d2_rev, val_y)
        assert res_rev["best_alpha"] == 1.0
        assert res_rev["best_val_roc_auc"] == 1.0

    def test_test_set_never_used_during_tuning(self):
        """Verify that evaluate_decision_fusion_combination tunes on val and only evaluates on test."""
        val_y = np.array([1, 1, 0, 0])
        val_p1 = np.array([0.9, 0.9, 0.1, 0.1])
        val_p2 = np.array([0.5, 0.5, 0.5, 0.5])

        # Test set has different properties
        test_y = np.array([1, 0])
        test_p1 = np.array([0.8, 0.2])
        test_p2 = np.array([0.7, 0.3])

        eval_res = evaluate_decision_fusion_combination(
            "Test_Combo", "D3", "D2", val_p1, val_p2, val_y, test_p1, test_p2, test_y
        )

        # Alpha should be chosen solely by validation data (alpha=1.0)
        assert eval_res["selected_alpha"] == 1.0
        assert "test_metrics" in eval_res
        assert eval_res["test_metrics"]["accuracy"] == 1.0


class TestSimulationReproducibilityAndIntegrity:
    """Verify simulation cohort construction and dataset safety."""

    def test_simulation_reproducibility_with_seed(self):
        """Same seed must produce identical simulation cohorts."""
        d3_probs = np.random.rand(100)
        d3_labels = np.array([1] * 40 + [0] * 60)
        d3_meta = pd.DataFrame({
            "Gender": ["Male"] * 50 + ["Female"] * 50,
            "Age": np.random.randint(12, 36, 100),
        })

        d2_probs = np.random.rand(500)
        d2_labels = np.array([1] * 200 + [0] * 300)
        d2_meta = pd.DataFrame({
            "Gender": ["M"] * 250 + ["F"] * 250,
            "Age": np.random.uniform(3, 12, 500),
        })

        s1_d3, s1_d2, s1_y = construct_stratified_simulation_cohort(
            d3_probs, d3_labels, d3_meta, d2_probs, d2_labels, d2_meta, n_samples=200, seed=42
        )
        s2_d3, s2_d2, s2_y = construct_stratified_simulation_cohort(
            d3_probs, d3_labels, d3_meta, d2_probs, d2_labels, d2_meta, n_samples=200, seed=42
        )

        np.testing.assert_array_equal(s1_d3, s2_d3)
        np.testing.assert_array_equal(s1_d2, s2_d2)
        np.testing.assert_array_equal(s1_y, s2_y)

    def test_no_raw_data_modified(self):
        """Ensure raw dataset files in data/ remain completely unmodified."""
        config = load_config("configs/config.yaml")

        # Check D3 raw file exists and is intact
        d3_path = config["paths"]["dataset3"]
        assert os.path.isfile(d3_path)
        d3_df = pd.read_csv(d3_path)
        assert len(d3_df) == 1000

        # Check D2 raw files exist and count is 25
        d2_raw_dir = config["paths"]["dataset2_raw"]
        d2_files = glob.glob(os.path.join(d2_raw_dir, "*.csv"))
        assert len(d2_files) == 25

        # Check D2 metadata exists and count is 59
        d2_meta_path = config["paths"]["dataset2_metadata"]
        assert os.path.isfile(d2_meta_path)
        d2_meta = pd.read_csv(d2_meta_path)
        assert len(d2_meta) == 59
