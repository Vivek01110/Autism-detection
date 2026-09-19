"""
Unit tests for the Standardized ASD Classification Interface.
Verifies schema compliance, validation rules, error handling, participant ID isolation,
reproducibility, and end-to-end model inference.
"""

import os
import sys
import json
import pytest
import numpy as np
import torch

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classification.classification_interface import (
    ASDClassificationResult,
    StandardizedASDClassifier,
    standardize_classification_output,
    set_seed,
)


class TestASDClassificationResultSchema:
    """Test schema validation and serialization of ASDClassificationResult."""

    def test_valid_instantiation(self):
        res = ASDClassificationResult(
            predicted_label=1,
            asd_probability=0.8245,
            model="kan",
            dataset="dataset_3",
            split="test",
            metadata={"age": 24, "gender": "Male"},
        )
        assert res.predicted_label == 1
        assert res.asd_probability == 0.8245
        assert res.model == "kan"
        assert res.dataset == "dataset_3"
        assert res.split == "test"
        assert res.metadata["age"] == 24

    def test_label_zero_td_valid(self):
        res = ASDClassificationResult(
            predicted_label=0,
            asd_probability=0.15,
            model="mamba",
            dataset="dataset_2",
            split="val",
        )
        assert res.predicted_label == 0
        assert res.asd_probability == 0.15

    def test_rejection_of_invalid_labels(self):
        """Reject labels outside {0, 1}."""
        for bad_label in [2, -1, 99, "ASD", 0.5]:
            with pytest.raises(ValueError):
                ASDClassificationResult(
                    predicted_label=bad_label,
                    asd_probability=0.7,
                    model="kan",
                    dataset="dataset_3",
                    split="test",
                )

    def test_rejection_of_invalid_probabilities(self):
        """Reject probabilities outside [0.0, 1.0], NaN, or infinite."""
        for bad_prob in [-0.01, 1.01, -1.0, 2.0, float("nan"), float("inf")]:
            with pytest.raises(ValueError):
                ASDClassificationResult(
                    predicted_label=1,
                    asd_probability=bad_prob,
                    model="patchtst",
                    dataset="dataset_2",
                    split="test",
                )

    def test_rejection_of_invalid_strings(self):
        """Reject empty or non-string model/dataset/split names."""
        with pytest.raises(ValueError):
            ASDClassificationResult(
                predicted_label=1,
                asd_probability=0.5,
                model="",
                dataset="dataset_3",
                split="test",
            )
        with pytest.raises(ValueError):
            ASDClassificationResult(
                predicted_label=1,
                asd_probability=0.5,
                model="kan",
                dataset="   ",
                split="test",
            )

    def test_to_dict_and_to_json(self):
        """Verify dictionary and json serialization."""
        res = ASDClassificationResult(
            predicted_label=1,
            asd_probability=0.75,
            model="tabnet",
            dataset="dataset_3",
            split="inference",
            metadata={"notes": "screening"},
        )
        d = res.to_dict()
        assert d["predicted_label"] == 1
        assert d["asd_probability"] == 0.75
        assert d["model"] == "tabnet"
        assert d["dataset"] == "dataset_3"
        assert d["split"] == "inference"
        assert d["metadata"] == {"notes": "screening"}

        json_str = res.to_json()
        parsed = json.loads(json_str)
        assert parsed == d

    def test_from_dict_reconstruction(self):
        """Verify from_dict correctly parses and validates a dictionary."""
        payload = {
            "predicted_label": 0,
            "asd_probability": 0.22,
            "model": "mamba",
            "dataset": "dataset_2",
            "split": "test",
            "metadata": {"trial": 1},
        }
        res = ASDClassificationResult.from_dict(payload)
        assert res.predicted_label == 0
        assert res.asd_probability == 0.22
        assert res.model == "mamba"

        # Missing key should raise KeyError
        bad_payload = dict(payload)
        del bad_payload["asd_probability"]
        with pytest.raises(KeyError):
            ASDClassificationResult.from_dict(bad_payload)

    def test_convenience_factory(self):
        """Verify standardize_classification_output convenience function."""
        res = standardize_classification_output(
            predicted_label=1,
            asd_probability=0.9,
            model="decision_fusion",
            dataset="decision_fusion",
            split="test",
            metadata={"alpha": 0.9},
        )
        assert isinstance(res, ASDClassificationResult)
        assert res.predicted_label == 1
        assert res.metadata["alpha"] == 0.9


class TestParticipantIdIsolationAndLeakagePrevention:
    """Ensure participant IDs are never fed as features to models."""

    def test_d3_strips_child_id_from_features(self):
        classifier = StandardizedASDClassifier(config_path="configs/config.yaml")

        sample_with_id = {
            "Child_ID": "CH-9999",
            "Age": 24,
            "Social_Interaction_Score": 4.0,
            "Communication_Score": 3.0,
            "Repetitive_Behavior_Score": 2.0,
            "Gender": "Male",
            "Jaundice": "No",
            "Family_ASD_History": "Yes",
            "Language_Delay": "No",
        }

        # Vector preparation must extract exactly 8 features and isolate Child_ID into metadata
        vec, extracted_meta = classifier._prepare_d3_vector(sample_with_id)
        assert vec.shape == (1, 8)
        assert "participant_id" in extracted_meta
        assert extracted_meta["participant_id"] == "CH-9999"
        # The input dict should not have corrupted the vector shape
        assert np.isfinite(vec).all()

    def test_d2_isolates_participant_id_in_metadata(self):
        classifier = StandardizedASDClassifier(config_path="configs/config.yaml")
        seq = np.random.randn(200, 8).astype(np.float32)

        result = classifier.classify_dataset2_sample(
            sequence=seq,
            model_name="mamba",
            participant_id=42,
        )
        assert result.metadata["participant_id"] == "42"
        # Underlying model received exactly (1, 200, 8) without ID channel
        assert classifier._prepare_d2_tensor(seq).shape == (1, 200, 8)


class TestDecisionFusionStandardization:
    """Test decision-level integration method."""

    def test_decision_fusion_output(self):
        classifier = StandardizedASDClassifier(config_path="configs/config.yaml")
        res = classifier.classify_decision_fusion(
            p_clinical=0.8,
            p_eye=0.4,
            clin_model="kan",
            eye_model="mamba",
            alpha=0.5,
            threshold=0.5,
        )
        assert isinstance(res, ASDClassificationResult)
        assert res.dataset == "decision_fusion"
        assert res.model == "kan_mamba_fusion"
        # Fused prob = 0.5 * 0.8 + 0.5 * 0.4 = 0.60
        assert pytest.approx(res.asd_probability, abs=1e-4) == 0.60
        assert res.predicted_label == 1
        assert res.metadata["fusion_alpha"] == 0.5
        assert res.metadata["p_clinical"] == 0.8
        assert res.metadata["p_eye"] == 0.4

    def test_decision_fusion_bounds_enforced(self):
        classifier = StandardizedASDClassifier(config_path="configs/config.yaml")
        with pytest.raises(ValueError):
            classifier.classify_decision_fusion(p_clinical=0.8, p_eye=0.4, alpha=1.5)
        with pytest.raises(ValueError):
            classifier.classify_decision_fusion(p_clinical=0.8, p_eye=0.4, threshold=-0.1)


@pytest.fixture(scope="module")
def classifier():
    return StandardizedASDClassifier(config_path="configs/config.yaml")


class TestEndToEndModelInferenceWithCheckpoints:
    """Test full inference with saved model checkpoints."""

    def test_dataset3_kan_inference(self, classifier):
        # 8 preprocessed feature values
        sample_x = np.array([0.5, -0.2, 0.8, -0.1, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        res = classifier.classify_dataset3_sample(sample_x, model_name="kan")
        assert isinstance(res, ASDClassificationResult)
        assert res.dataset == "dataset_3"
        assert res.model == "kan"
        assert 0.0 <= res.asd_probability <= 1.0
        assert res.predicted_label in (0, 1)

    def test_dataset3_tabnet_inference(self, classifier):
        sample_x = np.array([0.5, -0.2, 0.8, -0.1, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        res = classifier.classify_dataset3_sample(sample_x, model_name="tabnet")
        assert isinstance(res, ASDClassificationResult)
        assert res.dataset == "dataset_3"
        assert res.model == "tabnet"
        assert 0.0 <= res.asd_probability <= 1.0
        assert res.predicted_label in (0, 1)

    def test_dataset2_mamba_inference(self, classifier):
        seq = np.random.randn(200, 8).astype(np.float32)
        res = classifier.classify_dataset2_sample(seq, model_name="mamba", participant_id=10)
        assert isinstance(res, ASDClassificationResult)
        assert res.dataset == "dataset_2"
        assert res.model == "mamba"
        assert 0.0 <= res.asd_probability <= 1.0
        assert res.predicted_label in (0, 1)
        assert res.metadata["participant_id"] == "10"

    def test_dataset2_patchtst_inference(self, classifier):
        seq = np.random.randn(200, 8).astype(np.float32)
        res = classifier.classify_dataset2_sample(seq, model_name="patchtst", participant_id=17)
        assert isinstance(res, ASDClassificationResult)
        assert res.dataset == "dataset_2"
        assert res.model == "patchtst"
        assert 0.0 <= res.asd_probability <= 1.0
        assert res.predicted_label in (0, 1)

    def test_batch_inference_d3_and_d2(self, classifier):
        # D3 batch of 5 records
        batch_d3 = np.random.randn(5, 8).astype(np.float32)
        res_d3 = classifier.classify_dataset3_batch(batch_d3, model_name="kan")
        assert len(res_d3) == 5
        for r in res_d3:
            assert isinstance(r, ASDClassificationResult)
            assert 0.0 <= r.asd_probability <= 1.0
            assert r.predicted_label in (0, 1)

        # D2 batch of 3 sequences
        batch_d2 = np.random.randn(3, 200, 8).astype(np.float32)
        res_d2 = classifier.classify_dataset2_batch(
            batch_d2, model_name="mamba", participant_ids=[10, 11, 12]
        )
        assert len(res_d2) == 3
        for i, r in enumerate(res_d2):
            assert isinstance(r, ASDClassificationResult)
            assert 0.0 <= r.asd_probability <= 1.0
            assert r.predicted_label in (0, 1)
            assert r.metadata["participant_id"] == str([10, 11, 12][i])

    def test_reproducibility_with_seed(self, classifier):
        """Identical seed and inputs yield identical predictions."""
        set_seed(42)
        seq = np.random.randn(200, 8).astype(np.float32)
        res1 = classifier.classify_dataset2_sample(seq, model_name="mamba")

        set_seed(42)
        seq_copy = seq.copy()
        res2 = classifier.classify_dataset2_sample(seq_copy, model_name="mamba")

        assert res1.asd_probability == res2.asd_probability
        assert res1.predicted_label == res2.predicted_label
