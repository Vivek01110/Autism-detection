"""
Standardized ASD Classification Interface:
Provides a unified, contract-compliant interface for Dataset 2 (Eye-Tracking),
Dataset 3 (Clinical / Tabular), and Decision-Level Fusion models.

Standard Output Schema:
-----------------------
{
    "predicted_label": int (0 for TD / Non-ASD, 1 for ASD),
    "asd_probability": float in [0.0, 1.0],
    "model": str ("mamba" | "patchtst" | "kan" | "tabnet" | "decision_fusion"),
    "dataset": str ("dataset_2" | "dataset_3" | "decision_fusion"),
    "split": str ("train" | "val" | "test" | "inference"),
    "metadata": dict (optional demographic/trial/policy information)
}

Design & Safety Guarantees:
----------------------------
1. Strict Validation:
   - asd_probability is bounded strictly to [0.0, 1.0].
   - predicted_label is restricted strictly to {0, 1}.
2. Zero ID / Leakage:
   - ParticipantID / Child_ID are explicitly stripped from model feature inputs.
   - Demographics/IDs are stored strictly in metadata for auditability.
3. Determinism:
   - Predictable, reproducible inference.
4. Scientific Disclaimer:
   - Dataset 2 and Dataset 3 are unpaired cohorts. Decision-level integration
     is an experimental policy simulation, not a clinical diagnostic system.
"""

import os
import sys
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import joblib
import torch

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_preprocessing.dataset3_preprocessing import load_config
from src.classification.dataset23_decision_fusion import compute_fused_score, get_device


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ASDClassificationResult:
    """
    Standardized result contract for all ASD classification outputs.
    """
    predicted_label: int
    asd_probability: float
    model: str
    dataset: str
    split: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 1. Validate predicted_label
        if not isinstance(self.predicted_label, (int, np.integer)):
            if isinstance(self.predicted_label, (float, np.floating)) and self.predicted_label in (0.0, 1.0):
                self.predicted_label = int(self.predicted_label)
            else:
                raise ValueError(
                    f"predicted_label must be an integer (0 or 1), got {type(self.predicted_label)}: {self.predicted_label}"
                )
        self.predicted_label = int(self.predicted_label)
        if self.predicted_label not in (0, 1):
            raise ValueError(
                f"predicted_label must be 0 (TD / Non-ASD) or 1 (ASD), got {self.predicted_label}"
            )

        # 2. Validate asd_probability
        if not isinstance(self.asd_probability, (float, np.floating, int)):
            raise ValueError(
                f"asd_probability must be a float, got {type(self.asd_probability)}"
            )
        self.asd_probability = float(self.asd_probability)
        if np.isnan(self.asd_probability) or np.isinf(self.asd_probability):
            raise ValueError(
                f"asd_probability cannot be NaN or Inf, got {self.asd_probability}"
            )
        if not (0.0 <= self.asd_probability <= 1.0):
            raise ValueError(
                f"asd_probability must be in [0.0, 1.0], got {self.asd_probability}"
            )

        # 3. Validate string fields
        for field_name in ["model", "dataset", "split"]:
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"'{field_name}' must be a non-empty string, got: {val!r}")

        # 4. Validate metadata
        if not isinstance(self.metadata, dict):
            raise ValueError(f"'metadata' must be a dictionary, got: {type(self.metadata)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert classification result to standard dictionary."""
        return {
            "predicted_label": int(self.predicted_label),
            "asd_probability": float(self.asd_probability),
            "model": str(self.model),
            "dataset": str(self.dataset),
            "split": str(self.split),
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize classification result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ASDClassificationResult":
        """Reconstruct classification result from dictionary."""
        required_keys = ["predicted_label", "asd_probability", "model", "dataset", "split"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing required key in classification result: '{key}'")
        return cls(
            predicted_label=data["predicted_label"],
            asd_probability=data["asd_probability"],
            model=data["model"],
            dataset=data["dataset"],
            split=data["split"],
            metadata=data.get("metadata", {}),
        )


class StandardizedASDClassifier:
    """
    Unified classifier interface wrapping Dataset 2, Dataset 3, and Decision Fusion models.
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        device_override: Optional[str] = None,
        lazy_load: bool = True,
    ):
        self.config = load_config(config_path) if os.path.isfile(config_path) else {}
        self.device = (
            torch.device(device_override)
            if device_override
            else get_device(self.config.get("device", "auto"))
        )

        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, Any] = {}
        self._metadata_configs: Dict[str, Any] = {}

        # Default optimal decision fusion hyper-parameters (derived from validation tuning)
        # KAN + Mamba: alpha=0.9, threshold=0.23
        self.default_fusion_params = {
            "kan_mamba": {"alpha": 0.9, "threshold": 0.23},
            "kan_patchtst": {"alpha": 1.0, "threshold": 0.24},
            "tabnet_mamba": {"alpha": 0.9, "threshold": 0.27},
            "tabnet_patchtst": {"alpha": 1.0, "threshold": 0.30},
        }

        # Load scalers and metadata
        self._load_preprocessing_artifacts()

        if not lazy_load:
            self.load_all_models()

    def _load_preprocessing_artifacts(self) -> None:
        """Load pre-fitted scalers and feature metadata."""
        artifacts_dir = self.config.get("paths", {}).get(
            "preprocessing_artifacts", "results/preprocessing_artifacts"
        )
        # Dataset 3
        d3_scaler_path = os.path.join(artifacts_dir, "dataset3_scaler.joblib")
        if os.path.isfile(d3_scaler_path):
            self._scalers["dataset3"] = joblib.load(d3_scaler_path)

        d3_meta_path = os.path.join(artifacts_dir, "dataset3_metadata.json")
        if os.path.isfile(d3_meta_path):
            with open(d3_meta_path, "r") as f:
                self._metadata_configs["dataset3"] = json.load(f)

        # Dataset 2
        d2_scaler_path = os.path.join(artifacts_dir, "dataset2_scaler.joblib")
        if os.path.isfile(d2_scaler_path):
            self._scalers["dataset2"] = joblib.load(d2_scaler_path)

        d2_meta_path = os.path.join(artifacts_dir, "dataset2_metadata.json")
        if os.path.isfile(d2_meta_path):
            with open(d2_meta_path, "r") as f:
                self._metadata_configs["dataset2"] = json.load(f)

    def load_model(self, model_name: str, checkpoint_path: Optional[str] = None) -> Any:
        """Load and cache a specific model."""
        model_name = model_name.lower().strip()
        if model_name in self._models:
            return self._models[model_name]

        paths = self.config.get("paths", {})
        ckpt_dir = paths.get("checkpoints", "results/checkpoints")

        if model_name == "kan":
            from src.models.dataset_3_kan.inference import load_kan_model
            ckpt = checkpoint_path or os.path.join(ckpt_dir, "kan_best.pt")
            input_dim = self._metadata_configs.get("dataset3", {}).get("input_dim", 8)
            model = load_kan_model(ckpt, input_dim=input_dim, config=self.config)
            self._models["kan"] = model
            return model

        elif model_name == "tabnet":
            from src.models.dataset_3_tabnet.inference import load_tabnet_model
            ckpt = checkpoint_path or os.path.join(ckpt_dir, "tabnet_best.pt")
            input_dim = self._metadata_configs.get("dataset3", {}).get("input_dim", 8)
            model = load_tabnet_model(ckpt, input_dim=input_dim, config=self.config)
            self._models["tabnet"] = model
            return model

        elif model_name == "mamba":
            from src.models.dataset_2_mamba.inference import load_mamba_model
            ckpt = checkpoint_path or os.path.join(ckpt_dir, "mamba_best.pt")
            input_dim = self._metadata_configs.get("dataset2", {}).get("num_features", 8)
            model = load_mamba_model(ckpt, config=self.config, device=self.device, input_dim=input_dim)
            self._models["mamba"] = model
            return model

        elif model_name == "patchtst":
            from src.models.dataset_2_patchtst.inference import load_patchtst_model
            ckpt = checkpoint_path or os.path.join(ckpt_dir, "patchtst_best.pt")
            input_dim = self._metadata_configs.get("dataset2", {}).get("num_features", 8)
            seq_len = self._metadata_configs.get("dataset2", {}).get("sequence_length", 200)
            model = load_patchtst_model(
                ckpt, config=self.config, device=self.device, input_dim=input_dim, seq_len=seq_len
            )
            self._models["patchtst"] = model
            return model

        else:
            raise ValueError(f"Unsupported model name: '{model_name}'. Choose from: kan, tabnet, mamba, patchtst.")

    def load_all_models(self) -> None:
        """Load all supported models."""
        for m in ["kan", "tabnet", "mamba", "patchtst"]:
            try:
                self.load_model(m)
            except Exception as e:
                print(f"[Warning] Could not load model '{m}': {e}")

    # =========================================================================
    # Dataset 3 (Clinical / Tabular) Inference
    # =========================================================================

    def _prepare_d3_vector(
        self, features: Union[Dict[str, Any], np.ndarray, List[float]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract strictly the 8 clinical features and guarantee NO identifier leakage.
        Returns:
            (preprocessed_vector_1x8, extracted_metadata)
        """
        extracted_meta = {}
        if isinstance(features, dict):
            feat_dict = dict(features)
            # Remove any identifiers and store in metadata
            for id_key in ["Child_ID", "ParticipantID", "ID", "id", "child_id", "participant_id"]:
                if id_key in feat_dict:
                    extracted_meta["participant_id"] = str(feat_dict.pop(id_key))

            # Remove ground truth if present
            for target_key in ["Diagnosed_ASD", "target", "label", "ASD", "class"]:
                if target_key in feat_dict:
                    extracted_meta["ground_truth"] = feat_dict.pop(target_key)

            d3_meta = self._metadata_configs.get("dataset3", {})
            feature_order = d3_meta.get(
                "feature_names",
                [
                    "Age",
                    "Social_Interaction_Score",
                    "Communication_Score",
                    "Repetitive_Behavior_Score",
                    "Gender",
                    "Jaundice",
                    "Family_ASD_History",
                    "Language_Delay",
                ],
            )
            cat_mappings = d3_meta.get("categorical_mappings", {
                "Gender": {"Male": 1, "Female": 0},
                "Jaundice": {"Yes": 1, "No": 0},
                "Family_ASD_History": {"Yes": 1, "No": 0},
                "Language_Delay": {"Yes": 1, "No": 0},
            })

            vec = []
            for col in feature_order:
                if col not in feat_dict:
                    raise KeyError(f"Missing required feature for Dataset 3: '{col}'")
                val = feat_dict[col]
                if col in cat_mappings and isinstance(val, str):
                    if val not in cat_mappings[col]:
                        raise ValueError(f"Invalid category for '{col}': {val}")
                    val = cat_mappings[col][val]
                vec.append(float(val))

            arr = np.array(vec, dtype=np.float32).reshape(1, -1)
            # Scale numerical features (first 4)
            scaler = self._scalers.get("dataset3")
            if scaler is not None:
                arr[:, :4] = scaler.transform(arr[:, :4])
            return arr, extracted_meta

        elif isinstance(features, (list, np.ndarray)):
            arr = np.asarray(features, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != 8:
                raise ValueError(
                    f"Dataset 3 features must have exactly 8 dimensions, got shape {arr.shape}"
                )
            return arr, extracted_meta
        else:
            raise TypeError(f"Unsupported features type: {type(features)}")

    def classify_dataset3_sample(
        self,
        features: Union[Dict[str, Any], np.ndarray, List[float]],
        model_name: str = "kan",
        threshold: float = 0.5,
        split: str = "inference",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ASDClassificationResult:
        """
        Classify a single Dataset 3 clinical/tabular record.
        """
        model_name = model_name.lower().strip()
        if model_name not in ["kan", "tabnet"]:
            raise ValueError(f"Model '{model_name}' is not a Dataset 3 model (use 'kan' or 'tabnet').")

        model = self.load_model(model_name)
        X, auto_meta = self._prepare_d3_vector(features)

        merged_meta = dict(auto_meta)
        if metadata:
            merged_meta.update(metadata)
        merged_meta["decision_threshold"] = float(threshold)

        if model_name == "kan":
            from src.models.dataset_3_kan.inference import predict_batch
            probs = predict_batch(model, X, self.device)
        else:
            from src.models.dataset_3_tabnet.inference import predict_batch
            probs = predict_batch(model, X, self.device)

        prob = float(np.clip(probs[0], 0.0, 1.0))
        label = 1 if prob >= threshold else 0

        return ASDClassificationResult(
            predicted_label=label,
            asd_probability=prob,
            model=model_name,
            dataset="dataset_3",
            split=split,
            metadata=merged_meta,
        )

    def classify_dataset3_batch(
        self,
        X: np.ndarray,
        model_name: str = "kan",
        threshold: float = 0.5,
        split: str = "inference",
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ASDClassificationResult]:
        """
        Classify a batch of preprocessed Dataset 3 records of shape (N, 8).
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != 8:
            raise ValueError(f"Batch X must have shape (N, 8), got {X.shape}")

        model_name = model_name.lower().strip()
        model = self.load_model(model_name)

        if model_name == "kan":
            from src.models.dataset_3_kan.inference import predict_batch
            probs = predict_batch(model, X, self.device)
        elif model_name == "tabnet":
            from src.models.dataset_3_tabnet.inference import predict_batch
            probs = predict_batch(model, X, self.device)
        else:
            raise ValueError(f"Unsupported Dataset 3 model: {model_name}")

        results = []
        for i, p in enumerate(probs):
            prob = float(np.clip(p, 0.0, 1.0))
            label = 1 if prob >= threshold else 0
            meta = dict(metadata_list[i]) if metadata_list and i < len(metadata_list) else {}
            meta["decision_threshold"] = float(threshold)
            results.append(
                ASDClassificationResult(
                    predicted_label=label,
                    asd_probability=prob,
                    model=model_name,
                    dataset="dataset_3",
                    split=split,
                    metadata=meta,
                )
            )
        return results

    # =========================================================================
    # Dataset 2 (Eye Tracking) Inference
    # =========================================================================

    def _prepare_d2_tensor(
        self, sequence: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Validate and format eye-tracking sequence of shape (1, 200, 8).
        """
        if isinstance(sequence, torch.Tensor):
            arr = sequence.detach().cpu().numpy()
        else:
            arr = np.asarray(sequence, dtype=np.float32)

        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)  # (1, 200, 8)

        if arr.ndim != 3 or arr.shape[1] != 200 or arr.shape[2] != 8:
            raise ValueError(
                f"Dataset 2 sequence must have shape (200, 8) or (B, 200, 8), got {arr.shape}"
            )
        return arr

    def classify_dataset2_sample(
        self,
        sequence: Union[np.ndarray, torch.Tensor],
        model_name: str = "mamba",
        threshold: float = 0.5,
        split: str = "inference",
        participant_id: Optional[Union[int, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ASDClassificationResult:
        """
        Classify a single Dataset 2 eye-tracking sequence of shape (200, 8).
        """
        model_name = model_name.lower().strip()
        if model_name not in ["mamba", "patchtst"]:
            raise ValueError(f"Model '{model_name}' is not a Dataset 2 model (use 'mamba' or 'patchtst').")

        model = self.load_model(model_name)
        X = self._prepare_d2_tensor(sequence)

        merged_meta = dict(metadata or {})
        if participant_id is not None:
            # Participant ID is strictly isolated in metadata, NEVER passed to neural net
            merged_meta["participant_id"] = str(participant_id)
        merged_meta["decision_threshold"] = float(threshold)

        if model_name == "mamba":
            from src.models.dataset_2_mamba.inference import predict_batch
            probs = predict_batch(model, X, self.device, batch_size=1)
        else:
            from src.models.dataset_2_patchtst.inference import predict_batch
            probs = predict_batch(model, X, self.device, batch_size=1)

        prob = float(np.clip(probs[0], 0.0, 1.0))
        label = 1 if prob >= threshold else 0

        return ASDClassificationResult(
            predicted_label=label,
            asd_probability=prob,
            model=model_name,
            dataset="dataset_2",
            split=split,
            metadata=merged_meta,
        )

    def classify_dataset2_batch(
        self,
        sequences: Union[np.ndarray, torch.Tensor],
        model_name: str = "mamba",
        threshold: float = 0.5,
        split: str = "inference",
        participant_ids: Optional[List[Union[int, str]]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ASDClassificationResult]:
        """
        Classify a batch of Dataset 2 eye-tracking sequences of shape (N, 200, 8).
        """
        model_name = model_name.lower().strip()
        model = self.load_model(model_name)
        X = self._prepare_d2_tensor(sequences)

        if model_name == "mamba":
            from src.models.dataset_2_mamba.inference import predict_batch
            probs = predict_batch(model, X, self.device, batch_size=64)
        elif model_name == "patchtst":
            from src.models.dataset_2_patchtst.inference import predict_batch
            probs = predict_batch(model, X, self.device, batch_size=64)
        else:
            raise ValueError(f"Unsupported Dataset 2 model: {model_name}")

        results = []
        for i, p in enumerate(probs):
            prob = float(np.clip(p, 0.0, 1.0))
            label = 1 if prob >= threshold else 0
            meta = dict(metadata_list[i]) if metadata_list and i < len(metadata_list) else {}
            if participant_ids and i < len(participant_ids):
                meta["participant_id"] = str(participant_ids[i])
            meta["decision_threshold"] = float(threshold)
            results.append(
                ASDClassificationResult(
                    predicted_label=label,
                    asd_probability=prob,
                    model=model_name,
                    dataset="dataset_2",
                    split=split,
                    metadata=meta,
                )
            )
        return results

    # =========================================================================
    # Decision-Level Fusion Inference
    # =========================================================================

    def classify_decision_fusion(
        self,
        p_clinical: float,
        p_eye: float,
        clin_model: str = "kan",
        eye_model: str = "mamba",
        alpha: Optional[float] = None,
        threshold: Optional[float] = None,
        split: str = "inference",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ASDClassificationResult:
        """
        Perform decision-level integration between clinical and eye-tracking probabilities:
            P_combined = alpha * P_clinical + (1 - alpha) * P_eye

        Uses optimal parameters tuned strictly on validation simulation data if not provided.
        """
        clin_model = clin_model.lower().strip()
        eye_model = eye_model.lower().strip()
        combo_key = f"{clin_model}_{eye_model}"

        # Resolve alpha and threshold
        defaults = self.default_fusion_params.get(combo_key, {"alpha": 0.5, "threshold": 0.5})
        selected_alpha = defaults["alpha"] if alpha is None else float(alpha)
        selected_threshold = defaults["threshold"] if threshold is None else float(threshold)

        if not 0.0 <= selected_alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {selected_alpha}")
        if not 0.0 <= selected_threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {selected_threshold}")

        p_clin_arr = np.array([p_clinical], dtype=np.float32)
        p_eye_arr = np.array([p_eye], dtype=np.float32)
        fused_score = float(compute_fused_score(p_clin_arr, p_eye_arr, selected_alpha)[0])
        fused_prob = float(np.clip(fused_score, 0.0, 1.0))
        label = 1 if fused_prob >= selected_threshold else 0

        merged_meta = dict(metadata or {})
        merged_meta.update({
            "clinical_model": clin_model,
            "eye_tracking_model": eye_model,
            "p_clinical": float(p_clinical),
            "p_eye": float(p_eye),
            "fusion_alpha": selected_alpha,
            "decision_threshold": selected_threshold,
            "scientific_note": "Decision-level fusion from independent modality specialist scores.",
        })

        return ASDClassificationResult(
            predicted_label=label,
            asd_probability=fused_prob,
            model=f"{clin_model}_{eye_model}_fusion",
            dataset="decision_fusion",
            split=split,
            metadata=merged_meta,
        )


def standardize_classification_output(
    predicted_label: int,
    asd_probability: float,
    model: str,
    dataset: str,
    split: str = "inference",
    metadata: Optional[Dict[str, Any]] = None,
) -> ASDClassificationResult:
    """
    Convenience factory to generate and validate a contract-conforming ASDClassificationResult.
    """
    return ASDClassificationResult(
        predicted_label=predicted_label,
        asd_probability=asd_probability,
        model=model,
        dataset=dataset,
        split=split,
        metadata=metadata or {},
    )
