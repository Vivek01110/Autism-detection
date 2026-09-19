"""
ASD Classification Package:
Exposes models, evaluation scripts, decision fusion, and standardized output interfaces
for Dataset 2 (Eye Tracking) and Dataset 3 (Clinical / Tabular).
"""

from .classification_interface import (
    ASDClassificationResult,
    StandardizedASDClassifier,
    standardize_classification_output,
    set_seed,
)

__all__ = [
    "ASDClassificationResult",
    "StandardizedASDClassifier",
    "standardize_classification_output",
    "set_seed",
]
