from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelValidationScore:
    model_name: str
    mae: float
    rmse: float
    bias: float = 0.0
    stability: float = 1.0


@dataclass
class HybridDecision:
    selected_model: str
    weights: Dict[str, float]
    confidence: float
    reason: str


@dataclass
class HybridForecast:
    predictions: List[float]
    selected_model: str
    weights: Dict[str, float]
    confidence: float