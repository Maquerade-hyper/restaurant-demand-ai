from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DriftReport:
    feature: str
    reference_mean: float
    current_mean: float
    mean_shift: float
    psi: float
    drifted: bool


@dataclass
class RetrainingDecision:
    should_retrain: bool
    reasons: List[str] = field(default_factory=list)
    drifted_features: List[str] = field(default_factory=list)
    error_degradation: float = 0.0


@dataclass
class CandidateEvaluation:
    model_name: str
    champion_mae: float
    candidate_mae: float
    champion_rmse: float
    candidate_rmse: float
    champion_bias: float
    candidate_bias: float
    improvement_pct: float
    bias_change: float
    candidate_stable: bool
    passes: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class PromotionDecision:
    action: str
    model_name: Optional[str]
    reason: str
    previous_champion: Optional[str] = None


@dataclass
class LearningRunResult:
    drift: List[DriftReport]
    retraining: RetrainingDecision
    candidate: Optional[CandidateEvaluation]
    promotion: PromotionDecision
    run_id: str
    artifact_path: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "drift": [
                {
                    "feature": item.feature,
                    "reference_mean": item.reference_mean,
                    "current_mean": item.current_mean,
                    "mean_shift": item.mean_shift,
                    "psi": item.psi,
                    "drifted": item.drifted,
                }
                for item in self.drift
            ],
            "retraining": {
                "should_retrain": self.retraining.should_retrain,
                "reasons": self.retraining.reasons,
                "drifted_features": self.retraining.drifted_features,
                "error_degradation": self.retraining.error_degradation,
            },
            "candidate": (
                None
                if self.candidate is None
                else {
                    "model_name": self.candidate.model_name,
                    "champion_mae": self.candidate.champion_mae,
                    "candidate_mae": self.candidate.candidate_mae,
                    "champion_rmse": self.candidate.champion_rmse,
                    "candidate_rmse": self.candidate.candidate_rmse,
                    "champion_bias": self.candidate.champion_bias,
                    "candidate_bias": self.candidate.candidate_bias,
                    "improvement_pct": self.candidate.improvement_pct,
                    "bias_change": self.candidate.bias_change,
                    "candidate_stable": self.candidate.candidate_stable,
                    "passes": self.candidate.passes,
                    "reasons": self.candidate.reasons,
                }
            ),
            "promotion": {
                "action": self.promotion.action,
                "model_name": self.promotion.model_name,
                "reason": self.promotion.reason,
                "previous_champion": self.promotion.previous_champion,
            },
            "artifact_path": self.artifact_path,
        }