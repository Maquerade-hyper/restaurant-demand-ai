
"""
Part 28 - Continuous Learning

Safe continuous-learning infrastructure for the restaurant demand
forecasting system.

The package is deliberately model-agnostic at the orchestration layer.
A candidate model is only eligible for promotion after empirical
validation against the current champion.
"""

from .schemas import (
    DriftReport,
    RetrainingDecision,
    CandidateEvaluation,
    PromotionDecision,
    LearningRunResult,
)

from .drift import DriftMonitor
from .triggers import RetrainingTrigger
from .candidate import CandidateTrainer
from .registry import ModelRegistry
from .service import ContinuousLearningService

__all__ = [
    "DriftReport",
    "RetrainingDecision",
    "CandidateEvaluation",
    "PromotionDecision",
    "LearningRunResult",
    "DriftMonitor",
    "RetrainingTrigger",
    "CandidateTrainer",
    "ModelRegistry",
    "ContinuousLearningService",
]