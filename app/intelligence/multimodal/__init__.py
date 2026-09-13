"""
Part 27 - Multimodal Intelligence.

Provides lightweight CPU-first processing for:
- textual context
- visual context
- multimodal feature fusion
- validation and ablation
"""

from .schemas import (
    TextContext,
    VisualContext,
    MultimodalFeatures,
)

from .text import TextIntelligence
from .vision import VisualIntelligence
from .fusion import MultimodalFusion
from .service import MultimodalIntelligenceService
from .benchmark import MultimodalBenchmark

__all__ = [
    "TextContext",
    "VisualContext",
    "MultimodalFeatures",
    "TextIntelligence",
    "VisualIntelligence",
    "MultimodalFusion",
    "MultimodalIntelligenceService",
    "MultimodalBenchmark",
]