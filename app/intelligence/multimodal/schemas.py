from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TextContext:
    text: str
    source: str = "unknown"
    language: str = "unknown"
    metadata: Dict[str, str] = field(
        default_factory=dict
    )


@dataclass
class VisualContext:
    image_path: Optional[str] = None
    source: str = "unknown"
    metadata: Dict[str, str] = field(
        default_factory=dict
    )


@dataclass
class MultimodalFeatures:
    features: Dict[str, float]
    text_available: bool
    visual_available: bool
    modality_count: int
    fusion_score: float
    quality_score: float
    metadata: Dict[str, str] = field(
        default_factory=dict
    )