from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class DataPackage:
    package_id: str
    path: str
    files: List[str]
    is_real_client_data: bool
    discovered_at: str
    total_size_bytes: int


@dataclass
class DataValidationResult:
    package_id: str
    valid: bool
    rows: int
    files_found: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class TrainingDecision:
    should_train: bool
    reason: str
    package_id: str


@dataclass
class TrainingResult:
    package_id: str
    status: str
    rows: int
    candidate_path: str | None = None
    champion_path: str | None = None
    candidate_mae: float | None = None
    champion_mae: float | None = None
    improvement: float | None = None
    promoted: bool = False
    reason: str = ""
    started_at: str = ""
    completed_at: str = ""


@dataclass
class RuntimeStatus:
    running: bool
    last_scan: str | None
    last_training: str | None
    last_training_status: str | None
    client_data_available: bool
    training_enabled: bool
    synthetic_training_allowed: bool
    packages_detected: int = 0
    training_in_progress: bool = False
    last_message: str = ""