from __future__ import annotations

from datetime import datetime, timezone

from .schemas import DataPackage, DataValidationResult, TrainingDecision


class TrainingTrigger:
    """
    Decides whether a validated client package is eligible for training.

    Synthetic data is never accepted by this trigger.
    """

    def __init__(
        self,
        minimum_rows: int = 1000,
        require_real_data: bool = True,
        training_enabled: bool = True,
    ):
        self.minimum_rows = int(minimum_rows)
        self.require_real_data = bool(require_real_data)
        self.training_enabled = bool(training_enabled)

    def evaluate(
        self,
        package: DataPackage,
        validation: DataValidationResult,
    ) -> TrainingDecision:

        if not self.training_enabled:
            return TrainingDecision(
                False,
                "TRAINING_DISABLED",
                package.package_id,
            )

        if self.require_real_data and not package.is_real_client_data:
            return TrainingDecision(
                False,
                "REAL_CLIENT_DATA_REQUIRED",
                package.package_id,
            )

        if not validation.valid:
            return TrainingDecision(
                False,
                "VALIDATION_FAILED",
                package.package_id,
            )

        if validation.rows < self.minimum_rows:
            return TrainingDecision(
                False,
                f"INSUFFICIENT_DATA:{validation.rows}<{self.minimum_rows}",
                package.package_id,
            )

        return TrainingDecision(
            True,
            "TRAINING_ELIGIBLE",
            package.package_id,
        )

    @staticmethod
    def package_is_complete(package: DataPackage) -> bool:
        for filename in package.files:
            path = __import__("pathlib").Path(filename)

            if not path.exists():
                return False

            if path.stat().st_size <= 0:
                return False

        return True

    @staticmethod
    def timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()