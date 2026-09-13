from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Tuple

import joblib


class ProductionModelLoader:
    """
    Loads the champion model once and caches it in memory.

    The loader is deliberately lightweight and filesystem based.
    """

    def __init__(
        self,
        model_root: str = "models/champion",
    ):
        self.model_root = Path(
            model_root
        )

        self.model_path = (
            self.model_root
            / "model.joblib"
        )

        self.manifest_path = (
            self.model_root
            / "manifest.json"
        )

        self._model = None
        self._manifest = None
        self._load_ms: Optional[float] = None

    def exists(self) -> bool:
        return self.model_path.exists()

    def load(self):
        if self._model is not None:
            return self._model

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Champion model not found: "
                f"{self.model_path}"
            )

        start = time.perf_counter()

        self._model = joblib.load(
            self.model_path
        )

        self._load_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        if self.manifest_path.exists():
            self._manifest = json.loads(
                self.manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        else:
            self._manifest = {
                "model_name": "unknown",
                "model_version": "unknown",
            }

        return self._model

    def manifest(self) -> dict:
        if self._manifest is None:
            self.load()

        return dict(
            self._manifest
        )

    def model_name(self) -> str:
        return str(
            self.manifest().get(
                "model_name",
                "unknown",
            )
        )

    def model_version(self) -> str:
        return str(
            self.manifest().get(
                "run_id",
                self.manifest().get(
                    "promoted_at",
                    "unknown",
                ),
            )
        )

    def load_time_ms(self) -> float:
        if self._load_ms is None:
            self.load()

        return float(
            self._load_ms or 0.0
        )

    def clear(self):
        self._model = None
        self._manifest = None
        self._load_ms = None

    def reload(self):
        self.clear()
        return self.load()