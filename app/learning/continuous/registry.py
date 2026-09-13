from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib


class ModelRegistry:
    """
    Lightweight filesystem model registry.

    Structure:

        models/
          champion/
            model.joblib
            manifest.json
          candidates/
            <run_id>/
          archive/
            <timestamp>/
    """

    def __init__(
        self,
        root: str = "models",
    ):
        self.root = Path(root)

        self.champion_dir = (
            self.root / "champion"
        )

        self.candidates_dir = (
            self.root / "candidates"
        )

        self.archive_dir = (
            self.root / "archive"
        )

        self.champion_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.candidates_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.archive_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def champion_model_path(self) -> Path:
        return self.champion_dir / "model.joblib"

    @property
    def champion_manifest_path(self) -> Path:
        return self.champion_dir / "manifest.json"

    def save_candidate(
        self,
        model,
        run_id: str,
        manifest: Dict,
    ) -> Path:

        directory = (
            self.candidates_dir / run_id
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = directory / "model.joblib"
        manifest_path = directory / "manifest.json"

        joblib.dump(
            model,
            model_path,
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        return model_path

    def has_champion(self) -> bool:
        return self.champion_model_path.exists()

    def load_champion(self):
        if not self.has_champion():
            return None

        return joblib.load(
            self.champion_model_path
        )

    def load_champion_manifest(self) -> Optional[Dict]:
        if not self.champion_manifest_path.exists():
            return None

        return json.loads(
            self.champion_manifest_path.read_text(
                encoding="utf-8"
            )
        )

    def promote(
        self,
        candidate_model_path: str | Path,
        manifest: Dict,
    ) -> str:

        candidate_path = Path(
            candidate_model_path
        )

        if not candidate_path.exists():
            raise FileNotFoundError(
                candidate_path
            )

        previous_manifest = (
            self.load_champion_manifest()
        )

        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )

        if self.has_champion():
            archive_dir = (
                self.archive_dir / timestamp
            )

            archive_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                self.champion_model_path,
                archive_dir / "model.joblib",
            )

            if self.champion_manifest_path.exists():
                shutil.copy2(
                    self.champion_manifest_path,
                    archive_dir / "manifest.json",
                )

        shutil.copy2(
            candidate_path,
            self.champion_model_path,
        )

        promoted_manifest = dict(manifest)

        promoted_manifest.update(
            {
                "promoted_at": timestamp,
                "previous_champion": (
                    previous_manifest or {}
                ).get("model_name"),
            }
        )

        self.champion_manifest_path.write_text(
            json.dumps(
                promoted_manifest,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        return timestamp

    def rollback(
        self,
        archive_timestamp: Optional[str] = None,
    ) -> str:

        if archive_timestamp is None:
            archives = sorted(
                [
                    path
                    for path in self.archive_dir.iterdir()
                    if path.is_dir()
                ]
            )

            if not archives:
                raise FileNotFoundError(
                    "No archived champion available."
                )

            archive_dir = archives[-1]

        else:
            archive_dir = (
                self.archive_dir
                / archive_timestamp
            )

        archived_model = (
            archive_dir / "model.joblib"
        )

        archived_manifest = (
            archive_dir / "manifest.json"
        )

        if not archived_model.exists():
            raise FileNotFoundError(
                archived_model
            )

        shutil.copy2(
            archived_model,
            self.champion_model_path,
        )

        if archived_manifest.exists():
            shutil.copy2(
                archived_manifest,
                self.champion_manifest_path,
            )

        return archive_dir.name