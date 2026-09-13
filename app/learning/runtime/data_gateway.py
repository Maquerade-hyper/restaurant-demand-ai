from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass
class ClientDataPackage:
    """
    Represents one discovered real-client data package.
    """

    package_id: str
    files: List[Path]
    source_type: str
    detected_at: str
    row_estimate: int


class ClientDataGateway:
    """
    Gateway between external client data and the learning runtime.

    Responsibilities:

    INCOMING
        ↓
    discover package
        ↓
    validate package
        ↓
    process / reject
        ↓
    PROCESSED / REJECTED
        ↓
    ARCHIVE

    Synthetic project data is deliberately outside this gateway.
    """

    SUPPORTED_EXTENSIONS = {
        ".csv",
        ".parquet",
        ".json",
    }

    REQUIRED_COLUMNS = {
        "date",
        "outlet_id",
        "product_id",
        "quantity_sold",
    }

    def __init__(
        self,
        incoming_path: str = "data/client/INCOMING",
        processed_path: str = "data/client/PROCESSED",
        rejected_path: str = "data/client/REJECTED",
        archive_path: str = "data/client/ARCHIVE",
    ):
        self.incoming_path = Path(
            incoming_path
        )

        self.processed_path = Path(
            processed_path
        )

        self.rejected_path = Path(
            rejected_path
        )

        self.archive_path = Path(
            archive_path
        )

        self._ensure_directories()

    # ============================================================
    # DIRECTORY MANAGEMENT
    # ============================================================

    def _ensure_directories(self) -> None:
        """
        Create all client-runtime directories.
        """

        self.incoming_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.processed_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.rejected_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.archive_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ============================================================
    # PACKAGE DISCOVERY
    # ============================================================

    def discover_packages(
        self,
    ) -> List[ClientDataPackage]:
        """
        Discover supported files currently waiting in INCOMING.

        Files are grouped into one package for the current runtime
        cycle.

        Only real client files placed in the client INCOMING
        directory are considered.
        """

        files = []

        for path in sorted(
            self.incoming_path.iterdir()
        ):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                not in self.SUPPORTED_EXTENSIONS
            ):
                continue

            files.append(
                path
            )

        if not files:
            return []

        package_id = self._package_id(
            files
        )

        detected_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        row_estimate = sum(
            self._estimate_rows(
                path
            )
            for path in files
        )

        package = ClientDataPackage(
            package_id=package_id,
            files=files,
            source_type="real_client",
            detected_at=detected_at,
            row_estimate=row_estimate,
        )

        return [package]

    # ============================================================
    # PACKAGE ID
    # ============================================================

    @staticmethod
    def _package_id(
        files: List[Path],
    ) -> str:
        """
        Build a deterministic package identifier from
        the currently discovered files.
        """

        payload = []

        for path in sorted(files):

            try:
                stat = path.stat()

                payload.append(
                    (
                        str(path.resolve()),
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                )

            except FileNotFoundError:
                continue

        raw = repr(
            payload
        ).encode(
            "utf-8"
        )

        digest = hashlib.sha256(
            raw
        ).hexdigest()[:16]

        return (
            f"client_{digest}"
        )

    # ============================================================
    # ROW ESTIMATION
    # ============================================================

    @staticmethod
    def _estimate_rows(
        path: Path,
    ) -> int:
        """
        Estimate row count without requiring a complete
        in-memory load.

        Exact counts are used where inexpensive.
        """

        suffix = path.suffix.lower()

        try:

            if suffix == ".csv":

                with path.open(
                    "r",
                    encoding="utf-8-sig",
                    errors="ignore",
                ) as handle:

                    return max(
                        sum(
                            1
                            for _ in handle
                        )
                        - 1,
                        0,
                    )

            if suffix == ".json":

                frame = (
                    __import__(
                        "pandas"
                    )
                    .read_json(
                        path
                    )
                )

                return len(
                    frame
                )

            if suffix == ".parquet":

                import pandas as pd

                return len(
                    pd.read_parquet(
                        path,
                        columns=[],
                    )
                )

        except Exception:
            return 0

        return 0

    # ============================================================
    # SAMPLE VALIDATION
    # ============================================================

    def validate_package_sample(
        self,
        package: ClientDataPackage,
    ) -> dict:
        """
        Validate the package schema using a small sample.

        This does not train anything.
        """

        errors = []

        if not package.files:

            errors.append(
                "Package contains no files."
            )

            return {
                "passed": False,
                "errors": errors,
                "files": [],
            }

        for file_path in package.files:

            path = Path(
                file_path
            )

            if not path.exists():

                errors.append(
                    f"File does not exist: {path}"
                )

                continue

            try:

                suffix = (
                    path.suffix.lower()
                )

                if suffix == ".csv":

                    import pandas as pd

                    sample = pd.read_csv(
                        path,
                        nrows=10,
                    )

                elif suffix == ".parquet":

                    import pandas as pd

                    sample = pd.read_parquet(
                        path
                    ).head(
                        10
                    )

                elif suffix == ".json":

                    import pandas as pd

                    sample = pd.read_json(
                        path
                    ).head(
                        10
                    )

                else:

                    errors.append(
                        "Unsupported file type: "
                        f"{path.name}"
                    )

                    continue

                columns = {
                    str(column)
                    .strip()
                    .lower()
                    for column in sample.columns
                }

                missing = (
                    self.REQUIRED_COLUMNS
                    - columns
                )

                if missing:

                    errors.append(
                        f"{path.name}: missing "
                        "required columns: "
                        f"{sorted(missing)}"
                    )

            except Exception as exc:

                errors.append(
                    f"{path.name}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        return {
            "passed": not errors,
            "errors": errors,
            "files": [
                str(path)
                for path in package.files
            ],
        }

    # ============================================================
    # MANIFEST
    # ============================================================

    def manifest(
        self,
        package: ClientDataPackage,
        file_paths: List[Path] | None = None,
        status: str = "DISCOVERED",
        reason: str | None = None,
    ) -> dict:
        """
        Build a package manifest.

        file_paths allows callers to build a manifest against
        destination paths after a package has been moved.

        This explicitly prevents the old bug where manifest()
        attempted to stat files that had already been moved.
        """

        paths = (
            file_paths
            if file_paths is not None
            else package.files
        )

        file_records = []

        for file_path in paths:

            path = Path(
                file_path
            )

            record = {
                "path": str(path),
                "name": path.name,
                "exists": path.exists(),
            }

            if path.exists():

                try:

                    stat = path.stat()

                    record[
                        "size_bytes"
                    ] = stat.st_size

                    record[
                        "modified_at"
                    ] = datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    ).isoformat()

                except OSError:

                    record[
                        "size_bytes"
                    ] = None

                    record[
                        "modified_at"
                    ] = None

            else:

                record[
                    "size_bytes"
                ] = None

                record[
                    "modified_at"
                ] = None

            file_records.append(
                record
            )

        manifest = {
            "package_id": package.package_id,
            "source_type": package.source_type,
            "detected_at": package.detected_at,
            "row_estimate": package.row_estimate,
            "status": status,
            "files": file_records,
        }

        if reason is not None:

            manifest[
                "reason"
            ] = reason

        return manifest

    # ============================================================
    # SAFE DESTINATION PATH
    # ============================================================

    @staticmethod
    def _unique_destination(
        directory: Path,
        filename: str,
    ) -> Path:
        """
        Avoid overwriting an existing file.
        """

        destination = (
            directory / filename
        )

        if not destination.exists():
            return destination

        stem = destination.stem
        suffix = destination.suffix

        counter = 1

        while True:

            candidate = (
                directory
                / f"{stem}_{counter}{suffix}"
            )

            if not candidate.exists():
                return candidate

            counter += 1

    # ============================================================
    # MOVE FILES
    # ============================================================

    def move_files(
        self,
        package: ClientDataPackage,
        destination: Path,
    ) -> List[Path]:
        """
        Move package files to a destination directory.

        Returns the actual destination paths.
        """

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        moved = []

        for source in package.files:

            source = Path(
                source
            )

            if not source.exists():
                continue

            target = (
                self._unique_destination(
                    destination,
                    source.name,
                )
            )

            shutil.move(
                str(source),
                str(target),
            )

            moved.append(
                target
            )

        return moved

    # ============================================================
    # WRITE MANIFEST
    # ============================================================

    @staticmethod
    def _write_manifest(
        manifest: dict,
        directory: Path,
        package_id: str,
    ) -> Path:
        """
        Persist a package manifest.

        Two names are written:

        1. manifest.json
           Stable conventional manifest path used by tests
           and simple downstream consumers.

        2. <package_id>_manifest.json
           Package-specific audit artifact.

        The stable manifest is returned because it is the
        primary gateway contract.
        """

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------
        # Stable manifest
        # --------------------------------------------------------

        stable_path = (
            directory
            / "manifest.json"
        )

        with stable_path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                manifest,
                handle,
                indent=2,
                default=str,
            )

        # --------------------------------------------------------
        # Package-specific audit manifest
        # --------------------------------------------------------

        package_path = (
            directory
            / f"{package_id}_manifest.json"
        )

        with package_path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                manifest,
                handle,
                indent=2,
                default=str,
            )

        return stable_path

    # ============================================================
    # REJECT PACKAGE
    # ============================================================

    def reject_package(
        self,
        package: ClientDataPackage,
        reason: str = "REJECTED",
    ) -> Path:
        """
        Reject a client package.

        Files are moved to REJECTED/.
        A stable manifest.json and a package-specific
        audit manifest are written there.

        Returns the REJECTED directory so callers can
        access:

            rejected / "manifest.json"
        """

        # --------------------------------------------------------
        # Capture metadata BEFORE moving files
        # --------------------------------------------------------

        original_manifest = self.manifest(
            package,
            file_paths=list(
                package.files
            ),
            status="REJECTING",
            reason=reason,
        )

        # --------------------------------------------------------
        # Move package files
        # --------------------------------------------------------

        moved_files = self.move_files(
            package,
            self.rejected_path,
        )

        # --------------------------------------------------------
        # Build final manifest using destination paths
        # --------------------------------------------------------

        final_manifest = self.manifest(
            package,
            file_paths=moved_files,
            status="REJECTED",
            reason=reason,
        )

        final_manifest[
            "original_files"
        ] = original_manifest[
            "files"
        ]

        # --------------------------------------------------------
        # Write stable + package-specific manifests
        # --------------------------------------------------------

        self._write_manifest(
            final_manifest,
            self.rejected_path,
            package.package_id,
        )

        # --------------------------------------------------------
        # Return destination directory
        # --------------------------------------------------------

        return self.rejected_path

    # ============================================================
    # PROCESS PACKAGE
    # ============================================================

    def process_package(
        self,
        package: ClientDataPackage,
    ) -> Path:
        """
        Mark a successfully processed client package.

        The original files are moved to PROCESSED and a copy of
        the package metadata is retained in ARCHIVE.
        """

        # Capture metadata BEFORE moving.
        original_manifest = self.manifest(
            package,
            file_paths=list(
                package.files
            ),
            status="PROCESSING",
        )

        processed_files = (
            self.move_files(
                package,
                self.processed_path,
            )
        )

        processed_manifest = self.manifest(
            package,
            file_paths=processed_files,
            status="PROCESSED",
        )

        processed_manifest[
            "original_files"
        ] = original_manifest[
            "files"
        ]

        # Archive metadata separately.
        archive_manifest = dict(
            processed_manifest
        )

        archive_manifest[
            "archived_at"
        ] = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        self._write_manifest(
            archive_manifest,
            self.archive_path,
            package.package_id,
        )

        return (
            self.processed_path
        )

    # ============================================================
    # ARCHIVE MANIFEST ONLY
    # ============================================================

    def archive_manifest(
        self,
        package: ClientDataPackage,
        status: str,
        reason: str | None = None,
    ) -> Path:
        """
        Write an archive manifest without moving files.

        Useful for audit/debug workflows.
        """

        manifest = self.manifest(
            package,
            status=status,
            reason=reason,
        )

        manifest[
            "archived_at"
        ] = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        return self._write_manifest(
            manifest,
            self.archive_path,
            package.package_id,
        )