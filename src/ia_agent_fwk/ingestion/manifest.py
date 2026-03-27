"""Ingestion manifest — tracks processed files for idempotent re-ingestion.

Stores a SHA256 hash per file. On re-run, only files that changed (or are new)
get re-processed. Files that were deleted from disk are removed from the manifest.

Usage:
    manifest = IngestionManifest("data/ingestion_manifest.json")

    for file in files:
        if manifest.needs_processing(file):
            process(file)
            manifest.mark_processed(file)

    manifest.save()
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FileRecord:
    """Record for a single processed file."""

    path: str
    sha256: str
    chunk_count: int = 0
    parser_used: str = ""
    last_ingested: str = ""


class IngestionManifest:
    """Track processed files to avoid redundant re-ingestion.

    Parameters
    ----------
    manifest_path:
        Path to the manifest JSON file.

    """

    def __init__(self, manifest_path: str | Path = "data/ingestion_manifest.json") -> None:
        self._path = Path(manifest_path)
        self._records: dict[str, FileRecord] = {}
        self._load()

    def _load(self) -> None:
        """Load manifest from disk."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for key, rec in data.get("files", {}).items():
                    self._records[key] = FileRecord(**rec)
                logger.info("Loaded manifest: %d files tracked", len(self._records))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load manifest: %s", exc)

    def save(self) -> None:
        """Save manifest to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "files": {
                key: {
                    "path": rec.path,
                    "sha256": rec.sha256,
                    "chunk_count": rec.chunk_count,
                    "parser_used": rec.parser_used,
                    "last_ingested": rec.last_ingested,
                }
                for key, rec in self._records.items()
            },
        }
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def needs_processing(self, file_path: str | Path) -> bool:
        """Check if a file needs (re-)processing.

        Returns True if:
        - File is not in the manifest (new file)
        - File hash differs from the stored hash (file changed)
        """
        path = Path(file_path)
        key = str(path.resolve())
        current_hash = _file_hash(path)

        record = self._records.get(key)
        if record is None:
            return True
        return record.sha256 != current_hash

    def mark_processed(
        self,
        file_path: str | Path,
        chunk_count: int = 0,
        parser_used: str = "",
    ) -> None:
        """Mark a file as successfully processed."""
        import datetime  # noqa: PLC0415

        path = Path(file_path)
        key = str(path.resolve())
        self._records[key] = FileRecord(
            path=str(path),
            sha256=_file_hash(path),
            chunk_count=chunk_count,
            parser_used=parser_used,
            last_ingested=datetime.datetime.now(tz=datetime.UTC).isoformat(),
        )

    def remove_deleted(self, existing_files: list[Path]) -> list[str]:
        """Remove records for files that no longer exist on disk.

        Returns list of removed file paths.
        """
        existing_keys = {str(f.resolve()) for f in existing_files}
        removed = []
        for key in list(self._records):
            if key not in existing_keys:
                removed.append(self._records[key].path)
                del self._records[key]
        return removed

    def get_etag_map(self) -> dict[str, str]:
        """Return a mapping of ``record.path -> record.sha256`` for every tracked file.

        Useful for comparing ETags when polling an object store.
        """
        return {rec.path: rec.sha256 for rec in self._records.values()}

    def set_record(self, key: str, record: FileRecord) -> None:
        """Insert or replace the record stored under *key*."""
        self._records[key] = record

    def get_record_type(self) -> type[FileRecord]:
        """Return the ``FileRecord`` class used by this manifest."""
        return FileRecord

    @property
    def file_count(self) -> int:
        return len(self._records)

    def summary(self) -> str:
        """Human-readable summary."""
        return f"{self.file_count} files tracked in {self._path}"


def _file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
