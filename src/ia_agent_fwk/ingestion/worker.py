"""Ingestion worker — polls MinIO for new documents and ingests them.

Designed to run as a cronjob or long-running worker.

Flow:
  1. List files in MinIO bucket
  2. Compare ETags against manifest (skip unchanged)
  3. Download new/changed files
  4. Run ingestion pipeline (parse → clean → chunk → contextual → enrich → embed)
  5. Update manifest
  6. Cleanup temp files

Usage (cronjob):
    python -m ia_agent_fwk.ingestion.worker \
        --minio-endpoint localhost:9000 \
        --bucket documents \
        --db postgresql://postgres:postgres@localhost:5432/mydb \
        --tenant webdelseguro

Usage (from code):
    worker = IngestionWorker(storage=..., store=..., orchestrator=..., manifest=...)
    await worker.run()
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ia_agent_fwk.ingestion.manifest import IngestionManifest
from ia_agent_fwk.ingestion.storage import DocumentStorage, StoredFile

logger = logging.getLogger(__name__)


class IngestionWorker:
    """Poll MinIO for new documents and ingest them.

    Parameters
    ----------
    storage:
        MinIO/S3 document storage.
    store:
        EmbeddingStore for embed + store.
    orchestrator:
        IngestionOrchestrator for the full pipeline.
    manifest:
        Tracks processed files.
    faq_ingestor:
        Optional — if provided, JSON files are ingested as FAQs.

    """

    def __init__(
        self,
        storage: DocumentStorage,
        store: object,  # EmbeddingStore
        orchestrator: object,  # IngestionOrchestrator
        manifest: IngestionManifest,
        faq_ingestor: object | None = None,
        extra_metadata_fn: object | None = None,
        save_parsed_dir: str | None = None,
    ) -> None:
        self._storage = storage
        self._store = store
        self._orchestrator = orchestrator
        self._manifest = manifest
        self._faq_ingestor = faq_ingestor
        self._extra_metadata_fn = extra_metadata_fn
        self._save_parsed_dir = save_parsed_dir

    async def run(self, prefix: str = "") -> dict:
        """Run one ingestion cycle.

        Returns summary dict with counts.
        """
        logger.info("Ingestion worker: starting cycle")

        # 1. List files in MinIO
        all_files = self._storage.list_files(prefix=prefix)
        logger.info("Found %d files in storage", len(all_files))

        # 2. Build ETag manifest for comparison
        # We use the object key as the manifest key and ETag as hash
        etag_map = {}
        for f in all_files:
            etag_map[f.key] = f.etag

        # Check which files need processing
        new_files = self._storage.list_new_files(
            manifest_etags=self._get_manifest_etags(),
            prefix=prefix,
        )

        if not new_files:
            logger.info("All files up to date. Nothing to ingest.")
            return {"total": len(all_files), "new": 0, "ingested": 0, "failed": 0}

        logger.info("%d new/changed file(s) to ingest", len(new_files))

        # 3. Process each new file
        ingested = 0
        failed = 0

        with tempfile.TemporaryDirectory() as tmp_dir:
            for sf in new_files:
                try:
                    result = await self._process_file(sf, tmp_dir)
                    if result:
                        self._update_manifest(sf, result)
                        ingested += 1
                    else:
                        failed += 1
                except Exception:
                    logger.exception("Failed to ingest %s", sf.key)
                    failed += 1

        # 4. Save manifest
        self._manifest.save()

        summary = {
            "total": len(all_files),
            "new": len(new_files),
            "ingested": ingested,
            "failed": failed,
        }
        logger.info("Ingestion cycle done: %s", summary)
        return summary

    async def _process_file(self, sf: StoredFile, tmp_dir: str) -> dict | None:
        """Download and ingest a single file."""
        logger.info("Processing: %s (%d bytes)", sf.key, sf.size)

        # Download
        local_path = self._storage.download(sf.key, tmp_dir)

        # Check if it's a FAQ JSON
        if local_path.suffix.lower() == ".json" and self._faq_ingestor:
            return await self._process_faq(local_path, sf)

        # Build extra metadata
        extra_metadata = {}
        if self._extra_metadata_fn:
            extra_metadata = self._extra_metadata_fn(sf.key)

        # Run full pipeline
        result = await self._orchestrator.process(
            file_path=str(local_path),
            extra_metadata=extra_metadata,
            save_parsed=self._save_parsed_dir,
        )

        # Embed and store
        stored = await self._store.store_chunks(result.chunks, file_name=local_path.name)

        logger.info(
            "Ingested %s: %d chunks (%s, %s)",
            sf.key,
            stored,
            result.parser_used,
            result.classification.doc_type.value,
        )
        return {
            "chunks": stored,
            "parser": result.parser_used,
            "doc_type": result.classification.doc_type.value,
        }

    async def _process_faq(self, local_path: Path, sf: StoredFile) -> dict | None:
        """Process a FAQ JSON file."""
        from ia_agent_fwk.ingestion.faq_ingestor import ingest_faq_file  # noqa: PLC0415

        chunks = ingest_faq_file(local_path)
        if not chunks:
            return None

        stored = await self._store.store_chunks(chunks, file_name=local_path.name)
        logger.info("Ingested FAQ %s: %d chunks", sf.key, stored)
        return {"chunks": stored, "parser": "faq", "doc_type": "faq"}

    def _get_manifest_etags(self) -> dict[str, str]:
        """Build ETag lookup from manifest.

        We store the MinIO ETag in the manifest's sha256 field
        (repurposed for S3 objects).
        """
        return self._manifest.get_etag_map()

    def _update_manifest(self, sf: StoredFile, result: dict) -> None:
        """Update manifest with the processed file."""
        import datetime  # noqa: PLC0415

        from ia_agent_fwk.ingestion.manifest import FileRecord  # noqa: PLC0415

        record = FileRecord(
            path=sf.key,
            sha256=sf.etag,  # Use ETag as the "hash"
            chunk_count=result.get("chunks", 0),
            parser_used=result.get("parser", ""),
            last_ingested=datetime.datetime.now(tz=datetime.UTC).isoformat(),
        )
        self._manifest.set_record(sf.key, record)
