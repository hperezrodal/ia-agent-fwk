"""FAQ ingestor — converts structured FAQ JSON files into chunks.

Each question-answer pair becomes a single chunk. No parsing, cleaning,
or chunking needed — FAQs are already structured.

Expected JSON format:
    {
        "category": "siniestros",
        "scope": "faq",
        "questions": [
            {
                "question": "¿Qué hago si tengo un siniestro?",
                "answer": "Primero asegurate de que todos estén bien...",
                "tags": ["siniestro", "denuncia"]
            }
        ]
    }

Usage:
    from ia_agent_fwk.ingestion.faq_ingestor import ingest_faq_file, ingest_faq_directory

    chunks = ingest_faq_file("data/faq/siniestros.json")
    chunks = ingest_faq_directory("data/faq/")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ia_agent_fwk.ingestion.models import ProcessedChunk

logger = logging.getLogger(__name__)


def ingest_faq_file(file_path: str | Path) -> list[ProcessedChunk]:
    """Convert a FAQ JSON file into chunks.

    Each question-answer pair becomes one chunk.
    """
    path = Path(file_path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    category = data.get("category", path.stem)
    scope = data.get("scope", "faq")
    questions = data.get("questions", [])

    chunks: list[ProcessedChunk] = []
    for i, q in enumerate(questions):
        question = q.get("question", "").strip()
        answer = q.get("answer", "").strip()
        tags = q.get("tags", [])

        if not question or not answer:
            continue

        content = f"{question}\n\n{answer}"

        chunks.append(
            ProcessedChunk(
                content=content,
                metadata={
                    "chunk_type": "text",
                    "source_type": "faq",
                    "scope": scope,
                    "category": category,
                    "tags": ", ".join(tags) if tags else "",
                    "source": str(path),
                    "document_id": path.name,
                    "faq_index": i,
                },
            )
        )

    logger.info("FAQ ingestor: %s → %d chunks (category=%s)", path.name, len(chunks), category)
    return chunks


def ingest_faq_directory(directory: str | Path) -> list[ProcessedChunk]:
    """Ingest all FAQ JSON files from a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        logger.warning("FAQ directory not found: %s", dir_path)
        return []

    all_chunks: list[ProcessedChunk] = []
    for faq_file in sorted(dir_path.glob("*.json")):
        chunks = ingest_faq_file(faq_file)
        all_chunks.extend(chunks)

    logger.info(
        "FAQ directory: %s → %d total chunks from %d files",
        dir_path,
        len(all_chunks),
        len(list(dir_path.glob("*.json"))),
    )
    return all_chunks
