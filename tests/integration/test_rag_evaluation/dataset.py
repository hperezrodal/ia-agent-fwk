"""Open RAG Bench dataset loader and sampler.

Downloads the Open RAG Bench dataset from HuggingFace and provides
utilities for sampling documents and their associated QA pairs for
RAG pipeline evaluation.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """A single question-answer pair with its source metadata."""

    question: str
    answer: str
    context_id: str
    question_type: str


@dataclass
class DocumentSection:
    """A document section with its text content and associated QA pairs."""

    context_id: str
    doc_id: str
    section_id: int
    text: str
    title: str
    qa_pairs: list[QAPair] = field(default_factory=list)

    @property
    def filename(self) -> str:
        """Generate a deterministic filename for this section."""
        hash_suffix = hashlib.md5(self.context_id.encode()).hexdigest()[:8]  # noqa: S324
        return f"{self.doc_id}_s{self.section_id}_{hash_suffix}.txt"


@dataclass
class DocumentGroup:
    """All sections and QA pairs for a single document."""

    doc_id: str
    title: str
    sections: dict[str, DocumentSection] = field(default_factory=dict)

    @property
    def total_qa_pairs(self) -> int:
        return sum(len(s.qa_pairs) for s in self.sections.values())


def load_open_ragbench(
    max_documents: int | None = None,
    min_qa_per_doc: int = 3,
) -> list[DocumentGroup]:
    """Load and structure the Open RAG Bench dataset.

    Parameters
    ----------
    max_documents:
        Maximum number of documents to return. None for all.
    min_qa_per_doc:
        Minimum QA pairs a document must have to be included.

    Returns
    -------
    list[DocumentGroup]
        Structured documents with their sections and QA pairs.

    """
    from datasets import load_dataset

    logger.info("Loading Open RAG Bench dataset from HuggingFace...")
    ds = load_dataset("G4KMU/vectara_open_ragbench", "Open RAGBench", split="text_tables")

    # Group by doc_id
    doc_groups: dict[str, DocumentGroup] = {}

    for row in ds:
        doc_id = row["doc_id"]
        context_id = row["context_id"]

        if doc_id not in doc_groups:
            doc_groups[doc_id] = DocumentGroup(
                doc_id=doc_id,
                title=row["title"],
            )

        group = doc_groups[doc_id]

        # Create or get section
        if context_id not in group.sections:
            group.sections[context_id] = DocumentSection(
                context_id=context_id,
                doc_id=doc_id,
                section_id=row["section_id"],
                text=row["text"],
                title=row["title"],
            )

        # Add QA pair
        group.sections[context_id].qa_pairs.append(
            QAPair(
                question=row["question"],
                answer=row["answer"],
                context_id=context_id,
                question_type=row["question_type"],
            )
        )

    # Filter by minimum QA pairs
    filtered = [g for g in doc_groups.values() if g.total_qa_pairs >= min_qa_per_doc]
    # Sort by total QA pairs (descending) for consistent sampling
    filtered.sort(key=lambda g: g.total_qa_pairs, reverse=True)

    if max_documents is not None:
        filtered = filtered[:max_documents]

    total_sections = sum(len(g.sections) for g in filtered)
    total_qa = sum(g.total_qa_pairs for g in filtered)
    logger.info(
        "Dataset loaded: %d documents, %d sections, %d QA pairs",
        len(filtered),
        total_sections,
        total_qa,
    )

    return filtered


def write_sections_to_dir(
    groups: list[DocumentGroup],
    output_dir: Path,
) -> dict[str, Path]:
    """Write document sections as text files for pipeline ingestion.

    Parameters
    ----------
    groups:
        Document groups to write.
    output_dir:
        Directory to write .txt files into.

    Returns
    -------
    dict[str, Path]
        Mapping from context_id to file path.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    context_to_path: dict[str, Path] = {}

    for group in groups:
        for ctx_id, section in group.sections.items():
            file_path = output_dir / section.filename
            file_path.write_text(section.text, encoding="utf-8")
            context_to_path[ctx_id] = file_path

    logger.info("Wrote %d section files to %s", len(context_to_path), output_dir)
    return context_to_path


def collect_all_qa_pairs(
    groups: list[DocumentGroup],
) -> list[tuple[QAPair, DocumentSection]]:
    """Flatten all QA pairs with their parent section reference.

    Returns
    -------
    list[tuple[QAPair, DocumentSection]]
        Each tuple is (qa_pair, parent_section).

    """
    pairs: list[tuple[QAPair, DocumentSection]] = []
    for group in groups:
        for section in group.sections.values():
            for qa in section.qa_pairs:
                pairs.append((qa, section))
    return pairs
