"""RAG pipeline evaluation using Open RAG Bench dataset.

Integration tests that evaluate the full RAG pipeline (ingest -> query)
against the Open RAG Bench dataset. Requires:
- Qdrant running on localhost:6333 (docker-compose up qdrant)
- Ollama running on localhost:11434 with nomic-embed-text model
  OR OPENAI_API_KEY set for OpenAI embeddings.

Run with:
    pytest tests/integration/test_rag_evaluation/ -v -s

Configure via environment variables:
    RAG_EVAL_MAX_DOCS=10       Number of documents to sample (default: 10)
    RAG_EVAL_TOP_K=5           Retrieval top-k (default: 5)
    RAG_EVAL_CHUNK_SIZE=800    Chunk size in characters (default: 800)
    RAG_EVAL_CHUNK_OVERLAP=200 Chunk overlap in characters (default: 200)
    RAG_EVAL_LLM_MODEL=qwen3:14b  LLM model for generation/judge (default: qwen3:14b)
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import pytest

from tests.integration.test_rag_evaluation.conftest import (
    skip_no_ollama,
    skip_no_openai_key,
    skip_no_qdrant,
)
from tests.integration.test_rag_evaluation.dataset import (
    collect_all_qa_pairs,
    load_open_ragbench,
    write_sections_to_dir,
)
from tests.integration.test_rag_evaluation.llm_judge import (
    evaluate_correctness,
    evaluate_faithfulness,
    generate_rag_answer,
)
from tests.integration.test_rag_evaluation.metrics import (
    EvaluationReport,
    QueryInput,
    aggregate_evaluations,
    evaluate_query,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
MAX_DOCS = int(os.environ.get("RAG_EVAL_MAX_DOCS", "10"))
TOP_K = int(os.environ.get("RAG_EVAL_TOP_K", "5"))
CHUNK_SIZE = int(os.environ.get("RAG_EVAL_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("RAG_EVAL_CHUNK_OVERLAP", "200"))
OLLAMA_URL = os.environ.get("RAG_EVAL_OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.environ.get("RAG_EVAL_QDRANT_URL", "http://127.0.0.1:6333")
LLM_MODEL = os.environ.get("RAG_EVAL_LLM_MODEL", "qwen3:14b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _build_pipeline(
    embedding_provider_type: str,
    collection_name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):  # type: ignore[no-untyped-def]
    """Build a full RAG pipeline with real backends.

    Parameters
    ----------
    embedding_provider_type:
        Either "ollama" or "openai".
    collection_name:
        Qdrant collection name for test isolation.
    chunk_size:
        Character-level chunk size for the recursive chunker.
    chunk_overlap:
        Character-level overlap between chunks.

    Returns
    -------
    tuple[RAGPipeline, QdrantMemoryBackend]
        The configured pipeline and the backend (for cleanup).

    """
    from ia_agent_fwk.memory.backends.qdrant import QdrantMemoryBackend
    from ia_agent_fwk.memory.embeddings.ollama import OllamaEmbeddingProvider
    from ia_agent_fwk.memory.embeddings.openai import OpenAIEmbeddingProvider
    from ia_agent_fwk.rag.chunkers.recursive import RecursiveChunker
    from ia_agent_fwk.rag.loaders.registry import LoaderRegistry
    from ia_agent_fwk.rag.pipeline import RAGPipeline
    from ia_agent_fwk.rag.retrieval.vector import VectorRetriever

    # Embedding provider
    if embedding_provider_type == "ollama":
        embedding_provider = OllamaEmbeddingProvider(
            base_url=OLLAMA_URL,
            model="nomic-embed-text",
        )
        embedding_dim = 768
    elif embedding_provider_type == "openai":
        embedding_provider = OpenAIEmbeddingProvider(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model="text-embedding-3-small",
        )
        embedding_dim = 1536
    else:
        msg = f"Unknown embedding provider: {embedding_provider_type}"
        raise ValueError(msg)

    # Qdrant vector backend
    backend = QdrantMemoryBackend(
        url=QDRANT_URL,
        embedding_provider=embedding_provider,
        collection_name=collection_name,
        embedding_dimensions=embedding_dim,
        agent_namespace="rag_eval",
    )

    # Chunker
    chunker = RecursiveChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Loader registry (uses built-in TextLoader for .txt)
    loader_registry = LoaderRegistry()

    # Retriever
    retriever = VectorRetriever(
        backend=backend,
        embedding_provider=embedding_provider,
    )

    # Pipeline
    pipeline = RAGPipeline(
        loader_registry=loader_registry,
        chunker=chunker,
        embedding_provider=embedding_provider,
        memory_backend=backend,
        retriever=retriever,
    )

    return pipeline, backend


async def _cleanup_qdrant(collection_name: str) -> None:
    """Delete the test collection from Qdrant."""
    try:
        from qdrant_client import AsyncQdrantClient

        client = AsyncQdrantClient(url=QDRANT_URL)
        await client.delete_collection(collection_name)
        await client.close()
    except Exception:  # noqa: BLE001
        logger.warning("Failed to cleanup Qdrant collection %s", collection_name)


async def _run_evaluation(  # noqa: PLR0913, C901, PLR0912, PLR0915
    embedding_type: str,
    tmp_path: Path,
    top_k: int = TOP_K,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    max_docs: int = MAX_DOCS,
    *,
    enable_llm_judge: bool = False,
) -> EvaluationReport:
    """Run the full RAG evaluation pipeline.

    Parameters
    ----------
    embedding_type:
        "ollama" or "openai".
    tmp_path:
        Temporary directory for document files.
    top_k:
        Number of chunks to retrieve per query.
    chunk_size:
        Chunker character-level size.
    chunk_overlap:
        Chunker overlap.
    max_docs:
        Number of documents to sample from dataset.
    enable_llm_judge:
        If True, use LLM to generate answers and evaluate hallucination.

    Returns
    -------
    EvaluationReport
        Aggregated evaluation metrics.

    """
    collection_name = f"rag_eval_{uuid.uuid4().hex[:8]}"

    # Load dataset
    doc_groups = load_open_ragbench(max_documents=max_docs, min_qa_per_doc=3)
    if not doc_groups:
        pytest.skip("No documents matched sampling criteria")

    # Write sections to disk
    context_to_path = write_sections_to_dir(doc_groups, tmp_path / "docs")

    # Build pipeline
    pipeline, _backend = await _build_pipeline(
        embedding_provider_type=embedding_type,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    try:
        # Ingest all sections
        file_paths = list(context_to_path.values())
        logger.info("Ingesting %d document sections...", len(file_paths))
        results = await pipeline.ingest_batch(file_paths)

        total_chunks = sum(r.chunk_count for r in results)
        total_time = sum(r.duration_ms for r in results)
        logger.info(
            "Ingestion complete: %d sections, %d chunks in %.1fs (chunk_size=%d, overlap=%d)",
            len(results),
            total_chunks,
            total_time / 1000,
            chunk_size,
            chunk_overlap,
        )

        # Collect QA pairs and evaluate
        qa_pairs = collect_all_qa_pairs(doc_groups)
        logger.info(
            "Evaluating %d QA pairs (top_k=%d, llm_judge=%s)...",
            len(qa_pairs),
            top_k,
            enable_llm_judge,
        )

        evaluations = []
        for i, (qa, section) in enumerate(qa_pairs):
            query_result = await pipeline.query(qa.question, top_k=top_k)

            # Extract source filenames from retrieved chunks
            retrieved_sources = []
            retrieved_scores = []
            for rr in query_result.results:
                source = rr.chunk.source
                # Normalize: source may be full path, we need just filename
                # Also strip chunk suffix like ":chunk:N"
                source_name = Path(source).name if source else ""
                if ":chunk:" in source_name:
                    source_name = source_name.split(":chunk:")[0]
                retrieved_sources.append(source_name)
                retrieved_scores.append(rr.score)

            # LLM generation + hallucination evaluation
            generated_answer = ""
            faithfulness_score = 0.0
            correctness_score = 0.0

            if enable_llm_judge:
                try:
                    generated_answer = await generate_rag_answer(
                        question=qa.question,
                        context=query_result.context,
                        ollama_url=OLLAMA_URL,
                        model=LLM_MODEL,
                    )

                    faith_score, faith_reason = await evaluate_faithfulness(
                        question=qa.question,
                        context=query_result.context,
                        generated_answer=generated_answer,
                        ollama_url=OLLAMA_URL,
                        model=LLM_MODEL,
                    )
                    faithfulness_score = faith_score

                    corr_score, _corr_reason = await evaluate_correctness(
                        question=qa.question,
                        gold_answer=qa.answer,
                        generated_answer=generated_answer,
                        ollama_url=OLLAMA_URL,
                        model=LLM_MODEL,
                    )
                    correctness_score = corr_score

                    if faithfulness_score < 0.5:
                        logger.warning(
                            "  HALLUCINATION detected Q: %s\n    Faithfulness: %.2f (%s)\n    Generated: %s",
                            qa.question[:80],
                            faithfulness_score,
                            faith_reason[:100],
                            generated_answer[:150],
                        )
                except Exception:
                    logger.exception("LLM judge failed for query %d", i)

            # Evaluate
            evaluation = evaluate_query(
                QueryInput(
                    question=qa.question,
                    answer=qa.answer,
                    expected_source=section.filename,
                    retrieved_sources=retrieved_sources,
                    retrieved_scores=retrieved_scores,
                    assembled_context=query_result.context,
                    generated_answer=generated_answer,
                    faithfulness_score=faithfulness_score,
                    correctness_score=correctness_score,
                )
            )
            evaluations.append(evaluation)

            if (i + 1) % 10 == 0:
                logger.info("  Evaluated %d/%d queries...", i + 1, len(qa_pairs))

        report = aggregate_evaluations(evaluations)
        logger.info("\n%s", report.summary())

        # Diagnostic: analyze Answer-in-Context failures
        aic_failures = [e for e in report.per_query if not e.answer_in_context]
        if aic_failures:
            logger.info(
                "\n=== Answer-in-Context Failures: %d/%d ===",
                len(aic_failures),
                report.total_queries,
            )
            hit_but_no_aic = [e for e in aic_failures if e.hit_at_k]
            miss_and_no_aic = [e for e in aic_failures if not e.hit_at_k]
            logger.info(
                "  Retrieved correct chunk but answer NOT in context: %d",
                len(hit_but_no_aic),
            )
            logger.info(
                "  Did NOT retrieve correct chunk (expected): %d",
                len(miss_and_no_aic),
            )

        # Diagnostic: hallucination summary
        if enable_llm_judge:
            hallucinations = [e for e in report.per_query if e.is_hallucination]
            if hallucinations:
                logger.info(
                    "\n=== Hallucinations Detected: %d/%d ===",
                    len(hallucinations),
                    report.total_queries,
                )
                for j, e in enumerate(hallucinations[:5]):
                    logger.info(
                        "\n  [Hallucination %d] Q: %s\n    Faithfulness: %.2f\n    Generated: %s\n    Gold answer: %s",
                        j + 1,
                        e.question[:100],
                        e.faithfulness_score,
                        e.generated_answer[:200],
                        e.answer[:200],
                    )

        return report

    finally:
        await _cleanup_qdrant(collection_name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@skip_no_qdrant
@skip_no_ollama
async def test_rag_bench_ollama(tmp_path: Path) -> None:
    """Evaluate RAG pipeline with Ollama nomic-embed-text embeddings.

    Asserts minimum retrieval quality thresholds:
    - Recall@5 >= 50%
    - MRR >= 0.3
    - Answer-in-Context >= 30%
    """
    report = await _run_evaluation("ollama", tmp_path)

    assert report.total_queries > 0, "No queries evaluated"

    # Log detailed results for failed queries
    missed = [e for e in report.per_query if not e.hit_at_k]
    if missed:
        logger.info(
            "Missed %d/%d queries. First 3:",
            len(missed),
            report.total_queries,
        )
        for e in missed[:3]:
            logger.info(
                "  Q: %s\n  Expected: %s\n  Retrieved: %s",
                e.question[:80],
                e.expected_source,
                e.retrieved_sources[:3],
            )

    # Minimum quality thresholds
    assert report.recall_at_k >= 0.50, f"Recall@{TOP_K} too low: {report.recall_at_k:.2%}"
    assert report.mrr >= 0.30, f"MRR too low: {report.mrr:.4f}"
    assert report.answer_in_context_rate >= 0.30, f"Answer-in-Context too low: {report.answer_in_context_rate:.2%}"


@pytest.mark.integration
@skip_no_qdrant
@skip_no_ollama
async def test_rag_bench_hallucination(tmp_path: Path) -> None:
    """Evaluate RAG pipeline with LLM-as-judge hallucination detection.

    Uses qwen3:14b to:
    1. Generate answers from retrieved context
    2. Judge faithfulness (is the answer grounded in the context?)
    3. Judge correctness (does the answer match the gold answer?)

    Asserts:
    - Hallucination rate < 30%
    - Avg faithfulness >= 0.6
    - Avg correctness >= 0.3
    """
    report = await _run_evaluation(
        "ollama",
        tmp_path,
        enable_llm_judge=True,
    )

    assert report.total_queries > 0, "No queries evaluated"

    # Retrieval should still be good
    assert report.recall_at_k >= 0.50, f"Recall@{TOP_K} too low: {report.recall_at_k:.2%}"

    # Hallucination thresholds
    assert report.hallucination_rate <= 0.30, f"Hallucination rate too high: {report.hallucination_rate:.2%}"
    assert report.avg_faithfulness >= 0.60, f"Avg faithfulness too low: {report.avg_faithfulness:.2%}"
    assert report.avg_correctness >= 0.30, f"Avg correctness too low: {report.avg_correctness:.2%}"


@pytest.mark.integration
@skip_no_qdrant
@skip_no_openai_key
async def test_rag_bench_openai(tmp_path: Path) -> None:
    """Evaluate RAG pipeline with OpenAI text-embedding-3-small.

    OpenAI embeddings typically yield higher quality, so thresholds
    are set higher than the Ollama variant.
    """
    report = await _run_evaluation("openai", tmp_path)

    assert report.total_queries > 0, "No queries evaluated"

    # Higher thresholds for OpenAI
    assert report.recall_at_k >= 0.60, f"Recall@{TOP_K} too low: {report.recall_at_k:.2%}"
    assert report.mrr >= 0.40, f"MRR too low: {report.mrr:.4f}"
    assert report.answer_in_context_rate >= 0.40, f"Answer-in-Context too low: {report.answer_in_context_rate:.2%}"


@pytest.mark.integration
@skip_no_qdrant
@skip_no_ollama
async def test_chunking_comparison(tmp_path: Path) -> None:
    """Compare chunking strategies by running evaluation with different params.

    Tests that smaller chunks with more overlap improve precision.
    """
    # Baseline: larger chunks
    report_large = await _run_evaluation(
        "ollama",
        tmp_path / "large",
        chunk_size=1000,
        chunk_overlap=200,
        max_docs=5,
    )

    # Smaller chunks with more overlap
    report_small = await _run_evaluation(
        "ollama",
        tmp_path / "small",
        chunk_size=300,
        chunk_overlap=100,
        max_docs=5,
    )

    logger.info(
        "Chunking comparison:\n"
        "  Large (1000/200): Recall=%.2f%%, MRR=%.4f, AiC=%.2f%%\n"
        "  Small (300/100):  Recall=%.2f%%, MRR=%.4f, AiC=%.2f%%",
        report_large.recall_at_k * 100,
        report_large.mrr,
        report_large.answer_in_context_rate * 100,
        report_small.recall_at_k * 100,
        report_small.mrr,
        report_small.answer_in_context_rate * 100,
    )

    # Both should meet minimum thresholds
    for report, label in [(report_large, "large"), (report_small, "small")]:
        assert report.recall_at_k >= 0.40, f"{label}: Recall too low: {report.recall_at_k:.2%}"
