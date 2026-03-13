"""Unit tests for RAG evaluation metrics."""

from __future__ import annotations

import pytest

from tests.integration.test_rag_evaluation.metrics import (
    EvaluationReport,
    QueryInput,
    aggregate_evaluations,
    evaluate_query,
)


@pytest.mark.unit
class TestEvaluateQuery:
    """Tests for evaluate_query function."""

    def test_perfect_hit_at_rank_1(self) -> None:
        result = evaluate_query(
            QueryInput(
                question="What is X?",
                answer="X is Y",
                expected_source="doc1.txt",
                retrieved_sources=["doc1.txt", "doc2.txt", "doc3.txt"],
                retrieved_scores=[0.9, 0.7, 0.5],
                assembled_context="X is Y and more context here",
            )
        )
        assert result.hit_at_k is True
        assert result.reciprocal_rank == 1.0
        assert result.precision_at_1 is True
        assert result.answer_in_context is True

    def test_hit_at_rank_3(self) -> None:
        result = evaluate_query(
            QueryInput(
                question="What is X?",
                answer="X is Y",
                expected_source="doc3.txt",
                retrieved_sources=["doc1.txt", "doc2.txt", "doc3.txt"],
                retrieved_scores=[0.9, 0.7, 0.5],
                assembled_context="unrelated context",
            )
        )
        assert result.hit_at_k is True
        assert result.reciprocal_rank == pytest.approx(1 / 3)
        assert result.precision_at_1 is False
        assert result.answer_in_context is False

    def test_complete_miss(self) -> None:
        result = evaluate_query(
            QueryInput(
                question="What is X?",
                answer="X is Y",
                expected_source="doc_missing.txt",
                retrieved_sources=["doc1.txt", "doc2.txt"],
                retrieved_scores=[0.9, 0.7],
                assembled_context="nothing relevant",
            )
        )
        assert result.hit_at_k is False
        assert result.reciprocal_rank == 0.0
        assert result.precision_at_1 is False

    def test_empty_retrieved(self) -> None:
        result = evaluate_query(
            QueryInput(
                question="What is X?",
                answer="X is Y",
                expected_source="doc1.txt",
                retrieved_sources=[],
                retrieved_scores=[],
                assembled_context="",
            )
        )
        assert result.hit_at_k is False
        assert result.reciprocal_rank == 0.0
        assert result.precision_at_1 is False
        assert result.answer_in_context is False

    def test_answer_in_context_word_overlap(self) -> None:
        result = evaluate_query(
            QueryInput(
                question="What are the benefits?",
                answer="The benefits include improved efficiency and reduced costs",
                expected_source="doc1.txt",
                retrieved_sources=["doc1.txt"],
                retrieved_scores=[0.9],
                assembled_context="The benefits include improved efficiency, reduced costs, and better outcomes",
            )
        )
        assert result.answer_in_context is True

    def test_answer_in_context_low_overlap(self) -> None:
        result = evaluate_query(
            QueryInput(
                question="What is X?",
                answer="The quick brown fox jumps over the lazy dog",
                expected_source="doc1.txt",
                retrieved_sources=["doc1.txt"],
                retrieved_scores=[0.9],
                assembled_context="A cat sat on a mat",
            )
        )
        assert result.answer_in_context is False


@pytest.mark.unit
class TestAggregateEvaluations:
    """Tests for aggregate_evaluations function."""

    def test_empty_evaluations(self) -> None:
        report = aggregate_evaluations([])
        assert report.total_queries == 0
        assert report.recall_at_k == 0.0
        assert report.mrr == 0.0

    def test_all_perfect(self) -> None:
        evals = [
            evaluate_query(
                QueryInput(
                    question=f"Q{i}",
                    answer=f"A{i}",
                    expected_source="doc.txt",
                    retrieved_sources=["doc.txt"],
                    retrieved_scores=[0.95],
                    assembled_context=f"A{i} is the answer",
                )
            )
            for i in range(5)
        ]
        report = aggregate_evaluations(evals)
        assert report.total_queries == 5
        assert report.recall_at_k == 1.0
        assert report.mrr == 1.0
        assert report.precision_at_1 == 1.0
        assert report.answer_in_context_rate == 1.0

    def test_mixed_results(self) -> None:
        eval_hit = evaluate_query(
            QueryInput(
                question="Q1",
                answer="A1",
                expected_source="doc1.txt",
                retrieved_sources=["doc1.txt", "doc2.txt"],
                retrieved_scores=[0.9, 0.7],
                assembled_context="A1 context",
            )
        )
        eval_miss = evaluate_query(
            QueryInput(
                question="Q2",
                answer="A2",
                expected_source="doc3.txt",
                retrieved_sources=["doc1.txt", "doc2.txt"],
                retrieved_scores=[0.9, 0.7],
                assembled_context="wrong context",
            )
        )
        report = aggregate_evaluations([eval_hit, eval_miss])
        assert report.total_queries == 2
        assert report.recall_at_k == 0.5
        assert report.mrr == 0.5  # (1.0 + 0.0) / 2
        assert report.precision_at_1 == 0.5

    def test_summary_format(self) -> None:
        report = EvaluationReport(
            total_queries=100,
            recall_at_k=0.85,
            mrr=0.72,
            precision_at_1=0.65,
            answer_in_context_rate=0.78,
        )
        summary = report.summary()
        assert "100 queries" in summary
        assert "85.00%" in summary
        assert "0.7200" in summary
