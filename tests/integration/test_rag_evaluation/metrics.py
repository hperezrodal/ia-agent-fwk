"""RAG retrieval evaluation metrics.

Provides standard IR metrics for evaluating retrieval quality against
a gold-standard dataset of questions with known source documents.
Includes hallucination detection via LLM-as-judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryEvaluation:
    """Result of evaluating a single query against retrieved chunks."""

    question: str
    expected_source: str
    retrieved_sources: list[str]
    retrieved_scores: list[float]
    answer: str
    assembled_context: str
    hit_at_k: bool = False
    reciprocal_rank: float = 0.0
    precision_at_1: bool = False
    answer_in_context: bool = False
    # Hallucination metrics (populated when LLM evaluation is enabled)
    generated_answer: str = ""
    faithfulness_score: float = 0.0  # 0-1: is generated answer grounded in context?
    correctness_score: float = 0.0  # 0-1: does generated answer match gold answer?
    is_hallucination: bool = False  # True if faithfulness < threshold


@dataclass
class EvaluationReport:
    """Aggregated evaluation metrics over a set of queries."""

    total_queries: int = 0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    precision_at_1: float = 0.0
    answer_in_context_rate: float = 0.0
    # Hallucination metrics
    avg_faithfulness: float = 0.0
    avg_correctness: float = 0.0
    hallucination_rate: float = 0.0
    per_query: list[QueryEvaluation] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary of the evaluation."""
        lines = [
            f"RAG Evaluation Report ({self.total_queries} queries)",
            "=" * 50,
            "  Retrieval Metrics:",
            f"    Recall@k:            {self.recall_at_k:.2%}",
            f"    MRR:                 {self.mrr:.4f}",
            f"    Precision@1:         {self.precision_at_1:.2%}",
            f"    Answer-in-Context:   {self.answer_in_context_rate:.2%}",
        ]
        if self.avg_faithfulness > 0 or self.avg_correctness > 0:
            lines.extend(
                [
                    "  Hallucination Metrics:",
                    f"    Avg Faithfulness:    {self.avg_faithfulness:.2%}",
                    f"    Avg Correctness:     {self.avg_correctness:.2%}",
                    f"    Hallucination Rate:  {self.hallucination_rate:.2%}",
                ]
            )
        return "\n".join(lines)


@dataclass
class QueryInput:
    """Input data for evaluating a single query."""

    question: str
    answer: str
    expected_source: str
    retrieved_sources: list[str]
    retrieved_scores: list[float]
    assembled_context: str
    # Optional: populated when LLM evaluation is enabled
    generated_answer: str = ""
    faithfulness_score: float = 0.0
    correctness_score: float = 0.0


def evaluate_query(query_input: QueryInput) -> QueryEvaluation:
    """Evaluate a single query's retrieval results.

    Parameters
    ----------
    query_input:
        All inputs for the evaluation bundled in a single object.

    Returns
    -------
    QueryEvaluation
        Per-query evaluation with hit/miss, RR, answer-in-context, and hallucination.

    """
    # Recall@k: is the expected source in any of the retrieved chunks?
    hit_at_k = query_input.expected_source in query_input.retrieved_sources

    # MRR: reciprocal rank of the first hit
    reciprocal_rank = 0.0
    for i, src in enumerate(query_input.retrieved_sources):
        if src == query_input.expected_source:
            reciprocal_rank = 1.0 / (i + 1)
            break

    # Precision@1: is the top-1 result from the expected source?
    precision_at_1 = bool(
        query_input.retrieved_sources and query_input.retrieved_sources[0] == query_input.expected_source
    )

    # Answer-in-Context: does the assembled context contain the answer?
    answer_in_context = _check_answer_in_context(query_input.answer, query_input.assembled_context)

    # Hallucination: faithfulness < 0.5 means the LLM fabricated information
    is_hallucination = query_input.faithfulness_score < 0.5 if query_input.generated_answer else False

    return QueryEvaluation(
        question=query_input.question,
        expected_source=query_input.expected_source,
        retrieved_sources=query_input.retrieved_sources,
        retrieved_scores=query_input.retrieved_scores,
        answer=query_input.answer,
        assembled_context=query_input.assembled_context,
        hit_at_k=hit_at_k,
        reciprocal_rank=reciprocal_rank,
        precision_at_1=precision_at_1,
        answer_in_context=answer_in_context,
        generated_answer=query_input.generated_answer,
        faithfulness_score=query_input.faithfulness_score,
        correctness_score=query_input.correctness_score,
        is_hallucination=is_hallucination,
    )


def aggregate_evaluations(evaluations: list[QueryEvaluation]) -> EvaluationReport:
    """Aggregate per-query evaluations into an overall report.

    Parameters
    ----------
    evaluations:
        List of per-query evaluation results.

    Returns
    -------
    EvaluationReport
        Aggregated metrics.

    """
    n = len(evaluations)
    if n == 0:
        return EvaluationReport()

    # Hallucination metrics (only from queries that have generated answers)
    with_generation = [e for e in evaluations if e.generated_answer]
    n_gen = len(with_generation)

    return EvaluationReport(
        total_queries=n,
        recall_at_k=sum(1 for e in evaluations if e.hit_at_k) / n,
        mrr=sum(e.reciprocal_rank for e in evaluations) / n,
        precision_at_1=sum(1 for e in evaluations if e.precision_at_1) / n,
        answer_in_context_rate=sum(1 for e in evaluations if e.answer_in_context) / n,
        avg_faithfulness=sum(e.faithfulness_score for e in with_generation) / n_gen if n_gen else 0.0,
        avg_correctness=sum(e.correctness_score for e in with_generation) / n_gen if n_gen else 0.0,
        hallucination_rate=sum(1 for e in with_generation if e.is_hallucination) / n_gen if n_gen else 0.0,
        per_query=evaluations,
    )


def _check_answer_in_context(answer: str, context: str) -> bool:
    """Check whether the answer is present in the assembled context.

    Uses a sliding-window approach: if at least 60% of the answer's
    words (in order) appear in the context, it's considered a match.
    Also tries direct substring match first.
    """
    norm_answer = answer.lower().strip()
    norm_context = context.lower().strip()

    # Direct substring match
    if norm_answer in norm_context:
        return True

    # Word-overlap heuristic: check if significant portion of answer words
    # appear in the context (handles minor formatting differences)
    answer_words = norm_answer.split()
    if not answer_words:
        return False

    context_words_set = set(norm_context.split())
    matches = sum(1 for w in answer_words if w in context_words_set)
    return matches / len(answer_words) >= 0.6
