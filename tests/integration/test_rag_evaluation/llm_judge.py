"""LLM-as-judge for RAG hallucination evaluation.

Uses an LLM to:
1. Generate answers from retrieved context (RAG generation step)
2. Evaluate faithfulness: is the generated answer grounded in the context?
3. Evaluate correctness: does the generated answer match the gold answer?
"""

from __future__ import annotations

import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RAG_ANSWER_PROMPT = """You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information to answer, say "I cannot answer this based on the provided context."

Do NOT use any prior knowledge. Only use the context below.

Context:
{context}

Question: {question}

Answer concisely and directly."""

FAITHFULNESS_PROMPT = """You are an impartial judge evaluating whether an AI-generated \
answer is faithful to the provided context.

An answer is faithful if ALL claims in the answer are supported by the context. \
An answer is NOT faithful if it contains information not in the context (hallucination).

Context:
{context}

Question: {question}

Generated Answer: {generated_answer}

Evaluate the faithfulness of the generated answer. Respond with ONLY a JSON object:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}

Score guide:
- 1.0: Every claim is directly supported by the context
- 0.7-0.9: Most claims supported, minor extrapolations
- 0.4-0.6: Mix of supported and unsupported claims
- 0.1-0.3: Mostly unsupported or fabricated
- 0.0: Completely hallucinated, no basis in context"""

CORRECTNESS_PROMPT = """You are an impartial judge evaluating whether an AI-generated \
answer is semantically correct compared to a reference answer.

Reference Answer (gold standard): {gold_answer}

Generated Answer: {generated_answer}

Question: {question}

Evaluate how correct the generated answer is compared to the reference. Respond with ONLY a JSON object:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}

Score guide:
- 1.0: Semantically equivalent, captures all key information
- 0.7-0.9: Mostly correct, minor omissions or differences
- 0.4-0.6: Partially correct, captures some key points
- 0.1-0.3: Mostly incorrect or irrelevant
- 0.0: Completely wrong"""


# ---------------------------------------------------------------------------
# LLM client (direct httpx to Ollama, avoids framework coupling)
# ---------------------------------------------------------------------------


async def _ollama_generate(
    prompt: str,
    base_url: str,
    model: str,
    temperature: float = 0.1,
    request_timeout: float = 120,
) -> str:
    """Call Ollama generate API and return the response text."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(request_timeout)) as client:
        resp = await client.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


def _parse_judge_response(text: str) -> tuple[float, str]:
    """Extract score and reasoning from LLM judge response."""
    # Try to find JSON in the response
    json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            score = float(data.get("score", 0.0))
            reasoning = data.get("reasoning", "")
            return max(0.0, min(1.0, score)), reasoning
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: try to find a number
    num_match = re.search(r"(\d+\.?\d*)", text)
    if num_match:
        score = float(num_match.group(1))
        return max(0.0, min(1.0, score)), text[:200]

    logger.warning("Could not parse judge response: %s", text[:200])
    return 0.0, "parse_error"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def generate_rag_answer(
    question: str,
    context: str,
    ollama_url: str,
    model: str = "qwen3:14b",
) -> str:
    """Generate an answer using the LLM with retrieved context."""
    prompt = RAG_ANSWER_PROMPT.format(context=context, question=question)
    return await _ollama_generate(prompt, ollama_url, model)


async def evaluate_faithfulness(
    question: str,
    context: str,
    generated_answer: str,
    ollama_url: str,
    model: str = "qwen3:14b",
) -> tuple[float, str]:
    """Evaluate if the generated answer is faithful to the context.

    Returns (score, reasoning) where score is 0.0-1.0.
    """
    prompt = FAITHFULNESS_PROMPT.format(
        context=context,
        question=question,
        generated_answer=generated_answer,
    )
    response = await _ollama_generate(prompt, ollama_url, model, temperature=0.0)
    return _parse_judge_response(response)


async def evaluate_correctness(
    question: str,
    gold_answer: str,
    generated_answer: str,
    ollama_url: str,
    model: str = "qwen3:14b",
) -> tuple[float, str]:
    """Evaluate if the generated answer matches the gold answer.

    Returns (score, reasoning) where score is 0.0-1.0.
    """
    prompt = CORRECTNESS_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )
    response = await _ollama_generate(prompt, ollama_url, model, temperature=0.0)
    return _parse_judge_response(response)
