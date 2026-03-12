"""
Evaluation-oriented tests for retrieval quality on a small WixQA ExpertWritten sample.

These tests are intentionally light-touch and marked as integration/quality because
they depend on:
- network access for HuggingFace datasets
- an ingested WixQA corpus in Postgres with embeddings
"""

from typing import List

import pytest
try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None  # type: ignore[assignment]

from backend.rag.retriever import Retriever


SPLIT_NAME = "wixqa_expertwritten"
DATASET_NAME = "Wix/WixQA"
SAMPLE_SIZE = 20


def _sample_queries() -> List[str]:
    if load_dataset is None:
        pytest.skip("requires datasets dependency")
    ds = load_dataset(DATASET_NAME, SPLIT_NAME, split="train")
    questions: List[str] = []
    for row in ds:
        q = row.get("question") or ""
        if q and q.strip():
            questions.append(q.strip())
        if len(questions) >= SAMPLE_SIZE:
            break
    return questions


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieval_returns_results_for_sample_queries():
    """
    Smoke test: for a small set of real user questions from WixQA ExpertWritten,
    the retriever should return at least one document when corpus is ingested.
    """
    questions = _sample_queries()
    assert questions, "Expected non-empty sample of WixQA ExpertWritten questions"

    retriever = Retriever()

    empty_count = 0
    for q in questions:
        docs = await retriever.search(q, top_k=5, use_cache=False, rerank=False)
        if not docs:
            empty_count += 1

    # Allow a few misses, but majority should return something.
    assert empty_count < len(questions) // 2

