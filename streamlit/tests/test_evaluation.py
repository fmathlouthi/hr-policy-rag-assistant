import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import (
    RetrievalExample,
    evaluate_retrieval,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_metric_functions():
    retrieved = ["P-001", "P-003", "P-010"]
    relevant = {"P-003", "P-009"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5


@pytest.mark.asyncio
async def test_evaluate_retrieval_aggregate():
    class MockVectorStore:
        async def similarity_search(self, query: str, k: int = 5):
            if "remote" in query:
                return [Document(page_content="x", metadata={"id": "P-001"})]
            return [Document(page_content="x", metadata={"employee_id": "E110"})]

    examples = [
        RetrievalExample(query="remote work", relevant_ids={"P-001"}),
        RetrievalExample(query="pto E110", relevant_ids={"E110"}),
    ]
    metrics = await evaluate_retrieval(MockVectorStore(), examples, k=5)
    assert metrics["precision@k"] == 1.0
    assert metrics["recall@k"] == 1.0
    assert metrics["mrr"] == 1.0
