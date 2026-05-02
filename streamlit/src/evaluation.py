from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Sequence, Set


@dataclass
class RetrievalExample:
    query: str
    relevant_ids: Set[str]


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    top_k = list(retrieved_ids)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = list(retrieved_ids)[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _doc_identifier(metadata: Dict) -> str:
    return (
        str(metadata.get("id"))
        if metadata.get("id") is not None
        else str(metadata.get("employee_id") or metadata.get("title") or "")
    )


async def evaluate_retrieval(vector_store, examples: List[RetrievalExample], k: int = 5) -> Dict[str, float]:
    if not examples:
        return {"precision@k": 0.0, "recall@k": 0.0, "mrr": 0.0}

    precision_scores: List[float] = []
    recall_scores: List[float] = []
    mrr_scores: List[float] = []

    for example in examples:
        docs = await vector_store.similarity_search(example.query, k=k)
        retrieved_ids = [_doc_identifier(doc.metadata) for doc in docs]
        precision_scores.append(precision_at_k(retrieved_ids, example.relevant_ids, k))
        recall_scores.append(recall_at_k(retrieved_ids, example.relevant_ids, k))
        mrr_scores.append(reciprocal_rank(retrieved_ids, example.relevant_ids))

    return {
        "precision@k": mean(precision_scores),
        "recall@k": mean(recall_scores),
        "mrr": mean(mrr_scores),
    }
