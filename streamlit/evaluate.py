import asyncio
from pathlib import Path

from src.document_loader import PolicyDocumentLoader
from src.evaluation import RetrievalExample, evaluate_retrieval
from src.vector_store import PolicyVectorStore


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
POLICY_FILE = DATA_DIR / "policies.json"
HRIS_FILE = DATA_DIR / "hris.json"


async def main() -> None:
    loader = PolicyDocumentLoader(str(POLICY_FILE), str(HRIS_FILE))
    docs = loader.create_documents()
    vector_store = PolicyVectorStore()
    vector_store.create_vector_store(docs)

    examples = [
        RetrievalExample(query="remote work days per week", relevant_ids={"P-001"}),
        RetrievalExample(query="sick leave days annually", relevant_ids={"P-003"}),
        RetrievalExample(query="employee E110 PTO balance", relevant_ids={"E110"}),
    ]
    metrics = await evaluate_retrieval(vector_store, examples, k=5)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
