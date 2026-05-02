import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.document_loader import PolicyDocumentLoader
from src.rag_chain import HRPolicyRAG
from src.vector_store import PolicyVectorStore

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

mock_policies = [
    {
        "id": "P1",
        "title": "Leave Policy",
        "category": "Time Off",
        "description": "20 days off.",
        "status": "active",
        "effective_year": 2023,
    },
    {
        "id": "P2",
        "title": "Old Policy",
        "category": "Misc",
        "description": "Obsolete.",
        "status": "inactive",
        "effective_year": 2020,
    },
]

mock_hris = [{"employee_id": "E123", "name": "Alice", "pto_balance": 15, "pto_details": []}]


class TestPolicyDocumentLoader:
    @patch("os.path.exists", return_value=True)
    def test_load_policies_success(self, _):
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_policies))):
            loader = PolicyDocumentLoader("policies.json", "hris.json")
            policies = loader.load_policies()
            assert len(policies) == 2

    def test_load_policies_file_not_found(self):
        with patch("os.path.exists", return_value=False):
            loader = PolicyDocumentLoader("bad_path.json", "hris.json")
            with pytest.raises(FileNotFoundError):
                loader.load_policies()

    @patch.object(PolicyDocumentLoader, "load_policies", return_value=mock_policies)
    @patch.object(PolicyDocumentLoader, "load_hris_data", return_value=mock_hris)
    def test_create_documents_filtering(self, _, __):
        loader = PolicyDocumentLoader("policies.json", "hris.json")
        docs = loader.create_documents()
        assert len(docs) == 2
        assert docs[0].metadata["type"] == "policy"
        assert docs[1].metadata["type"] == "hris"

    @patch.object(
        PolicyDocumentLoader,
        "load_policies",
        return_value=[
            {
                "id": "P3",
                "title": "Long Policy",
                "category": "Compliance",
                "description": " ".join(["rule"] * 500),
                "status": "active",
                "effective_year": 2026,
            }
        ],
    )
    @patch.object(PolicyDocumentLoader, "load_hris_data", return_value=[])
    def test_create_documents_policy_chunking(self, _, __):
        loader = PolicyDocumentLoader("policies.json", "hris.json")
        docs = loader.create_documents()
        assert len(docs) > 1
        assert all(doc.metadata["type"] == "policy" for doc in docs)
        assert all("chunk_id" in doc.metadata for doc in docs)


class TestPolicyVectorStore:
    def test_similarity_search_duplicate_filtering(self):
        store = PolicyVectorStore()
        mock_faiss = MagicMock()
        mock_faiss.similarity_search.return_value = [
            Document(page_content="A", metadata={"title": "Policy A"}),
            Document(page_content="B", metadata={"title": "Policy A"}),
            Document(page_content="C", metadata={"title": "Policy B"}),
        ]
        store.vector_store = mock_faiss
        unique_docs = store.similarity_search_sync("query", k=2)
        assert [doc.metadata["title"] for doc in unique_docs] == ["Policy A", "Policy B"]


@pytest.mark.asyncio
class TestHRPolicyRAG:
    async def test_call_hris_api_found(self):
        rag = HRPolicyRAG(vector_store=MagicMock(), hris_data=mock_hris)
        result = await rag.call_hris_api("E123")
        assert result["name"] == "Alice"

    async def test_call_hris_api_not_found(self):
        rag = HRPolicyRAG(vector_store=MagicMock(), hris_data=mock_hris)
        result = await rag.call_hris_api("E999")
        assert "error" in result

    async def test_query(self):
        mock_store = MagicMock()
        mock_store.similarity_search = AsyncMock(
            return_value=[Document(page_content="Leave policy details.", metadata={"title": "Leave Policy"})]
        )
        rag = HRPolicyRAG(vector_store=mock_store, hris_data=mock_hris)
        rag.chain = MagicMock()
        rag.chain.ainvoke = AsyncMock(return_value="Alice has 15 PTO days.")
        response = await rag.query("How much leave do I have?", employee_id="E123")
        assert response == "Alice has 15 PTO days."

    async def test_query_detects_employee_id_from_question(self):
        mock_store = MagicMock()
        mock_store.similarity_search = AsyncMock(
            return_value=[Document(page_content="Annual leave policy.", metadata={"title": "Annual Leave"})]
        )
        rag = HRPolicyRAG(
            vector_store=mock_store,
            hris_data=[{"employee_id": "E110", "name": "James", "pto_balance": 5, "pto_details": []}],
        )
        response = await rag.query("I am E110 — how many leave days are available for me?")
        assert "5 PTO days" in response
