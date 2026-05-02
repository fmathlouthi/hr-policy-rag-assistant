import json
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import (
    HYBRID_DENSE_WEIGHT,
    HYBRID_KEYWORD_WEIGHT,
    INDEX_VERSION,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBEDDING_MODEL,
    RERANK_TOP_N,
)


class PolicyVectorStore:
    def __init__(self, index_dir: Optional[str] = None, index_version: str = INDEX_VERSION):
        self.embeddings = OpenAIEmbeddings(
            model=OPENROUTER_EMBEDDING_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self.vector_store = None
        self.keyword_index: Dict[str, List[str]] = {}
        self.doc_lookup: Dict[str, Document] = {}
        self.index_dir = Path(index_dir) if index_dir else None
        self.index_version = index_version
        self.manifest_name = "index_manifest.json"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _doc_key(self, doc: Document) -> str:
        return doc.metadata.get("doc_hash") or sha256(
            f"{doc.metadata.get('id')}::{doc.metadata.get('employee_id')}::{doc.page_content}".encode("utf-8")
        ).hexdigest()

    def _get_all_docs(self) -> List[Document]:
        if not self.vector_store:
            return []
        store = getattr(self.vector_store.docstore, "_dict", {})
        return [doc for doc in store.values() if isinstance(doc, Document)]

    def _rebuild_keyword_index(self) -> None:
        self.keyword_index = {}
        self.doc_lookup = {}
        for doc in self._get_all_docs():
            key = self._doc_key(doc)
            self.doc_lookup[key] = doc
            corpus = f"{doc.page_content} {json.dumps(doc.metadata, sort_keys=True)}"
            self.keyword_index[key] = self._tokenize(corpus)

    def _keyword_search(self, query: str, k: int = 10) -> List[Document]:
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []
        scored = []
        for key, tokens in self.keyword_index.items():
            if not tokens:
                continue
            overlap = len(query_tokens.intersection(tokens))
            if overlap:
                scored.append((overlap / len(query_tokens), key))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self.doc_lookup[key] for _, key in scored[:k]]

    def _rrf_merge(self, dense_docs: List[Document], keyword_docs: List[Document], k: int) -> List[Document]:
        scores: Dict[str, float] = {}
        for rank, doc in enumerate(dense_docs, start=1):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0.0) + HYBRID_DENSE_WEIGHT * (1.0 / (60 + rank))
            self.doc_lookup[key] = doc
        for rank, doc in enumerate(keyword_docs, start=1):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0.0) + HYBRID_KEYWORD_WEIGHT * (1.0 / (60 + rank))
            self.doc_lookup[key] = doc
        ranked_keys = [key for key, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
        return [self.doc_lookup[key] for key in ranked_keys[: max(k, RERANK_TOP_N)]]

    def _rerank(self, query: str, docs: List[Document], k: int) -> List[Document]:
        query_tokens = set(self._tokenize(query))
        scored: List[tuple] = []
        for doc in docs:
            text = doc.page_content.lower()
            doc_tokens = set(self._tokenize(text))
            lexical = len(query_tokens.intersection(doc_tokens))
            phrase_bonus = 2 if query.lower() in text else 0
            scored.append((lexical + phrase_bonus, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def create_vector_store(self, documents: List[Document]) -> None:
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self._rebuild_keyword_index()

    def _manifest_path(self) -> Optional[Path]:
        if not self.index_dir:
            return None
        return self.index_dir / self.manifest_name

    def _load_manifest(self) -> Dict:
        path = self._manifest_path()
        if not path or not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_manifest(self, source_fingerprint: str) -> None:
        if not self.index_dir:
            return
        self.index_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "index_version": self.index_version,
            "source_fingerprint": source_fingerprint,
            "updated_at": datetime.utcnow().isoformat(),
            "doc_keys": sorted(self.keyword_index.keys()),
        }
        self._manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def build_or_load_index(self, documents: List[Document], source_fingerprint: str) -> None:
        if not self.index_dir:
            self.create_vector_store(documents)
            return
        self.index_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest()
        can_reuse = (
            manifest.get("index_version") == self.index_version
            and manifest.get("source_fingerprint") == source_fingerprint
            and (self.index_dir / "index.faiss").exists()
            and (self.index_dir / "index.pkl").exists()
        )

        if can_reuse:
            loaded = PolicyVectorStore.load_local(str(self.index_dir))
            self.vector_store = loaded.vector_store
            self._rebuild_keyword_index()
            return

        if (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists():
            loaded = PolicyVectorStore.load_local(str(self.index_dir))
            self.vector_store = loaded.vector_store
            self._rebuild_keyword_index()
            known = set(self.keyword_index.keys())
            new_docs = [doc for doc in documents if self._doc_key(doc) not in known]
            if new_docs:
                self.vector_store.add_documents(new_docs)
                self._rebuild_keyword_index()
        else:
            self.create_vector_store(documents)

        self.save_local(str(self.index_dir))
        self._save_manifest(source_fingerprint)

    async def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vector_store:
            raise ValueError("Vector store is not initialized")

        dense_results = await self.vector_store.asimilarity_search(query, k=max(k * 4, RERANK_TOP_N))
        keyword_results = self._keyword_search(query, k=max(k * 4, RERANK_TOP_N))
        merged_results = self._rrf_merge(dense_results, keyword_results, k=max(k * 4, RERANK_TOP_N))
        results = self._rerank(query, merged_results, k=max(k * 3, RERANK_TOP_N))
        unique_docs: List[Document] = []
        seen_titles = set()

        for doc in results:
            title = doc.metadata.get("title", "")
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)
            if len(unique_docs) == k:
                break
        return unique_docs

    def similarity_search_sync(self, query: str, k: int = 5) -> List[Document]:
        if not self.vector_store:
            raise ValueError("Vector store is not initialized")

        dense_results = self.vector_store.similarity_search(query, k=max(k * 4, RERANK_TOP_N))
        keyword_results = self._keyword_search(query, k=max(k * 4, RERANK_TOP_N))
        merged_results = self._rrf_merge(dense_results, keyword_results, k=max(k * 4, RERANK_TOP_N))
        results = self._rerank(query, merged_results, k=max(k * 3, RERANK_TOP_N))
        unique_docs: List[Document] = []
        seen_titles = set()
        for doc in results:
            title = doc.metadata.get("title", "")
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)
            if len(unique_docs) == k:
                break
        return unique_docs

    def save_local(self, path: str) -> None:
        if not self.vector_store:
            raise ValueError("Vector store is not initialized")
        self.vector_store.save_local(path)

    @classmethod
    def load_local(cls, path: str):
        instance = cls(index_dir=path)
        instance.vector_store = FAISS.load_local(
            path,
            instance.embeddings,
            allow_dangerous_deserialization=True,
        )
        instance._rebuild_keyword_index()
        return instance
