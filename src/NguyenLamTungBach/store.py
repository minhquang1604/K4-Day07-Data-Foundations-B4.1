from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._client = None
        self._next_index = 0

        try:
            import chromadb

            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(name=collection_name)
            self._next_index = self._collection.count()
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None
            self._client = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = dict(doc.metadata or {})
        metadata.setdefault("doc_id", doc.id)
        record = {
            "id": doc.id,
            "storage_id": f"{doc.id}::{self._next_index}",
            "content": doc.content,
            "metadata": metadata,
            "embedding": [float(value) for value in self._embedding_fn(doc.content)],
        }
        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0 or not records:
            return []

        query_embedding = self._embedding_fn(query)
        ranked = [
            {
                "id": record["id"],
                "content": record["content"],
                "metadata": dict(record["metadata"]),
                "score": _dot(query_embedding, record["embedding"]),
            }
            for record in records
        ]
        ranked.sort(key=lambda result: result["score"], reverse=True)
        return ranked[:top_k]

    def _get_chroma_records(self) -> list[dict[str, Any]]:
        """Read Chroma records into the same shape used by the memory backend."""
        if self._collection is None:
            return []

        data = self._collection.get(include=["documents", "metadatas", "embeddings"])
        ids = data.get("ids") or []
        documents = data.get("documents")
        metadatas = data.get("metadatas")
        embeddings = data.get("embeddings")
        documents = documents if documents is not None else [""] * len(ids)
        metadatas = metadatas if metadatas is not None else [{}] * len(ids)
        embeddings = embeddings if embeddings is not None else [[]] * len(ids)

        records: list[dict[str, Any]] = []
        for storage_id, content, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            normalized_metadata = dict(metadata or {})
            record_id = normalized_metadata.pop("record_id", storage_id)
            records.append(
                {
                    "id": record_id,
                    "storage_id": storage_id,
                    "content": content or "",
                    "metadata": normalized_metadata,
                    "embedding": [float(value) for value in embedding],
                }
            )
        return records

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        records = [self._make_record(doc) for doc in docs]
        if self._use_chroma and self._collection is not None:
            metadatas = []
            for record in records:
                metadata = dict(record["metadata"])
                metadata["record_id"] = record["id"]
                metadatas.append(metadata)
            self._collection.add(
                ids=[record["storage_id"] for record in records],
                documents=[record["content"] for record in records],
                embeddings=[record["embedding"] for record in records],
                metadatas=metadatas,
            )
        else:
            self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        records = self._get_chroma_records() if self._use_chroma else self._store
        return self._search_records(query, records, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return int(self._collection.count())
        return len(self._store)

    def search_with_filter(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k=top_k)

        records = self._get_chroma_records() if self._use_chroma else self._store
        filtered_records = [
            record
            for record in records
            if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
        ]
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            matching_ids = [
                record["storage_id"]
                for record in self._get_chroma_records()
                if record["metadata"].get("doc_id") == doc_id
            ]
            if not matching_ids:
                return False
            self._collection.delete(ids=matching_ids)
            return True

        original_size = len(self._store)
        self._store = [
            record
            for record in self._store
            if record["metadata"].get("doc_id") != doc_id
        ]
        return len(self._store) < original_size
