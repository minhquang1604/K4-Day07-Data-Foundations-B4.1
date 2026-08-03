from __future__ import annotations

import os
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

            persist_dir = os.getenv("CHROMA_PERSIST_DIR")
            self._client = (
                chromadb.PersistentClient(path=persist_dir)
                if persist_dir
                else chromadb.EphemeralClient()
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "ip"},
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._client = None
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = dict(doc.metadata)
        # Ingested chunks already carry their parent doc_id.  Plain Documents
        # use their own ID so delete_document works in both cases.
        metadata.setdefault("doc_id", doc.id)
        record = {
            "id": f"{doc.id}::{self._next_index}",
            "document_id": doc.id,
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
        ranked = sorted(
            records,
            key=lambda record: _dot(query_embedding, record["embedding"]),
            reverse=True,
        )
        return [
            {
                "id": record["document_id"],
                "content": record["content"],
                "metadata": dict(record["metadata"]),
                "score": float(_dot(query_embedding, record["embedding"])),
            }
            for record in ranked[:top_k]
        ]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        records = [self._make_record(doc) for doc in docs]
        if not records:
            return

        if self._use_chroma and self._collection is not None:
            try:
                self._collection.add(
                    ids=[record["id"] for record in records],
                    documents=[record["content"] for record in records],
                    embeddings=[record["embedding"] for record in records],
                    metadatas=[record["metadata"] for record in records],
                )
            except Exception:
                # Chroma is an optional backend.  If it rejects an unsupported
                # metadata value, retain full functionality in memory.
                self._use_chroma = False
                self._collection = None
                self._client = None

        self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k=top_k)

        candidates = [
            record
            for record in self._store
            if all(
                record["metadata"].get(key) == expected
                for key, expected in metadata_filter.items()
            )
        ]
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        record_ids = [
            record["id"]
            for record in self._store
            if record["metadata"].get("doc_id") == doc_id
        ]
        if not record_ids:
            return False

        ids_to_remove = set(record_ids)
        self._store = [
            record for record in self._store if record["id"] not in ids_to_remove
        ]
        if self._use_chroma and self._collection is not None:
            try:
                self._collection.delete(ids=record_ids)
            except Exception:
                # The in-memory store is the canonical runtime copy, so the
                # requested deletion has still succeeded for this instance.
                pass
        return True
