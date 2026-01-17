from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .embedder import EmbeddingModel
from ..utils.hash import id_from_text
from ..schemas import ChunkRecord, NoteRecord

CHUNK_COLLECTION = "obsidian_chunks"
NOTE_COLLECTION = "obsidian_notes"


class VectorStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        prefer_grpc: bool = False,
        model_name: str = "openai/text-embedding-3-large",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        vault_id: Optional[str] = None,
    ) -> None:
        self.client = QdrantClient(host=host, port=port, prefer_grpc=prefer_grpc)
        self.embedder = EmbeddingModel(model_name, api_key=api_key, base_url=base_url)
        self.vault_id = vault_id
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        collections = {collection.name for collection in self.client.get_collections().collections}

        if CHUNK_COLLECTION not in collections:
            self.client.recreate_collection(
                collection_name=CHUNK_COLLECTION,
                vectors_config=qm.VectorParams(
                    size=self.embedder.dim,
                    distance=qm.Distance.COSINE
                )
            )

        if NOTE_COLLECTION not in collections:
            self.client.recreate_collection(
                collection_name=NOTE_COLLECTION,
                vectors_config=qm.VectorParams(
                    size=self.embedder.dim,
                    distance=qm.Distance.COSINE
                )
            )

    def upsert_chunks(self, chunks: Iterable[ChunkRecord]) -> None:
        chunks = list(chunks)
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        vectors = self.embedder.encode(texts)

        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = {
                "chunk_id": chunk.chunk_id,
                "note_id": chunk.note_id,
                "block_id": chunk.block_id,
                "index": chunk.index,
                "links_to_chunks": chunk.links_to_chunks,
            }
            if self.vault_id:
                payload["vault_id"] = self.vault_id
            points.append(qm.PointStruct(
                id=id_from_text(chunk.chunk_id),
                vector=vector.astype(float).tolist(),
                payload=payload
            ))

        self.client.upsert(
            collection_name=CHUNK_COLLECTION,
            points=points
        )

    def aggregate_note_embedding(self, chunks: List[ChunkRecord]) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        if not texts:
            return np.zeros(self.embedder.dim, dtype=float)
        chunk_vectors = self.embedder.encode(texts)
        return chunk_vectors.mean(axis=0)

    def upsert_note(self, note: NoteRecord, embedding: np.ndarray) -> None:
        payload = {
            "note_id": note.note_id,
            "path": note.path,
            "title": note.title,
            "block_ids": note.block_ids,
            "chunk_ids": note.chunk_ids,
            "links_to_notes": note.links_to_notes,
        }
        if self.vault_id:
            payload["vault_id"] = self.vault_id

        point = qm.PointStruct(
            id=id_from_text(note.note_id),
            vector=embedding.astype(float).tolist(),
            payload=payload
        )

        self.client.upsert(
            collection_name=NOTE_COLLECTION,
            points=[point]
        )

    def search_chunks(
        self,
        query: str,
        top_k: int = 10,
        note_ids: Optional[List[str]] = None,
        block_ids: Optional[List[str]] = None,
    ) -> List[qm.ScoredPoint]:
        query_vec = self.embedder.encode_one(query).flatten()

        must = []

        if self.vault_id:
            must.append(qm.FieldCondition(
                key="vault_id",
                match=qm.MatchValue(value=self.vault_id)
            ))

        if note_ids is not None:
            must.append(qm.FieldCondition(
                key="note_id",
                match=qm.MatchAny(any=note_ids)
            ))

        if block_ids is not None:
            must.append(qm.FieldCondition(
                key="block_ids",
                match=qm.MatchAny(any=block_ids)
            ))

        query_filter = qm.Filter(must=must) if must else None

        result = self.client.query_points(
            collection_name=CHUNK_COLLECTION,
            query=query_vec.astype(float).tolist(),
            limit=top_k,
            query_filter=query_filter
        ).points
        return result

    def search_notes(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[qm.ScoredPoint]:
        query_vec = self.embedder.encode_one(query).flatten()

        must = []
        if self.vault_id:
            must.append(qm.FieldCondition(
                key="vault_id",
                match=qm.MatchValue(value=self.vault_id)
            ))

        query_filter = qm.Filter(must=must) if must else None

        result = self.client.query_points(
            collection_name=NOTE_COLLECTION,
            query=query_vec.astype(float).tolist(),
            limit=top_k,
            query_filter=query_filter
        ).points
        return result

    def delete_chunks(self, chunk_ids: List[str]) -> None:
        if not chunk_ids:
            return
        self.client.delete(
            collection_name=CHUNK_COLLECTION,
            points_selector=qm.PointIdsList(points=[id_from_text(cid) for cid in chunk_ids])
        )

    def delete_notes(self, note_ids: List[str]) -> None:
        if not note_ids:
            return
        self.client.delete(
            collection_name=NOTE_COLLECTION,
            points_selector=qm.PointIdsList(points=[id_from_text(nid) for nid in note_ids])
        )
