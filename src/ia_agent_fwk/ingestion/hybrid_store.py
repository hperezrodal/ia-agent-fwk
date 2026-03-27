"""Hybrid vector store for Qdrant (dense + sparse BM25).

Manages a Qdrant collection with two named vector spaces:
  - ``dense``: semantic embeddings (e.g. nomic-embed-text 768d)
  - ``sparse``: BM25 sparse vectors for keyword matching

Search uses Reciprocal Rank Fusion (RRF) to combine both signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """Search strategy."""

    HYBRID = "hybrid"  # dense + sparse with RRF fusion (default)
    DENSE = "dense"  # semantic only
    SPARSE = "sparse"  # keyword/BM25 only


class FusionStrategy(str, Enum):
    """Fusion algorithm for hybrid search."""

    RRF = "rrf"  # Reciprocal Rank Fusion (default, rank-based)
    DBSF = "dbsf"  # Distribution-Based Score Fusion (score-based)


@dataclass
class HybridSearchResult:
    """A single hybrid search result."""

    content: str
    score: float
    metadata: dict[str, Any]


class HybridStore:
    """Qdrant collection with dense + sparse vectors and configurable search.

    Parameters
    ----------
    url:
        Qdrant server URL.
    collection_name:
        Name of the Qdrant collection.
    dense_dim:
        Dimensionality of dense embeddings.
    candidate_multiplier:
        How many candidates to prefetch per signal (top_k * multiplier).
    fusion:
        Fusion strategy for hybrid search (RRF or DBSF).

    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "ia_agent_fwk_rag",
        dense_dim: int = 768,
        embedding_model: str = "",
        candidate_multiplier: int = 5,
        fusion: FusionStrategy = FusionStrategy.RRF,
    ) -> None:
        self._candidate_multiplier = candidate_multiplier
        self._fusion = fusion
        self._client = QdrantClient(url=url, check_compatibility=False)
        self._collection = collection_name
        self._dense_dim = dense_dim
        self._embedding_model = embedding_model
        self._sparse_model = None  # lazy init
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist, with dense + sparse vectors."""
        if self._client.collection_exists(self._collection):
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=self._dense_dim,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )

        # Store embedding model info as a special metadata point
        if self._embedding_model:
            self._client.upsert(
                collection_name=self._collection,
                points=[
                    models.PointStruct(
                        id=0,
                        vector={"dense": [0.0] * self._dense_dim, "sparse": models.SparseVector(indices=[], values=[])},
                        payload={"_meta": True, "embedding_model": self._embedding_model, "dense_dim": self._dense_dim},
                    )
                ],
            )

        logger.info(
            "Created hybrid collection '%s' (dense=%dd + sparse BM25)",
            self._collection,
            self._dense_dim,
        )

    def _get_sparse_model(self):
        """Lazy-init BM25 sparse embedding model."""
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding  # noqa: PLC0415

            self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        return self._sparse_model

    def _sparse_embed(self, text: str) -> models.SparseVector:
        """Generate BM25 sparse vector for a text."""
        model = self._get_sparse_model()
        result = next(iter(model.embed([text])))
        return models.SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist(),
        )

    def _sparse_embed_batch(self, texts: list[str]) -> list[models.SparseVector]:
        """Generate BM25 sparse vectors for a batch of texts."""
        model = self._get_sparse_model()
        results = list(model.embed(texts))
        return [
            models.SparseVector(
                indices=r.indices.tolist(),
                values=r.values.tolist(),
            )
            for r in results
        ]

    def upsert(
        self,
        chunk_id: str,
        content: str,
        dense_embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Store a chunk with both dense and sparse vectors."""
        sparse_vector = self._sparse_embed(content)

        self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=self._make_point_id(chunk_id),
                    vector={
                        "dense": dense_embedding,
                        "sparse": sparse_vector,
                    },
                    payload={
                        "key": chunk_id,
                        "value": content,
                        "metadata": metadata,
                    },
                ),
            ],
        )

    def upsert_batch(
        self,
        chunk_ids: list[str],
        contents: list[str],
        dense_embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Store multiple chunks with both dense and sparse vectors."""
        sparse_vectors = self._sparse_embed_batch(contents)

        points = [
            models.PointStruct(
                id=self._make_point_id(cid),
                vector={
                    "dense": dense_emb,
                    "sparse": sparse_vec,
                },
                payload={
                    "key": cid,
                    "value": content,
                    "metadata": meta,
                },
            )
            for cid, content, dense_emb, sparse_vec, meta in zip(
                chunk_ids,
                contents,
                dense_embeddings,
                sparse_vectors,
                metadatas,
                strict=True,
            )
        ]

        # Batch upsert in groups of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self._collection,
                points=points[i : i + batch_size],
            )

    def search(
        self,
        query: str,
        dense_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, str] | None = None,
        mode: SearchMode | None = None,
    ) -> list[HybridSearchResult]:
        """Search with configurable strategy.

        Parameters
        ----------
        query:
            Query text (used for sparse BM25 encoding).
        dense_embedding:
            Pre-computed dense vector for the query.
        top_k:
            Number of results to return.
        filters:
            Metadata filters (exact match). Applied server-side in Qdrant.
        mode:
            Override the search mode. If None, uses HYBRID.

        """
        search_mode = mode or SearchMode.HYBRID

        # Build Qdrant filter
        qdrant_filter = None
        if filters:
            must_conditions = []
            for k, v in filters.items():
                if k.endswith("__in") and isinstance(v, list):
                    # Multi-value filter: scope__in → match any of the values
                    field_name = k.removesuffix("__in")
                    must_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{field_name}",
                            match=models.MatchAny(any=v),
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{k}",
                            match=models.MatchValue(value=v),
                        )
                    )
            qdrant_filter = models.Filter(must=must_conditions)

        candidate_limit = top_k * self._candidate_multiplier

        if search_mode == SearchMode.DENSE:
            # Semantic only — no BM25
            results = self._client.query_points(
                collection_name=self._collection,
                query=dense_embedding,
                using="dense",
                limit=top_k,
                query_filter=qdrant_filter,
            )

        elif search_mode == SearchMode.SPARSE:
            # BM25 keyword only — no dense
            sparse_vector = self._sparse_embed(query)
            results = self._client.query_points(
                collection_name=self._collection,
                query=sparse_vector,
                using="sparse",
                limit=top_k,
                query_filter=qdrant_filter,
            )

        else:
            # Hybrid: dense + sparse with fusion
            sparse_vector = self._sparse_embed(query)
            fusion_map = {
                FusionStrategy.RRF: models.Fusion.RRF,
                FusionStrategy.DBSF: models.Fusion.DBSF,
            }
            results = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_embedding,
                        using="dense",
                        limit=candidate_limit,
                        filter=qdrant_filter,
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using="sparse",
                        limit=candidate_limit,
                        filter=qdrant_filter,
                    ),
                ],
                query=models.FusionQuery(
                    fusion=fusion_map.get(self._fusion, models.Fusion.RRF),
                ),
                limit=top_k,
            )

        return [
            HybridSearchResult(
                content=pt.payload.get("value", ""),
                score=pt.score,
                metadata=pt.payload.get("metadata", {}),
            )
            for pt in results.points
        ]

    def get_stored_model(self) -> str | None:
        """Read the embedding model name stored in the collection.

        Returns the model name (e.g. "bge-m3") or None if not found.
        """
        if not self._client.collection_exists(self._collection):
            return None
        try:
            points = self._client.retrieve(
                collection_name=self._collection,
                ids=[0],
                with_payload=True,
            )
            if points and points[0].payload.get("_meta"):
                return points[0].payload.get("embedding_model")
        except Exception:  # noqa: BLE001
            logger.debug("Failed to detect embedding model from collection")
        return None

    def delete_collection(self) -> None:
        """Delete the collection."""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
            logger.info("Deleted collection '%s'", self._collection)

    def collection_info(self) -> dict[str, Any]:
        """Get collection stats."""
        info = self._client.get_collection(self._collection)
        return {
            "points_count": info.points_count,
            "vectors_count": info.indexed_vectors_count,
            "status": info.status.value,
        }

    @staticmethod
    def _make_point_id(chunk_id: str) -> int:
        """Deterministic int ID from chunk_id string."""
        import hashlib  # noqa: PLC0415

        return int(hashlib.md5(chunk_id.encode()).hexdigest()[:16], 16)  # noqa: S324
