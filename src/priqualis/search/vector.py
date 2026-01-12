"""
Vector Store for Priqualis.

Qdrant-based vector storage for semantic search.
"""

import logging
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from priqualis.core.config import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class EmbeddingModel(Protocol):
    """Protocol for embedding generation."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        ...


# =============================================================================
# Embedding Service
# =============================================================================


class EmbeddingService:
    """
    Generate embeddings using sentence-transformers.

    Uses e5-small or similar multilingual model.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """
        Initialize embedding service.

        Args:
            model_name: HuggingFace model name
            device: "cpu" or "cuda"
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def model(self) -> Any:
        """Lazy load model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed single text with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check cache
        if text in self._cache:
            return self._cache[text]

        # Encode
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        self._cache[text] = embedding
        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Batch embedding for efficiency.

        Args:
            texts: List of texts to embed

        Returns:
            2D array of embeddings (n_texts, dimension)
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )

    def claim_to_text(self, claim: dict[str, Any]) -> str:
        """
        Convert claim to searchable text representation.

        Format: "JGP:{code} ICD10:{main} {secondary} PROC:{procedures}"
        """
        parts = []

        # JGP
        jgp = claim.get("jgp_code")
        if jgp:
            parts.append(f"JGP:{jgp}")

        # ICD-10
        icd_main = claim.get("icd10_main")
        if icd_main:
            parts.append(f"ICD10:{icd_main}")

        icd_secondary = claim.get("icd10_secondary", [])
        if icd_secondary:
            if isinstance(icd_secondary, str):
                icd_secondary = [s.strip() for s in icd_secondary.split(",") if s.strip()]
            for code in icd_secondary[:5]:  # Limit to first 5
                parts.append(f"ICD10:{code}")

        # Procedures
        procedures = claim.get("procedures", [])
        if procedures:
            if isinstance(procedures, str):
                procedures = [p.strip() for p in procedures.split(",") if p.strip()]
            for proc in procedures[:5]:  # Limit to first 5
                parts.append(f"PROC:{proc}")

        # Department
        dept = claim.get("department_code")
        if dept:
            parts.append(f"DEPT:{dept}")

        return " ".join(parts) if parts else "EMPTY"

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()
        logger.debug("Cleared embedding cache")


# =============================================================================
# Vector Store (Qdrant)
# =============================================================================


class VectorStore:
    """
    Qdrant vector store for ANN search.

    Supports both remote Qdrant server and in-memory mode for development.
    """

    def __init__(
        self,
        collection: str = "claims",
        host: str | None = None,
        port: int = 6333,
        api_key: str | None = None,
        in_memory: bool = False,
    ):
        """
        Initialize vector store.

        Args:
            collection: Collection name
            host: Qdrant host (None for in-memory)
            port: Qdrant port
            api_key: Optional API key
            in_memory: Use in-memory storage (for dev/testing)
        """
        self.collection = collection
        self.host = host
        self.port = port
        self.api_key = api_key
        self.in_memory = in_memory
        self._client = None

    @property
    def client(self):
        """Lazy initialize Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient

            if self.in_memory or self.host is None:
                logger.info("Using in-memory Qdrant")
                self._client = QdrantClient(":memory:")
            else:
                logger.info("Connecting to Qdrant at %s:%d", self.host, self.port)
                self._client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key,
                )
        return self._client

    def create_collection(self, vector_size: int = 384) -> None:
        """
        Create collection with HNSW index.

        Args:
            vector_size: Embedding dimension (384 for e5-small)
        """
        from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

        # Check if exists
        collections = self.client.get_collections().collections
        if any(c.name == self.collection for c in collections):
            logger.info("Collection '%s' already exists", self.collection)
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )
        logger.info("Created collection '%s' with %d dimensions", self.collection, vector_size)

    def upsert(self, case_id: str, vector: np.ndarray, payload: dict[str, Any]) -> None:
        """
        Insert or update single vector.

        Args:
            case_id: Unique identifier
            vector: Embedding vector
            payload: Metadata payload
        """
        from qdrant_client.models import PointStruct

        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=hash(case_id) % (2**63),  # Convert to int ID
                    vector=vector.tolist(),
                    payload={"case_id": case_id, **payload},
                )
            ],
        )

    def upsert_batch(
        self,
        items: list[tuple[str, np.ndarray, dict]],
        batch_size: int = 100,
    ) -> int:
        """
        Batch upsert for efficiency.

        Args:
            items: List of (case_id, vector, payload) tuples
            batch_size: Upload batch size

        Returns:
            Number of points upserted
        """
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=hash(case_id) % (2**63),
                vector=vector.tolist(),
                payload={"case_id": case_id, **payload},
            )
            for case_id, vector, payload in items
        ]

        # Upload in batches
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection, points=batch)

        logger.debug("Upserted %d points to collection '%s'", len(points), self.collection)
        return len(points)

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """
        ANN search with optional payload filters.

        Args:
            vector: Query embedding
            top_k: Number of results
            filters: Payload filters (optional)

        Returns:
            List of (case_id, score, payload) tuples
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Build filter if provided
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            query_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
        )

        return [
            (r.payload.get("case_id", "unknown"), r.score, r.payload)
            for r in results
        ]

    def delete(self, case_ids: list[str]) -> None:
        """Remove vectors by case_id."""
        from qdrant_client.models import PointIdsList

        ids = [hash(cid) % (2**63) for cid in case_ids]
        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=ids),
        )
        logger.debug("Deleted %d points", len(ids))

    def count(self) -> int:
        """Get number of points in collection."""
        info = self.client.get_collection(self.collection)
        return info.points_count


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Get cached embedding service instance."""
    settings = get_settings()
    return EmbeddingService(
        model_name=getattr(settings, "embedding_model", "intfloat/multilingual-e5-small"),
        device=getattr(settings, "embedding_device", "cpu"),
    )


@lru_cache
def get_vector_store() -> VectorStore:
    """Get cached vector store instance."""
    settings = get_settings()
    return VectorStore(
        collection=getattr(settings, "qdrant_collection", "claims"),
        host=getattr(settings, "qdrant_host", None),
        port=getattr(settings, "qdrant_port", 6333),
        in_memory=getattr(settings, "qdrant_in_memory", True),
    )
