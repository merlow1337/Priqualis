"""
Similarity Service for Priqualis.

High-level service for finding similar approved cases.
"""

import logging
from typing import Any

from priqualis.search.bm25 import BM25Index
from priqualis.search.hybrid import HybridSearch
from priqualis.search.models import (
    AttributeDiff,
    CaseStatus,
    DiffType,
    SearchQuery,
    SimilarCase,
)
from priqualis.search.rerank import Reranker, get_reranker
from priqualis.search.vector import EmbeddingService, VectorStore

logger = logging.getLogger(__name__)


# =============================================================================
# Similarity Service
# =============================================================================


class SimilarityService:
    """
    High-level service for finding similar approved cases.

    Integrates hybrid search with attribute diff computation
    for AutoFix suggestions.
    """

    def __init__(
        self,
        bm25_index: BM25Index | None = None,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        reranker: Reranker | None = None,
        alpha: float = 0.5,
    ):
        """
        Initialize similarity service.

        Args:
            bm25_index: BM25 index (created if None)
            vector_store: Vector store (created if None)
            embedding_service: Embedding service (created if None)
            reranker: Optional reranker
            alpha: BM25 weight for fusion
        """
        self.bm25 = bm25_index or BM25Index()
        self.vectors = vector_store or VectorStore(in_memory=True)
        self.embeddings = embedding_service or EmbeddingService()
        self.reranker = reranker

        self.hybrid = HybridSearch(
            bm25_index=self.bm25,
            vector_store=self.vectors,
            embedding_service=self.embeddings,
            alpha=alpha,
        )

        self._approved_cases: dict[str, dict] = {}

    def load_approved_cases(self, cases: list[dict]) -> int:
        """
        Load approved cases for similarity lookup.

        Args:
            cases: List of approved claim records

        Returns:
            Number of cases loaded
        """
        self._approved_cases = {c["case_id"]: c for c in cases if "case_id" in c}
        self.hybrid.set_claim_cache(self._approved_cases)
        logger.info("Loaded %d approved cases", len(self._approved_cases))
        return len(self._approved_cases)

    def find_similar(
        self,
        claim: dict,
        top_k: int = 5,
        include_diffs: bool = True,
    ) -> list[SimilarCase]:
        """
        Find top-k similar approved cases for a claim.

        Args:
            claim: Query claim record
            top_k: Number of results
            include_diffs: Whether to compute attribute differences

        Returns:
            List of SimilarCase with attribute diffs
        """
        # Build query
        query = SearchQuery.from_claim(claim)

        # Hybrid search
        results = self.hybrid.search(query, top_k=top_k * 2)  # Get more for reranking

        # Optional reranking
        if self.reranker and len(results) > top_k:
            results = self.reranker.rerank(query.text, results, top_k=top_k)
        else:
            results = results[:top_k]

        # Convert to SimilarCase with diffs
        similar_cases = []
        for result in results:
            match_data = result.claim_data or self._approved_cases.get(result.case_id, {})

            diffs = []
            if include_diffs and match_data:
                diffs = self._compute_attribute_diffs(claim, match_data)

            similar_cases.append(SimilarCase(
                case_id=result.case_id,
                similarity_score=result.score,
                attribute_diffs=diffs,
                jgp_code=match_data.get("jgp_code"),
                status=CaseStatus.APPROVED,
                claim_data=match_data,
            ))

        return similar_cases

    def _compute_attribute_diffs(
        self,
        query_claim: dict,
        match_claim: dict,
    ) -> list[AttributeDiff]:
        """
        Compute field-by-field differences.

        Args:
            query_claim: Query claim
            match_claim: Matched claim

        Returns:
            List of AttributeDiff
        """
        diffs = []

        # Fields to compare
        compare_fields = [
            "icd10_main",
            "icd10_secondary",
            "jgp_code",
            "procedures",
            "admission_mode",
            "department_code",
        ]

        for field in compare_fields:
            query_val = query_claim.get(field)
            match_val = match_claim.get(field)

            # Skip if both are None/empty
            if not query_val and not match_val:
                continue

            # Determine diff type
            if query_val and not match_val:
                diff_type = DiffType.EXTRA
            elif not query_val and match_val:
                diff_type = DiffType.MISSING
            elif query_val != match_val:
                diff_type = DiffType.DIFFERENT
            else:
                continue  # Same value, no diff

            diffs.append(AttributeDiff(
                field=field,
                query_value=query_val,
                match_value=match_val,
                diff_type=diff_type,
            ))

        return diffs

    def get_approved_case(self, case_id: str) -> dict | None:
        """Get approved case by ID."""
        return self._approved_cases.get(case_id)


# =============================================================================
# Indexer
# =============================================================================


class ClaimIndexer:
    """
    Index approved claims for similarity search.

    Builds both BM25 and vector indices.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
    ):
        """
        Initialize indexer.

        Args:
            bm25_index: BM25 index to populate
            vector_store: Vector store to populate
            embedding_service: Embedding generator
        """
        self.bm25 = bm25_index
        self.vectors = vector_store
        self.embeddings = embedding_service

    def index_claims(
        self,
        claims: list[dict[str, Any]],
        _show_progress: bool = True,  # Reserved for future progress bar
    ) -> dict[str, Any]:
        """
        Index batch of approved claims.

        Args:
            claims: List of claim records
            show_progress: Show progress bar

        Returns:
            Indexing statistics
        """
        import time
        start = time.time()

        errors = []
        indexed_bm25 = 0
        indexed_vector = 0

        # Prepare documents for BM25
        bm25_docs = []
        for claim in claims:
            case_id = claim.get("case_id")
            if not case_id:
                errors.append("Claim missing case_id")
                continue

            text = self.embeddings.claim_to_text(claim)
            bm25_docs.append((case_id, text))

        # Build BM25 index
        if bm25_docs:
            self.bm25.build(bm25_docs)
            indexed_bm25 = len(bm25_docs)

        # Build vector index
        try:
            # Create collection
            self.vectors.create_collection(vector_size=self.embeddings.dimension)

            # Generate embeddings
            texts = [doc[1] for doc in bm25_docs]
            embeddings = self.embeddings.embed_batch(texts)

            # Upsert to vector store
            items = [
                (bm25_docs[i][0], embeddings[i], claims[i])
                for i in range(len(bm25_docs))
            ]
            indexed_vector = self.vectors.upsert_batch(items)

        except Exception as e:
            errors.append(f"Vector indexing failed: {e}")
            logger.error("Vector indexing failed: %s", e)

        duration = time.time() - start

        stats = {
            "total_claims": len(claims),
            "indexed_bm25": indexed_bm25,
            "indexed_vector": indexed_vector,
            "duration_seconds": round(duration, 2),
            "errors": errors,
        }

        logger.info(
            "Indexed %d claims (BM25: %d, Vector: %d) in %.2fs",
            len(claims),
            indexed_bm25,
            indexed_vector,
            duration,
        )

        return stats


# =============================================================================
# Factory Functions
# =============================================================================


def create_similarity_service(
    rerank_enabled: bool = False,
    in_memory: bool = True,
) -> SimilarityService:
    """
    Create configured similarity service.

    Args:
        rerank_enabled: Whether to use cross-encoder reranking
        in_memory: Use in-memory vector store

    Returns:
        Configured SimilarityService
    """
    return SimilarityService(
        bm25_index=BM25Index(),
        vector_store=VectorStore(in_memory=in_memory),
        embedding_service=EmbeddingService(),
        reranker=get_reranker(rerank_enabled),
    )
