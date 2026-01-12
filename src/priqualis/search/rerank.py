"""
Reranker for Priqualis.

Optional cross-encoder reranking for top-N refinement.
"""

import logging
from typing import Any, Protocol

from priqualis.search.models import SearchResult

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class CrossEncoderModel(Protocol):
    """Protocol for cross-encoder models."""

    def predict(self, sentence_pairs: list[tuple[str, str]]) -> list[float]:
        """Predict similarity scores for sentence pairs."""
        ...


# =============================================================================
# Reranker
# =============================================================================


class Reranker:
    """
    Cross-encoder reranker for top-N refinement.

    Uses a cross-encoder model to score query-candidate pairs
    for more accurate ranking than bi-encoder embeddings.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        """
        Initialize reranker.

        Args:
            model_name: HuggingFace cross-encoder model
            device: "cpu" or "cuda"
        """
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy load cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Rerank candidates using cross-encoder scores.

        Args:
            query: Query text
            candidates: Search results to rerank
            top_k: Number of results to return

        Returns:
            Reranked SearchResult list
        """
        if not candidates:
            return []

        # Build query-candidate pairs
        pairs = []
        for c in candidates:
            # Convert claim data to text
            claim_text = self._claim_to_text(c.claim_data) if c.claim_data else f"Case {c.case_id}"
            pairs.append((query, claim_text))

        # Score pairs
        scores = self.model.predict(pairs)

        # Combine with candidates and sort
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Update ranks and return top-k
        results = []
        for rank, (candidate, score) in enumerate(scored_candidates[:top_k], start=1):
            # Create new result with updated score and rank
            results.append(SearchResult(
                case_id=candidate.case_id,
                score=float(score),
                source=candidate.source,
                rank=rank,
                claim_data=candidate.claim_data,
            ))

        logger.debug("Reranked %d candidates → top %d", len(candidates), len(results))
        return results

    def _claim_to_text(self, claim: dict[str, Any]) -> str:
        """Convert claim to text for reranking."""
        parts = []

        jgp = claim.get("jgp_code")
        if jgp:
            parts.append(f"JGP:{jgp}")

        icd_main = claim.get("icd10_main")
        if icd_main:
            parts.append(f"ICD10:{icd_main}")

        procedures = claim.get("procedures", [])
        if procedures:
            if isinstance(procedures, list):
                procedures = procedures[:3]
            parts.append(f"PROC:{procedures}")

        return " ".join(parts) if parts else "EMPTY"


# =============================================================================
# Factory
# =============================================================================


def get_reranker(enabled: bool = True) -> Reranker | None:
    """Get reranker if enabled."""
    if not enabled:
        return None
    return Reranker()
