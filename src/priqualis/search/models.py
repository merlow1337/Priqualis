"""
Search Models for Priqualis.

Data models for hybrid search (BM25 + Vector).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class SearchSource(str, Enum):
    """Source of search result."""

    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"


class DiffType(str, Enum):
    """Type of attribute difference."""

    MISSING = "missing"
    DIFFERENT = "different"
    EXTRA = "extra"


class CaseStatus(str, Enum):
    """Status of a case in the index."""

    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"


# =============================================================================
# Search Query
# =============================================================================


@dataclass(slots=True)
class SearchQuery:
    """Query for similarity search."""

    case_id: str
    text: str  # Concatenated searchable fields
    jgp_code: str | None = None
    icd10_codes: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_claim(cls, claim: dict[str, Any]) -> "SearchQuery":
        """Create query from claim record."""
        # Build searchable text
        parts = []

        jgp = claim.get("jgp_code")
        if jgp:
            parts.append(f"JGP:{jgp}")

        icd_main = claim.get("icd10_main")
        if icd_main:
            parts.append(f"ICD10:{icd_main}")

        icd_secondary = claim.get("icd10_secondary", [])
        if icd_secondary:
            if isinstance(icd_secondary, str):
                icd_secondary = [s.strip() for s in icd_secondary.split(",") if s.strip()]
            parts.extend([f"ICD10:{code}" for code in icd_secondary])

        procedures = claim.get("procedures", [])
        if procedures:
            if isinstance(procedures, str):
                procedures = [p.strip() for p in procedures.split(",") if p.strip()]
            parts.extend([f"PROC:{proc}" for proc in procedures])

        dept = claim.get("department_code")
        if dept:
            parts.append(f"DEPT:{dept}")

        return cls(
            case_id=claim.get("case_id", "unknown"),
            text=" ".join(parts),
            jgp_code=jgp,
            icd10_codes=[icd_main] + icd_secondary if icd_main else icd_secondary,
            procedures=procedures if isinstance(procedures, list) else [],
        )


# =============================================================================
# Search Results
# =============================================================================


@dataclass(slots=True)
class SearchResult:
    """Single search result."""

    case_id: str
    score: float
    source: SearchSource
    rank: int
    claim_data: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "SearchResult") -> bool:
        """For sorting by score (descending)."""
        return self.score > other.score


@dataclass(slots=True)
class AttributeDiff:
    """Difference between query and match for a single field."""

    field: str
    query_value: Any
    match_value: Any
    diff_type: DiffType


@dataclass(slots=True)
class SimilarCase:
    """Similar case with attribute differences."""

    case_id: str
    similarity_score: float
    attribute_diffs: list[AttributeDiff] = field(default_factory=list)
    jgp_code: str | None = None
    status: CaseStatus = CaseStatus.APPROVED
    claim_data: dict[str, Any] = field(default_factory=dict)

    @property
    def diff_count(self) -> int:
        """Number of attribute differences."""
        return len(self.attribute_diffs)


# =============================================================================
# Indexing Stats
# =============================================================================


@dataclass(slots=True)
class IndexingStats:
    """Statistics from indexing operation."""

    total_claims: int = 0
    indexed_bm25: int = 0
    indexed_vector: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Percentage of successfully indexed claims."""
        if self.total_claims == 0:
            return 1.0
        return (self.indexed_bm25 + self.indexed_vector) / (2 * self.total_claims)
