"""
Custom exceptions for Priqualis.
"""


class PriqualisError(Exception):
    """Base exception for all Priqualis errors."""

    pass


# =============================================================================
# ETL Exceptions
# =============================================================================


class ETLError(PriqualisError):
    """Base exception for ETL errors."""

    pass


class SchemaValidationError(ETLError):
    """Raised when input data fails schema validation."""

    pass


class ImportError(ETLError):
    """Raised when file import fails."""

    pass


class PIIMaskingError(ETLError):
    """Raised when PII masking fails."""

    pass


# =============================================================================
# Rule Engine Exceptions
# =============================================================================


class RuleEngineError(PriqualisError):
    """Base exception for rule engine errors."""

    pass


class RuleParseError(RuleEngineError):
    """Raised when YAML rule parsing fails."""

    pass


class RuleExecutionError(RuleEngineError):
    """Raised when rule execution fails."""

    pass


class InvalidRuleStateError(RuleEngineError):
    """Raised when rule returns invalid state (not SAT/VIOL/WARN)."""

    pass


# =============================================================================
# Search Exceptions
# =============================================================================


class SearchError(PriqualisError):
    """Base exception for search errors."""

    pass


class IndexNotFoundError(SearchError):
    """Raised when search index doesn't exist."""

    pass


class EmbeddingError(SearchError):
    """Raised when embedding generation fails."""

    pass


class VectorStoreError(SearchError):
    """Raised when vector store operation fails."""

    pass


# =============================================================================
# AutoFix Exceptions
# =============================================================================


class AutoFixError(PriqualisError):
    """Base exception for AutoFix errors."""

    pass


class PatchGenerationError(AutoFixError):
    """Raised when patch generation fails."""

    pass


class PatchApplicationError(AutoFixError):
    """Raised when patch application fails."""

    pass