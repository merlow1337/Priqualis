"""
PII Masking for Priqualis.

Masks personally identifiable information (PII) in claim data.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols (Dependency Inversion)
# =============================================================================


class MaskingStrategy(Protocol):
    """Protocol for masking strategies."""

    def __call__(self, value: str) -> str:
        """Mask a single value."""
        ...


class DataFrameMasker(Protocol):
    """Protocol for DataFrame masking."""

    def mask_dataframe(self, df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
        """Mask PII in DataFrame, return (masked_df, count)."""
        ...


# =============================================================================
# Masking Strategies
# =============================================================================


def mask_pesel(value: str) -> str:
    """
    Mask PESEL with deterministic hash.

    Uses SHA-256 hash (first 8 chars) for consistent masking
    across batches, enabling joins on masked data.

    Args:
        value: Original PESEL (11 digits)

    Returns:
        Masked value in format PESEL_XXXXXXXX
    """
    # Guard clause
    if not value or len(value) != 11:
        return value

    hash_val = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"PESEL_{hash_val}"


def mask_name(value: str) -> str:
    """
    Mask patient name with deterministic hash.

    Args:
        value: Original name

    Returns:
        Masked value in format PAT_XXXXXXXX
    """
    if not value:
        return value

    hash_val = hashlib.sha256(value.encode()).hexdigest()[:8].upper()
    return f"PAT_{hash_val}"


def mask_address(value: str) -> str:
    """Mask address with placeholder."""
    if not value:
        return value
    return "[MASKED_ADDRESS]"


def mask_phone(value: str) -> str:
    """Mask phone number with placeholder."""
    if not value:
        return value
    return "[MASKED_PHONE]"


def mask_email(value: str) -> str:
    """Mask email address with placeholder."""
    if not value:
        return value
    return "[MASKED_EMAIL]"


# =============================================================================
# Regex Patterns (compiled at module load)
# =============================================================================


# PESEL: 11 consecutive digits
PESEL_PATTERN = re.compile(r"\b\d{11}\b")

# Polish phone numbers (various formats)
PHONE_PATTERN = re.compile(
    r"\b(?:\+48\s?)?(?:\d{3}[\s-]?\d{3}[\s-]?\d{3}|\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2})\b"
)

# Email pattern
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")


# Default protected fields (identifiers and codes that should never be masked)
DEFAULT_PROTECTED_FIELDS: frozenset[str] = frozenset({
    "case_id",
    "patient_id",
    "pesel_masked",
    "jgp_code",
    "department_code",
    "icd10_main",
    "icd10_secondary",
    "procedures",
})


# =============================================================================
# Field Configuration
# =============================================================================


@dataclass(slots=True, frozen=True)
class MaskingRule:
    """Immutable configuration for masking a specific field."""

    field_name: str
    mask_fn: Callable[[str], str]
    description: str = ""


def get_default_masking_rules() -> tuple[MaskingRule, ...]:
    """Get default masking rules. Immutable tuple for safety."""
    return (
        MaskingRule("pesel", mask_pesel, "PESEL national ID"),
        MaskingRule("pesel_raw", mask_pesel, "Raw PESEL"),
        MaskingRule("patient_name", mask_name, "Patient name"),
        MaskingRule("name", mask_name, "Name field"),
        MaskingRule("surname", mask_name, "Surname field"),
        MaskingRule("address", mask_address, "Address"),
        MaskingRule("phone", mask_phone, "Phone number"),
        MaskingRule("email", mask_email, "Email address"),
    )


@dataclass(slots=True)
class PIIMasker:
    """
    Masks PII fields in claim data.

    Supports both column-level masking (specific fields) and
    content-level masking (regex patterns in text fields).

    Implements DataFrameMasker protocol for DI compatibility.
    """

    field_rules: tuple[MaskingRule, ...] = field(default_factory=get_default_masking_rules)
    scan_content: bool = True
    protected_fields: frozenset[str] = DEFAULT_PROTECTED_FIELDS

    def mask_dataframe(self, df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
        """
        Mask PII in DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (masked DataFrame, count of masked fields)
        """
        masked_count = 0
        result = df.clone()

        # Build lowercase -> actual column name map
        col_name_map = {col.lower(): col for col in result.columns}

        # Apply field-level masking
        for rule in self.field_rules:
            actual_col = col_name_map.get(rule.field_name.lower())

            # Guard clauses
            if actual_col is None:
                continue
            if actual_col in self.protected_fields:
                continue

            result = result.with_columns(
                pl.col(actual_col)
                .map_elements(rule.mask_fn, return_dtype=pl.Utf8)
                .alias(actual_col)
            )
            masked_count += 1
            logger.debug("Masked field '%s' with %s", actual_col, rule.description)

        # Scan text content for PII patterns
        if self.scan_content:
            result = self._mask_text_columns(result)

        return result, masked_count

    def _mask_text_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply regex-based masking to text columns."""
        result = df

        for col in df.columns:
            # Guard clauses
            if col in self.protected_fields:
                continue
            if df[col].dtype != pl.Utf8:
                continue

            result = result.with_columns(
                pl.col(col)
                .map_elements(self._mask_content, return_dtype=pl.Utf8)
                .alias(col)
            )

        return result

    def _mask_content(self, text: str | None) -> str | None:
        """Mask PII patterns in text content using regex."""
        if not text:
            return text

        # Apply regex substitutions
        text = PESEL_PATTERN.sub("[MASKED_PESEL]", text)
        text = PHONE_PATTERN.sub("[MASKED_PHONE]", text)
        text = EMAIL_PATTERN.sub("[MASKED_EMAIL]", text)

        return text

    def mask_dict(self, record: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """
        Mask PII in a dictionary record.

        Args:
            record: Input record

        Returns:
            Tuple of (masked record, count of masked fields)
        """
        masked = record.copy()
        masked_count = 0

        for rule in self.field_rules:
            # Guard clauses
            if rule.field_name not in masked:
                continue
            if rule.field_name in self.protected_fields:
                continue

            value = masked[rule.field_name]
            if not value:
                continue

            masked[rule.field_name] = rule.mask_fn(str(value))
            masked_count += 1

        return masked, masked_count
