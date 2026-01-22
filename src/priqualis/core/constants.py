"""
Domain constants for Priqualis.

These are business-logic constants that should rarely change at runtime.
For environment-configurable values, use config.py instead.

Note: These constants are moved from various modules to centralize
domain knowledge and avoid magic values scattered throughout the codebase.
"""

from typing import Any


# =============================================================================
# NFZ Error Code Mapping
# =============================================================================


# Maps NFZ error codes to Priqualis rule IDs
NFZ_ERROR_MAPPING: dict[str, str] = {
    "CWV_001": "R001",  # Missing main diagnosis
    "CWV_002": "R002",  # Invalid date range
    "CWV_003": "R003",  # Missing JGP code
    "CWV_010": "R004",  # Procedure mismatch
    "CWV_015": "R005",  # Invalid admission mode
    "CWV_020": "R006",  # Missing department code
    "CWV_025": "R007",  # Invalid tariff
}


# =============================================================================
# Medical Code Defaults (Placeholders)
# =============================================================================


# Default ICD-10 code for missing diagnosis
DEFAULT_ICD10_PLACEHOLDER: str = "Z00.0"  # General examination

# Default JGP (Jednorodne Grupy Pacjentów) code
DEFAULT_JGP_CODE: str = "A01"  # Generic group

# Default department code (Internal medicine)
DEFAULT_DEPARTMENT_CODE: str = "4000"


# =============================================================================
# AutoFix Default Values
# =============================================================================


# Default values for add_if_absent operations
AUTOFIX_DEFAULT_VALUES: dict[str, Any] = {
    # ICD-10 codes
    "icd10_main": DEFAULT_ICD10_PLACEHOLDER,
    "icd10_secondary": [],
    # JGP
    "jgp_code": DEFAULT_JGP_CODE,
    "jgp_name": "General Medicine",
    "tariff_value": 0.0,
    # Administrative
    "admission_mode": "planned",
    "department_code": DEFAULT_DEPARTMENT_CODE,
    # Procedures
    "procedures": [],
}


# Suggested values for common rule violations
AUTOFIX_SUGGESTED_FIXES: dict[str, dict[str, Any]] = {
    "R001": {  # Missing main diagnosis
        "field": "icd10_main",
        "value": DEFAULT_ICD10_PLACEHOLDER,
        "rationale": "Added placeholder diagnosis code. Requires clinical review.",
    },
    "R002": {  # Invalid date range
        "field": "discharge_date",
        "value": None,  # Will be computed from admission_date
        "rationale": "Set discharge date to admission date (same-day visit).",
    },
    "R003": {  # Missing JGP
        "field": "jgp_code",
        "value": DEFAULT_JGP_CODE,
        "rationale": "Added default JGP code. Requires classification review.",
    },
    "R005": {  # Invalid admission mode
        "field": "admission_mode",
        "value": "planned",
        "rationale": "Set to 'planned' as default admission mode.",
    },
    "R006": {  # Missing department code
        "field": "department_code",
        "value": DEFAULT_DEPARTMENT_CODE,
        "rationale": "Added internal medicine department code.",
    },
}


# =============================================================================
# Date Parsing Formats
# =============================================================================


DATE_PARSE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
)
