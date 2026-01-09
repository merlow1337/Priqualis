"""
Pytest configuration and shared fixtures.
"""

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_path() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent.parent / "data" / "fixtures"


@pytest.fixture
def sample_claim() -> dict:
    """Sample claim record for testing."""
    return {
        "case_id": "ENC12345",
        "patient_id": "PAT_MASKED_001",
        "admission_date": "2024-01-15",
        "discharge_date": "2024-01-18",
        "jgp_code": "A01",
        "icd10_main": "J18.9",
        "icd10_secondary": ["E11.9", "I10"],
        "procedures": ["88.761", "99.04"],
        "mode": "emergency",
        "department_code": "4000",
        "tariff_value": 3250.0,
    }


@pytest.fixture
def sample_rule_yaml() -> str:
    """Sample YAML rule definition."""
    return """
rule_id: R001
name: Required main diagnosis
description: Main ICD-10 diagnosis must be present
severity: error
condition: |
    record.get('icd10_main') is not None 
    and len(record.get('icd10_main', '')) >= 3
on_violation:
    message: "Missing or invalid main diagnosis (ICD-10)"
    autofix_hint: "add_if_absent"
"""


@pytest.fixture
def sample_batch(sample_claim) -> list[dict]:
    """Sample batch of claims for testing."""
    return [
        sample_claim,
        {**sample_claim, "case_id": "ENC12346", "icd10_main": None},  # Invalid
        {**sample_claim, "case_id": "ENC12347", "jgp_code": "B02"},
    ]