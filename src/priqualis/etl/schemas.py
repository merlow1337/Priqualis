"""
ETL Schemas for Priqualis.

Pydantic models for claim data validation and serialization.
"""

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# Enums
# =============================================================================


class AdmissionMode(str, Enum):
    """Mode of patient admission."""

    EMERGENCY = "emergency"
    PLANNED = "planned"
    TRANSFER = "transfer"


class ClaimStatus(str, Enum):
    """Status of claim in validation pipeline."""

    NEW = "new"
    VALIDATED = "validated"
    FIXED = "fixed"
    SUBMITTED = "submitted"
    REJECTED = "rejected"


class Gender(str, Enum):
    """Patient gender."""

    MALE = "M"
    FEMALE = "F"


# =============================================================================
# Component Models
# =============================================================================


class PatientInfo(BaseModel):
    """Patient demographic information (PII masked)."""

    patient_id: str = Field(..., description="Masked patient identifier (PAT_XXXXXXXX)")
    pesel_masked: str = Field(..., description="Masked PESEL (PESEL_XXXXXXXX)")
    birth_date: date
    gender: Gender


class DiagnosisInfo(BaseModel):
    """ICD-10 diagnosis information."""

    icd10_main: str | None = Field(
        None,
        min_length=3,
        max_length=10,
        description="Main ICD-10 diagnosis code",
    )
    icd10_secondary: list[str] = Field(
        default_factory=list,
        description="Secondary/comorbidity ICD-10 codes",
    )

    @field_validator("icd10_main", mode="before")
    @classmethod
    def empty_string_to_none(cls, v: str | None) -> str | None:
        """Convert empty strings to None."""
        if v == "":
            return None
        return v


class ProcedureInfo(BaseModel):
    """Medical procedure information (ICD-9-CM codes)."""

    procedures: list[str] = Field(
        default_factory=list,
        description="List of procedure codes",
    )

    @field_validator("procedures", mode="before")
    @classmethod
    def split_comma_separated(cls, v: list[str] | str) -> list[str]:
        """Handle comma-separated string input from CSV."""
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v


class JGPInfo(BaseModel):
    """JGP (Diagnosis Related Group) classification."""

    jgp_code: str | None = Field(
        None,
        min_length=2,
        max_length=10,
        description="JGP group code (e.g., A01, B02)",
    )
    jgp_name: str | None = Field(None, description="JGP group name")
    tariff_value: float = Field(
        0.0,
        ge=0.0,
        description="Tariff value in PLN",
    )

    @field_validator("jgp_code", mode="before")
    @classmethod
    def empty_string_to_none(cls, v: str | None) -> str | None:
        """Convert empty strings to None."""
        if v == "":
            return None
        return v


# =============================================================================
# Main Claim Record
# =============================================================================


class ClaimRecord(BaseModel):
    """
    Single hospitalization claim record.

    Represents one billing episode submitted to NFZ for reimbursement.
    """

    # Identifiers
    case_id: str = Field(..., description="Unique case/encounter identifier")
    patient_id: str = Field(..., description="Masked patient identifier")
    pesel_masked: str = Field(..., description="Masked PESEL")

    # Demographics
    birth_date: date
    gender: Gender

    # Hospitalization
    admission_date: date
    discharge_date: date
    admission_mode: AdmissionMode | str = Field(
        ...,
        description="Mode of admission (emergency/planned/transfer)",
    )
    department_code: str | None = Field(
        None,
        description="NFZ department code",
    )

    # Clinical
    jgp_code: str | None = Field(None, description="JGP group code")
    jgp_name: str | None = Field(None, description="JGP group name")
    tariff_value: float = Field(0.0, ge=0.0, description="Tariff value in PLN")
    icd10_main: str | None = Field(None, description="Main ICD-10 diagnosis")
    icd10_secondary: list[str] = Field(
        default_factory=list,
        description="Secondary diagnoses",
    )
    procedures: list[str] = Field(
        default_factory=list,
        description="Procedure codes",
    )

    # Status
    status: ClaimStatus = ClaimStatus.NEW
    has_error: bool = False
    error_type: str | None = None

    @field_validator("icd10_secondary", "procedures", mode="before")
    @classmethod
    def split_comma_separated(cls, v: list[str] | str | None) -> list[str]:
        """Handle comma-separated string input from CSV/Parquet."""
        if v is None:
            return []
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return list(v)

    @field_validator("admission_mode", mode="before")
    @classmethod
    def validate_admission_mode(cls, v: str) -> AdmissionMode | str:
        """Allow invalid admission modes for error detection."""
        try:
            return AdmissionMode(v)
        except ValueError:
            # Return as string for validation rule to catch
            return v
    
    @field_validator("department_code", mode="before")
    @classmethod
    def coerce_to_string(cls, v: str | int | float | None) -> str | None:
        """Coerce numeric codes (like 5200) to string."""
        if v is None:
            return None
        if isinstance(v, float):
            return str(int(v))
        return str(v)

    @property
    def length_of_stay(self) -> int:
        """Calculate length of stay in days."""
        return (self.discharge_date - self.admission_date).days

    @property
    def is_valid_date_range(self) -> bool:
        """Check if discharge is after admission."""
        return self.discharge_date >= self.admission_date

    model_config = {
        "use_enum_values": True,
        "validate_assignment": True,
    }


# =============================================================================
# Batch Models
# =============================================================================


class ClaimBatch(BaseModel):
    """Batch of claim records for processing."""

    records: list[ClaimRecord] = Field(
        default_factory=list,
        description="List of claim records",
    )
    source_file: str | None = Field(None, description="Source file path")
    created_at: str | None = Field(None, description="ISO timestamp of batch creation")

    @property
    def count(self) -> int:
        """Number of records in batch."""
        return len(self.records)

    @property
    def error_count(self) -> int:
        """Number of records with errors."""
        return sum(1 for r in self.records if r.has_error)

    @property
    def valid_count(self) -> int:
        """Number of valid records."""
        return self.count - self.error_count

    def get_by_case_id(self, case_id: str) -> ClaimRecord | None:
        """Find record by case ID."""
        for record in self.records:
            if record.case_id == case_id:
                return record
        return None

    def filter_by_status(self, status: ClaimStatus) -> list[ClaimRecord]:
        """Filter records by status."""
        return [r for r in self.records if r.status == status]

    def filter_by_jgp(self, jgp_code: str) -> list[ClaimRecord]:
        """Filter records by JGP code."""
        return [r for r in self.records if r.jgp_code == jgp_code]


# =============================================================================
# Processed Output
# =============================================================================


class ProcessedBatch(BaseModel):
    """Result of ETL processing."""

    batch: ClaimBatch
    output_path: str | None = None
    processing_time_ms: float = 0.0
    pii_fields_masked: int = 0
    schema_errors: list[str] = Field(default_factory=list)
