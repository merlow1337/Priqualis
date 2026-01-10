"""
Patch Generator for Priqualis.

Generates patches (autofix suggestions) for rule violations.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from priqualis.rules.models import AutoFixOperation, RuleResult

logger = logging.getLogger(__name__)


# =============================================================================
# Patch Models
# =============================================================================


class PatchOperation(BaseModel):
    """Single field modification operation."""

    op: AutoFixOperation = Field(
        ..., description="Type of operation to perform"
    )
    field: str = Field(..., description="Field name to modify")
    value: Any = Field(None, description="New value to set")
    old_value: Any | None = Field(None, description="Previous value (for audit)")

    model_config = {"use_enum_values": True}


class Patch(BaseModel):
    """
    Complete patch for a single claim record.

    Contains one or more operations to fix rule violations.
    """

    case_id: str = Field(..., description="ID of the case to patch")
    rule_id: str = Field(..., description="Rule that triggered this patch")
    changes: list[PatchOperation] = Field(
        default_factory=list,
        min_length=1,
        description="List of field changes",
    )
    rationale: str = Field(
        ..., description="Human-readable explanation of the fix"
    )
    risk_note: str = Field(
        default="Formal verification remains with provider.",
        description="Risk disclaimer",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the fix (0.0 to 1.0)",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When patch was generated",
    )
    applied: bool = Field(False, description="Whether patch has been applied")

    @property
    def is_safe(self) -> bool:
        """Check if patch is safe to auto-apply (confidence > 0.8)."""
        return self.confidence >= 0.8

    def to_yaml_dict(self) -> dict:
        """Export as YAML-friendly dict."""
        return {
            "case_id": self.case_id,
            "rule_id": self.rule_id,
            "changes": [
                {
                    "op": c.op,
                    "field": c.field,
                    "value": c.value,
                    "old_value": c.old_value,
                }
                for c in self.changes
            ],
            "rationale": self.rationale,
            "risk_note": self.risk_note,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }


class AuditEntry(BaseModel):
    """Audit trail entry for applied patches."""

    patch_id: str = Field(..., description="Unique patch identifier")
    case_id: str = Field(..., description="Case that was patched")
    rule_id: str = Field(..., description="Rule that triggered the patch")
    user: str = Field(..., description="User who applied the patch")
    changes: list[PatchOperation] = Field(default_factory=list)
    applied_at: datetime = Field(default_factory=datetime.now)
    approved: bool = Field(default=False)


# =============================================================================
# Protocols
# =============================================================================


class FixStrategy(Protocol):
    """Protocol for fix generation strategies."""

    def generate(
        self,
        violation: RuleResult,
        record: dict,
    ) -> list[PatchOperation]:
        """Generate patch operations for a violation."""
        ...


# =============================================================================
# Default Values per Field
# =============================================================================


# Default values for add_if_absent operations
DEFAULT_VALUES: dict[str, Any] = {
    # ICD-10 codes
    "icd10_main": "Z00.0",  # General examination
    "icd10_secondary": [],

    # JGP
    "jgp_code": "A01",  # Generic group
    "jgp_name": "General Medicine",
    "tariff_value": 0.0,

    # Administrative
    "admission_mode": "planned",
    "department_code": "4000",  # Internal medicine

    # Procedures
    "procedures": [],
}

# Suggested values for common violations
SUGGESTED_FIXES: dict[str, dict[str, Any]] = {
    "R001": {  # Missing main diagnosis
        "field": "icd10_main",
        "value": "Z00.0",
        "rationale": "Added placeholder diagnosis code. Requires clinical review.",
    },
    "R002": {  # Invalid date range
        "field": "discharge_date",
        "value": None,  # Will be computed from admission_date
        "rationale": "Set discharge date to admission date (same-day visit).",
    },
    "R003": {  # Missing JGP
        "field": "jgp_code",
        "value": "A01",
        "rationale": "Added default JGP code. Requires classification review.",
    },
    "R005": {  # Invalid admission mode
        "field": "admission_mode",
        "value": "planned",
        "rationale": "Set to 'planned' as default admission mode.",
    },
    "R006": {  # Missing department code
        "field": "department_code",
        "value": "4000",
        "rationale": "Added internal medicine department code.",
    },
}


# =============================================================================
# Patch Generator
# =============================================================================


class PatchGenerator:
    """
    Generates patches for rule violations.

    Strategies:
    - add_if_absent: Add missing field with default/inferred value
    - set: Override field value with suggested fix
    - remove: Remove invalid field value
    - replace: Replace value based on similar approved case
    """

    def __init__(
        self,
        default_values: dict[str, Any] | None = None,
        suggested_fixes: dict[str, dict[str, Any]] | None = None,
    ):
        """
        Initialize generator.

        Args:
            default_values: Default values for fields
            suggested_fixes: Rule-specific fix suggestions
        """
        self.default_values = default_values or DEFAULT_VALUES
        self.suggested_fixes = suggested_fixes or SUGGESTED_FIXES

    def generate(
        self,
        violation: RuleResult,
        record: dict,
    ) -> Patch | None:
        """
        Generate patch for a violation.

        Args:
            violation: Rule execution result (must be VIOL or WARN)
            record: Original claim record

        Returns:
            Patch with fix operations, or None if no fix available
        """
        # Guard: only generate for violations
        if violation.is_satisfied:
            return None

        # Guard: no autofix hint
        if violation.autofix_hint is None:
            logger.debug(
                "No autofix hint for %s on %s",
                violation.rule_id,
                violation.case_id,
            )
            return None

        # Generate operations based on hint
        operations = self._generate_operations(violation, record)

        # Guard: no operations generated
        if not operations:
            logger.warning(
                "Could not generate operations for %s on %s",
                violation.rule_id,
                violation.case_id,
            )
            return None

        # Build patch
        rationale = self._get_rationale(violation)
        confidence = self._estimate_confidence(violation, operations)

        return Patch(
            case_id=violation.case_id,
            rule_id=violation.rule_id,
            changes=operations,
            rationale=rationale,
            confidence=confidence,
        )

    def _generate_operations(
        self,
        violation: RuleResult,
        record: dict,
    ) -> list[PatchOperation]:
        """Generate operations based on autofix hint."""
        hint = violation.autofix_hint
        operations: list[PatchOperation] = []

        # Get suggested fix for this rule
        fix_config = self.suggested_fixes.get(violation.rule_id, {})
        field = fix_config.get("field")
        suggested_value = fix_config.get("value")

        # Guard: no field configured
        if not field:
            return []

        old_value = record.get(field)

        if hint == "add_if_absent":
            # Only add if field is missing/empty
            if not old_value:
                value = suggested_value or self.default_values.get(field)
                operations.append(PatchOperation(
                    op=AutoFixOperation.ADD_IF_ABSENT,
                    field=field,
                    value=value,
                    old_value=old_value,
                ))

        elif hint == "set":
            # Handle special case for date fix
            if field == "discharge_date" and suggested_value is None:
                # Set discharge = admission for date range errors
                suggested_value = record.get("admission_date")

            if suggested_value is not None:
                operations.append(PatchOperation(
                    op=AutoFixOperation.SET,
                    field=field,
                    value=suggested_value,
                    old_value=old_value,
                ))

        elif hint == "remove":
            if old_value is not None:
                operations.append(PatchOperation(
                    op=AutoFixOperation.REMOVE,
                    field=field,
                    value=None,
                    old_value=old_value,
                ))

        elif hint == "replace":
            # Replace requires lookup from similar cases
            # For now, fall back to set
            if suggested_value is not None:
                operations.append(PatchOperation(
                    op=AutoFixOperation.REPLACE,
                    field=field,
                    value=suggested_value,
                    old_value=old_value,
                ))

        return operations

    def _get_rationale(self, violation: RuleResult) -> str:
        """Get rationale for the fix."""
        fix_config = self.suggested_fixes.get(violation.rule_id, {})
        return fix_config.get(
            "rationale",
            f"Auto-fix for {violation.rule_id}: {violation.message or 'N/A'}",
        )

    def _estimate_confidence(
        self,
        violation: RuleResult,
        operations: list[PatchOperation],
    ) -> float:
        """
        Estimate confidence in the fix.

        Higher confidence for:
        - Simple fixes (single operation)
        - add_if_absent (less risky)
        - Known rules with tested fixes
        """
        base_confidence = 0.5

        # Single operation = higher confidence
        if len(operations) == 1:
            base_confidence += 0.1

        # add_if_absent is safer than set/replace
        if operations and operations[0].op == AutoFixOperation.ADD_IF_ABSENT:
            base_confidence += 0.2
        elif operations and operations[0].op == AutoFixOperation.SET:
            base_confidence += 0.1

        # Known rule with configured fix
        if violation.rule_id in self.suggested_fixes:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def generate_batch(
        self,
        violations: list[RuleResult],
        records: dict[str, dict],
    ) -> list[Patch]:
        """
        Generate patches for multiple violations.

        Args:
            violations: List of rule violations
            records: Mapping of case_id to record

        Returns:
            List of generated patches
        """
        patches: list[Patch] = []

        for violation in violations:
            record = records.get(violation.case_id, {})
            patch = self.generate(violation, record)
            if patch:
                patches.append(patch)

        logger.info(
            "Generated %d patches from %d violations",
            len(patches),
            len(violations),
        )

        return patches
