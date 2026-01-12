"""
Rule Models for Priqualis.

Pydantic models for rule definitions and execution results.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class RuleSeverity(str, Enum):
    """Severity level of a rule violation."""

    ERROR = "error"
    WARNING = "warning"


class RuleState(str, Enum):
    """Outcome state of rule execution."""

    SAT = "SAT"      # Satisfied - rule passed
    VIOL = "VIOL"    # Violated - rule failed
    WARN = "WARN"    # Warning - soft failure


class AutoFixOperation(str, Enum):
    """Types of autofix operations."""

    ADD_IF_ABSENT = "add_if_absent"
    SET = "set"
    REMOVE = "remove"
    REPLACE = "replace"


# =============================================================================
# Violation Action
# =============================================================================


class ViolationAction(BaseModel):
    """Action to take when a rule is violated."""

    message: str = Field(..., description="User-facing error message")
    autofix_hint: AutoFixOperation | None = Field(
        None, description="Suggested autofix operation"
    )
    suggested_value: str | None = Field(
        None, description="Suggested value for autofix"
    )


# =============================================================================
# Rule Definition
# =============================================================================


class RuleDefinition(BaseModel):
    """
    Definition of a validation rule loaded from YAML.

    Rules are expressed as Python boolean expressions that are
    evaluated against claim records.
    """

    rule_id: str = Field(
        ...,
        pattern=r"^R\d{3}$",
        description="Rule identifier (e.g., R001, R002)",
    )
    name: str = Field(..., min_length=3, description="Human-readable rule name")
    description: str = Field(..., description="What this rule checks")
    severity: RuleSeverity = Field(
        RuleSeverity.ERROR,
        description="Severity level (error or warning)",
    )
    condition: str = Field(
        ...,
        min_length=1,
        description="Python boolean expression to evaluate",
    )
    on_violation: ViolationAction = Field(
        ..., description="Action when rule is violated"
    )
    jgp_groups: list[str] | None = Field(
        None, description="JGP groups this rule applies to (None = all)"
    )
    enabled: bool = Field(True, description="Whether rule is active")
    version: str = Field("1.0", description="Rule version for tracking changes")

    @field_validator("condition")
    @classmethod
    def validate_condition_syntax(cls, v: str) -> str:
        """Validate that condition is valid Python syntax."""
        try:
            compile(v, "<rule>", "eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid condition syntax: {e}") from e
        return v

    @property
    def autofix_hint(self) -> AutoFixOperation | None:
        """Convenience accessor for autofix hint."""
        return self.on_violation.autofix_hint

    model_config = {"use_enum_values": True}


# =============================================================================
# Rule Execution Result
# =============================================================================


class RuleResult(BaseModel):
    """Result of executing a single rule on a single record."""

    rule_id: str = Field(..., description="ID of the executed rule")
    case_id: str = Field(..., description="ID of the evaluated record")
    state: RuleState = Field(..., description="Outcome: SAT, VIOL, or WARN")
    message: str | None = Field(None, description="Error message if violated")
    impact_score: float | None = Field(
        None,
        ge=0.0,
        description="Calculated impact score for prioritization",
    )
    autofix_hint: AutoFixOperation | None = Field(
        None, description="Suggested autofix operation"
    )
    executed_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of execution",
    )

    @property
    def is_violation(self) -> bool:
        """Check if result is a violation."""
        return self.state == RuleState.VIOL

    @property
    def is_warning(self) -> bool:
        """Check if result is a warning."""
        return self.state == RuleState.WARN

    @property
    def is_satisfied(self) -> bool:
        """Check if rule was satisfied."""
        return self.state == RuleState.SAT

    model_config = {"use_enum_values": True}


# =============================================================================
# Validation Report
# =============================================================================


class ValidationReport(BaseModel):
    """Aggregated results of validating a batch of claims."""

    source_file: str | None = Field(None, description="Source file validated")
    total_records: int = Field(0, ge=0, description="Total records processed")
    total_rules: int = Field(0, ge=0, description="Total rules executed")
    results: list[RuleResult] = Field(
        default_factory=list, description="All rule results"
    )
    executed_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of validation",
    )

    @property
    def violations(self) -> list[RuleResult]:
        """Get all violations."""
        return [r for r in self.results if r.state == RuleState.VIOL]

    @property
    def warnings(self) -> list[RuleResult]:
        """Get all warnings."""
        return [r for r in self.results if r.state == RuleState.WARN]

    @property
    def satisfied(self) -> list[RuleResult]:
        """Get all satisfied results."""
        return [r for r in self.results if r.state == RuleState.SAT]

    @property
    def violation_count(self) -> int:
        """Count of violations."""
        return len(self.violations)

    @property
    def warning_count(self) -> int:
        """Count of warnings."""
        return len(self.warnings)

    @property
    def pass_rate(self) -> float:
        """Percentage of rule checks that passed (0.0 to 1.0)."""
        if not self.results:
            return 1.0
        return len(self.satisfied) / len(self.results)

    def get_violations_by_rule(self, rule_id: str) -> list[RuleResult]:
        """Get all violations for a specific rule."""
        return [r for r in self.violations if r.rule_id == rule_id]

    def get_violations_for_case(self, case_id: str) -> list[RuleResult]:
        """Get all violations for a specific case."""
        return [r for r in self.violations if r.case_id == case_id]

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        return {
            "total_records": self.total_records,
            "total_rules": self.total_rules,
            "total_checks": len(self.results),
            "violations": self.violation_count,
            "warnings": self.warning_count,
            "pass_rate": f"{self.pass_rate:.1%}",
        }
