"""
Rules module for Priqualis.

Provides rule-based validation for healthcare claim data.
"""

from priqualis.rules.engine import (
    RuleEngine,
    RuleExecutor,
    load_rules,
    safe_eval,
    validate_rule,
)
from priqualis.rules.models import (
    AutoFixOperation,
    RuleDefinition,
    RuleResult,
    RuleSeverity,
    RuleState,
    ValidationReport,
    ViolationAction,
)
from priqualis.rules.scoring import (
    DEFAULT_WEIGHTS,
    ImpactScorer,
    ScoringWeights,
)

__all__ = [
    # Engine
    "RuleEngine",
    "RuleExecutor",
    "load_rules",
    "validate_rule",
    "safe_eval",
    # Models
    "AutoFixOperation",
    "RuleDefinition",
    "RuleResult",
    "RuleSeverity",
    "RuleState",
    "ValidationReport",
    "ViolationAction",
    # Scoring
    "ImpactScorer",
    "ScoringWeights",
    "DEFAULT_WEIGHTS",
]