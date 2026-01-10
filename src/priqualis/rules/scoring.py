"""
Rule Scoring for Priqualis.

Calculates impact scores for violations to prioritize fixes.
"""

import logging
from dataclasses import dataclass
from typing import Protocol

from priqualis.rules.models import RuleResult, RuleSeverity

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class ImpactCalculator(Protocol):
    """Protocol for impact score calculation."""

    def calculate(self, result: RuleResult, record: dict) -> float:
        """Calculate impact score for a violation."""
        ...


# =============================================================================
# Scoring Configuration
# =============================================================================


@dataclass(slots=True, frozen=True)
class ScoringWeights:
    """
    Immutable weights for impact score calculation.

    Formula: impact = rejection_risk × tariff × fix_cost_factor
    """

    # Base rejection risk by severity
    error_rejection_risk: float = 0.9
    warning_rejection_risk: float = 0.3

    # Fix cost multipliers by autofix availability
    autofix_available_cost: float = 0.2
    manual_fix_cost: float = 1.0

    # Tariff normalization (PLN)
    tariff_baseline: float = 5000.0


DEFAULT_WEIGHTS = ScoringWeights()


# =============================================================================
# Impact Scorer
# =============================================================================


class ImpactScorer:
    """
    Calculates impact scores for rule violations.

    Impact score formula:
        impact = rejection_risk × (tariff / baseline) × fix_cost

    Higher scores indicate violations that should be fixed first.
    """

    def __init__(self, weights: ScoringWeights | None = None):
        """
        Initialize scorer with weights.

        Args:
            weights: Scoring weights (uses defaults if None)
        """
        self.weights = weights or DEFAULT_WEIGHTS

    def calculate(self, result: RuleResult, record: dict) -> float:
        """
        Calculate impact score for a violation.

        Args:
            result: Rule execution result
            record: The claim record (dict or Pydantic model)

        Returns:
            Impact score (higher = more important to fix)
        """
        # Guard: only calculate for violations/warnings
        if result.is_satisfied:
            return 0.0

        # Get record data (handle both dict and Pydantic model)
        if hasattr(record, "model_dump"):
            record_data = record.model_dump()
        else:
            record_data = record

        # Component 1: Rejection risk based on severity
        rejection_risk = self._get_rejection_risk(result)

        # Component 2: Tariff value (normalized)
        tariff = record_data.get("tariff_value", self.weights.tariff_baseline)
        tariff_factor = tariff / self.weights.tariff_baseline

        # Component 3: Fix cost (lower if autofix available)
        fix_cost = self._get_fix_cost(result)

        # Calculate final score
        impact = rejection_risk * tariff_factor * fix_cost

        logger.debug(
            "Impact for %s/%s: risk=%.2f, tariff=%.2f, fix=%.2f → %.2f",
            result.rule_id,
            result.case_id,
            rejection_risk,
            tariff_factor,
            fix_cost,
            impact,
        )

        return round(impact, 4)

    def _get_rejection_risk(self, result: RuleResult) -> float:
        """Get rejection risk based on state."""
        # state is a string due to model_config use_enum_values=True
        state = result.state if isinstance(result.state, str) else result.state.value
        if state == "VIOL":
            return self.weights.error_rejection_risk
        elif state == "WARN":
            return self.weights.warning_rejection_risk
        return 0.0

    def _get_fix_cost(self, result: RuleResult) -> float:
        """Get fix cost factor based on autofix availability."""
        if result.autofix_hint is not None:
            return self.weights.autofix_available_cost
        return self.weights.manual_fix_cost

    def score_batch(
        self,
        results: list[RuleResult],
        records: dict[str, dict],
    ) -> list[tuple[RuleResult, float]]:
        """
        Score a batch of results.

        Args:
            results: List of rule results
            records: Dict mapping case_id to record

        Returns:
            List of (result, score) tuples sorted by score descending
        """
        scored = []
        for result in results:
            record = records.get(result.case_id, {})
            score = self.calculate(result, record)
            scored.append((result, score))

        # Sort by score descending (highest priority first)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
