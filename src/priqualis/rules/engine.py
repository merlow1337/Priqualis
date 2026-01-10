"""
Rule Engine for Priqualis.

Parses, executes, and orchestrates validation rules.
"""

import ast
import logging
from pathlib import Path
from typing import Any, Protocol

import yaml

from priqualis.core.exceptions import RuleEngineError, RuleExecutionError, RuleParseError
from priqualis.etl.schemas import ClaimBatch, ClaimRecord
from priqualis.rules.models import (
    AutoFixOperation,
    RuleDefinition,
    RuleResult,
    RuleSeverity,
    RuleState,
    ValidationReport,
    ViolationAction,
)
from priqualis.rules.scoring import ImpactScorer

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class RuleLoader(Protocol):
    """Protocol for rule loading."""

    def load(self, rules_path: Path) -> list[RuleDefinition]:
        """Load rules from path."""
        ...


class RuleEvaluator(Protocol):
    """Protocol for rule evaluation."""

    def execute(self, rule: RuleDefinition, record: dict) -> RuleResult:
        """Execute a rule on a record."""
        ...


# =============================================================================
# Safe Expression Evaluator
# =============================================================================


# Allowed names in rule conditions (restricted for security)
SAFE_BUILTINS: dict[str, Any] = {
    # Type checks
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    # Comparisons
    "min": min,
    "max": max,
    "abs": abs,
    "sum": sum,
    "all": all,
    "any": any,
    # Membership
    "in": lambda x, y: x in y,
    "isinstance": isinstance,
    # None check
    "None": None,
    "True": True,
    "False": False,
}

# Disallowed AST nodes for security
DISALLOWED_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Call,  # Will be selectively allowed
    ast.Attribute,  # Will be selectively allowed
)


def validate_expression(expr: str) -> bool:
    """
    Validate that expression is safe to evaluate.

    Args:
        expr: Python expression string

    Returns:
        True if safe

    Raises:
        RuleParseError: If expression is unsafe
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise RuleParseError(f"Invalid expression syntax: {e}") from e

    # Check for dangerous operations
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise RuleParseError("Import statements not allowed in conditions")

    return True


def safe_eval(expr: str, context: dict[str, Any]) -> Any:
    """
    Safely evaluate a Python expression with restricted context.

    Args:
        expr: Python expression
        context: Variable context (record fields)

    Returns:
        Evaluation result

    Raises:
        RuleExecutionError: If evaluation fails
    """
    # Build evaluation context
    eval_globals = {"__builtins__": SAFE_BUILTINS}
    eval_locals = context.copy()

    try:
        return eval(expr, eval_globals, eval_locals)
    except Exception as e:
        raise RuleExecutionError(
            f"Failed to evaluate expression '{expr}': {e}"
        ) from e


# =============================================================================
# Rule Parser
# =============================================================================


def load_rules(rules_path: Path) -> list[RuleDefinition]:
    """
    Load all YAML rules from directory or file.

    Args:
        rules_path: Path to rules directory or single YAML file

    Returns:
        List of RuleDefinition objects

    Raises:
        RuleParseError: If loading or parsing fails
    """
    rules: list[RuleDefinition] = []

    # Handle single file or directory
    if rules_path.is_file():
        yaml_files = [rules_path]
    elif rules_path.is_dir():
        yaml_files = list(rules_path.glob("*.yaml")) + list(rules_path.glob("*.yml"))
    else:
        raise RuleParseError(f"Rules path not found: {rules_path}")

    # Guard: no rules found
    if not yaml_files:
        logger.warning("No YAML rule files found in %s", rules_path)
        return rules

    for yaml_file in sorted(yaml_files):
        file_rules = _load_rules_from_file(yaml_file)
        rules.extend(file_rules)

    logger.info("Loaded %d rules from %s", len(rules), rules_path)
    return rules


def _load_rules_from_file(file_path: Path) -> list[RuleDefinition]:
    """Load rules from a single YAML file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
    except Exception as e:
        raise RuleParseError(f"Failed to load {file_path}: {e}") from e

    # Guard: empty file
    if not data:
        logger.debug("Empty rule file: %s", file_path)
        return []

    # Handle single rule or list of rules
    if isinstance(data, dict):
        # Single rule in file
        if "rule_id" in data:
            return [_parse_rule(data, file_path)]
        # Rules under 'rules' key
        elif "rules" in data:
            return [_parse_rule(r, file_path) for r in data["rules"]]
        else:
            raise RuleParseError(f"Invalid rule file format: {file_path}")
    elif isinstance(data, list):
        return [_parse_rule(r, file_path) for r in data]
    else:
        raise RuleParseError(f"Unexpected format in {file_path}")


def _parse_rule(data: dict, source_file: Path) -> RuleDefinition:
    """Parse a single rule from dict."""
    try:
        # Build ViolationAction
        on_violation_data = data.get("on_violation", {})
        if isinstance(on_violation_data, str):
            on_violation = ViolationAction(message=on_violation_data)
        else:
            on_violation = ViolationAction(**on_violation_data)

        # Parse rule
        rule = RuleDefinition(
            rule_id=data["rule_id"],
            name=data["name"],
            description=data.get("description", data["name"]),
            severity=data.get("severity", "error"),
            condition=data["condition"],
            on_violation=on_violation,
            jgp_groups=data.get("jgp_groups"),
            enabled=data.get("enabled", True),
            version=data.get("version", "1.0"),
        )

        # Validate expression syntax
        validate_expression(rule.condition)

        logger.debug("Parsed rule %s from %s", rule.rule_id, source_file.name)
        return rule

    except KeyError as e:
        raise RuleParseError(
            f"Missing required field {e} in rule from {source_file}"
        ) from e
    except Exception as e:
        raise RuleParseError(
            f"Failed to parse rule in {source_file}: {e}"
        ) from e


def validate_rule(rule: RuleDefinition) -> bool:
    """
    Validate rule syntax and expression.

    Args:
        rule: Rule to validate

    Returns:
        True if valid

    Raises:
        RuleParseError: If validation fails
    """
    validate_expression(rule.condition)
    return True


# =============================================================================
# Rule Executor
# =============================================================================


class RuleExecutor:
    """
    Executes validation rules against claim records.

    Uses safe expression evaluation with restricted builtins.
    """

    def __init__(self, scorer: ImpactScorer | None = None):
        """
        Initialize executor.

        Args:
            scorer: Impact scorer for calculating violation scores
        """
        self.scorer = scorer or ImpactScorer()

    def execute(self, rule: RuleDefinition, record: dict | ClaimRecord) -> RuleResult:
        """
        Execute a single rule on a single record.

        Args:
            rule: Rule definition
            record: Claim record (dict or ClaimRecord)

        Returns:
            RuleResult with state (SAT/VIOL/WARN)
        """
        # Convert to dict if needed
        if hasattr(record, "model_dump"):
            record_dict = record.model_dump()
        else:
            record_dict = dict(record)

        case_id = record_dict.get("case_id", "unknown")

        # Guard: rule not enabled
        if not rule.enabled:
            return RuleResult(
                rule_id=rule.rule_id,
                case_id=case_id,
                state=RuleState.SAT,
                message="Rule disabled",
            )

        # Guard: JGP filter doesn't match
        if rule.jgp_groups:
            record_jgp = record_dict.get("jgp_code")
            if record_jgp and record_jgp not in rule.jgp_groups:
                return RuleResult(
                    rule_id=rule.rule_id,
                    case_id=case_id,
                    state=RuleState.SAT,
                    message="JGP not in rule scope",
                )

        # Evaluate condition
        try:
            # Add 'record' to context for expressions like record.get('field')
            context = {**record_dict, "record": record_dict}
            result = safe_eval(rule.condition, context)

            if result:
                # Condition True = rule satisfied
                return RuleResult(
                    rule_id=rule.rule_id,
                    case_id=case_id,
                    state=RuleState.SAT,
                )
            else:
                # Condition False = violation
                state = (
                    RuleState.WARN
                    if rule.severity == RuleSeverity.WARNING
                    else RuleState.VIOL
                )
                return RuleResult(
                    rule_id=rule.rule_id,
                    case_id=case_id,
                    state=state,
                    message=rule.on_violation.message,
                    autofix_hint=rule.on_violation.autofix_hint,
                )

        except RuleExecutionError as e:
            logger.error("Rule %s failed on %s: %s", rule.rule_id, case_id, e)
            return RuleResult(
                rule_id=rule.rule_id,
                case_id=case_id,
                state=RuleState.VIOL,
                message=f"Rule execution error: {e}",
            )

    def execute_batch(
        self,
        rules: list[RuleDefinition],
        records: list[dict | ClaimRecord],
    ) -> list[RuleResult]:
        """
        Execute all rules on all records.

        Args:
            rules: List of rule definitions
            records: List of claim records

        Returns:
            List of all rule results
        """
        results: list[RuleResult] = []
        enabled_rules = [r for r in rules if r.enabled]

        for record in records:
            for rule in enabled_rules:
                result = self.execute(rule, record)
                results.append(result)

        logger.debug(
            "Executed %d rules × %d records = %d results",
            len(enabled_rules),
            len(records),
            len(results),
        )

        return results


# =============================================================================
# Rule Engine
# =============================================================================


class RuleEngine:
    """
    Main orchestrator for rule validation.

    Combines rule loading, execution, and scoring into a cohesive pipeline.

    Example:
        engine = RuleEngine(Path("config/rules"))
        report = engine.validate(claim_batch)
        print(f"Violations: {report.violation_count}")
    """

    def __init__(
        self,
        rules_path: Path,
        *,
        executor: RuleExecutor | None = None,
        scorer: ImpactScorer | None = None,
    ):
        """
        Initialize engine with rules path.

        Args:
            rules_path: Path to rules directory or file
            executor: Custom rule executor (uses default if None)
            scorer: Custom impact scorer (uses default if None)
        """
        self.rules_path = Path(rules_path)
        self.scorer = scorer or ImpactScorer()
        self.executor = executor or RuleExecutor(scorer=self.scorer)
        self.rules: list[RuleDefinition] = []

        # Load rules
        self._load_rules()

    def _load_rules(self) -> None:
        """Load rules from configured path."""
        self.rules = load_rules(self.rules_path)

    def reload_rules(self) -> int:
        """
        Reload rules from disk.

        Returns:
            Number of rules loaded
        """
        self._load_rules()
        return len(self.rules)

    def validate(
        self,
        batch: ClaimBatch,
        *,
        calculate_impact: bool = True,
    ) -> ValidationReport:
        """
        Run all rules against a claim batch.

        Args:
            batch: Claim batch to validate
            calculate_impact: Whether to calculate impact scores

        Returns:
            ValidationReport with all results
        """
        # Guard: no rules
        if not self.rules:
            logger.warning("No rules loaded, returning empty report")
            return ValidationReport(
                source_file=batch.source_file,
                total_records=batch.count,
                total_rules=0,
            )

        # Execute rules
        results = self.executor.execute_batch(self.rules, batch.records)

        # Calculate impact scores
        if calculate_impact:
            records_map = {r.case_id: r for r in batch.records}
            for result in results:
                if result.is_violation or result.is_warning:
                    record = records_map.get(result.case_id)
                    if record:
                        result.impact_score = self.scorer.calculate(result, record)

        logger.info(
            "Validation complete: %d records, %d rules, %d violations",
            batch.count,
            len(self.rules),
            sum(1 for r in results if r.is_violation),
        )

        return ValidationReport(
            source_file=batch.source_file,
            total_records=batch.count,
            total_rules=len(self.rules),
            results=results,
        )

    def calculate_impact(
        self,
        violation: RuleResult,
        record: dict | ClaimRecord,
    ) -> float:
        """
        Calculate impact score for a violation.

        Args:
            violation: Rule result
            record: Associated claim record

        Returns:
            Impact score
        """
        return self.scorer.calculate(violation, record)

    def get_rule(self, rule_id: str) -> RuleDefinition | None:
        """Get rule by ID."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    @property
    def enabled_rules(self) -> list[RuleDefinition]:
        """Get only enabled rules."""
        return [r for r in self.rules if r.enabled]
