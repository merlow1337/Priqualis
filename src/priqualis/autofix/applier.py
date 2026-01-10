"""
Patch Applier for Priqualis.

Applies generated patches to claim records with audit trail.
"""

import copy
import logging
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml

from priqualis.autofix.generator import AuditEntry, Patch, PatchOperation
from priqualis.core.exceptions import AutoFixError
from priqualis.rules.models import AutoFixOperation

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ApplyMode(str, Enum):
    """Mode for applying patches."""

    DRY_RUN = "dry-run"   # Return modified copy, don't change original
    COMMIT = "commit"      # Modify in place


# =============================================================================
# Patch Applier
# =============================================================================


class PatchApplier:
    """
    Applies patches to claim records.

    Supports dry-run mode for preview and commit mode for actual changes.
    Creates audit trail for all applied patches.
    """

    def __init__(self, audit_dir: Path | None = None):
        """
        Initialize applier.

        Args:
            audit_dir: Directory to store audit logs (optional)
        """
        self.audit_dir = audit_dir
        self._applied_patches: list[AuditEntry] = []

    @property
    def applied_patches(self) -> list[AuditEntry]:
        """Get list of applied patches."""
        return self._applied_patches.copy()

    def apply(
        self,
        patch: Patch,
        record: dict,
        mode: ApplyMode | str = ApplyMode.DRY_RUN,
    ) -> dict:
        """
        Apply patch to a record.

        Args:
            patch: Patch to apply
            record: Target record (dict or Pydantic model)
            mode: "dry-run" (return modified copy) or "commit" (modify in place)

        Returns:
            Modified record

        Raises:
            AutoFixError: If patch cannot be applied
        """
        # Normalize mode
        if isinstance(mode, str):
            mode = ApplyMode(mode)

        # Get record data
        if hasattr(record, "model_dump"):
            record_dict = record.model_dump()
        else:
            record_dict = record

        # Guard: case_id must match
        if record_dict.get("case_id") != patch.case_id:
            raise AutoFixError(
                f"Case ID mismatch: patch is for {patch.case_id}, "
                f"record is {record_dict.get('case_id')}"
            )

        # Create working copy for dry-run, or use original for commit
        if mode == ApplyMode.DRY_RUN:
            working = copy.deepcopy(record_dict)
        else:
            working = record_dict

        # Apply each operation
        for op in patch.changes:
            self._apply_operation(op, working)

        # Log
        logger.info(
            "Applied patch %s to %s (mode=%s, %d changes)",
            patch.rule_id,
            patch.case_id,
            mode.value,
            len(patch.changes),
        )

        return working

    def _apply_operation(
        self,
        op: PatchOperation,
        record: dict,
    ) -> None:
        """Apply a single operation to record."""
        field = op.field
        value = op.value

        # Handle nested fields (e.g., "patient.name")
        parts = field.split(".")
        target = record
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        final_field = parts[-1]

        # Get current value for logging
        old_value = target.get(final_field)

        # Apply based on operation type
        op_type = op.op if isinstance(op.op, str) else op.op.value

        if op_type == "add_if_absent":
            if final_field not in target or target[final_field] is None:
                target[final_field] = value
                logger.debug("Added %s = %s", field, value)
            else:
                logger.debug("Skipped %s (already present)", field)

        elif op_type == "set":
            target[final_field] = value
            logger.debug("Set %s: %s → %s", field, old_value, value)

        elif op_type == "remove":
            if final_field in target:
                del target[final_field]
                logger.debug("Removed %s (was %s)", field, old_value)

        elif op_type == "replace":
            target[final_field] = value
            logger.debug("Replaced %s: %s → %s", field, old_value, value)

        else:
            raise AutoFixError(f"Unknown operation type: {op_type}")

    def create_audit_entry(
        self,
        patch: Patch,
        user: str = "system",
        approved: bool = False,
    ) -> AuditEntry:
        """
        Create audit trail entry for an applied patch.

        Args:
            patch: Applied patch
            user: User who applied/approved the patch
            approved: Whether patch was manually approved

        Returns:
            AuditEntry for the applied patch
        """
        entry = AuditEntry(
            patch_id=str(uuid.uuid4())[:8],
            case_id=patch.case_id,
            rule_id=patch.rule_id,
            user=user,
            changes=patch.changes,
            approved=approved,
        )

        self._applied_patches.append(entry)

        # Save to file if audit_dir configured
        if self.audit_dir:
            self._save_audit_entry(entry)

        return entry

    def _save_audit_entry(self, entry: AuditEntry) -> None:
        """Save audit entry to file."""
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp
        timestamp = entry.applied_at.strftime("%Y%m%d_%H%M%S")
        filename = f"audit_{timestamp}_{entry.patch_id}.yaml"
        filepath = self.audit_dir / filename

        # Serialize
        data = {
            "patch_id": entry.patch_id,
            "case_id": entry.case_id,
            "rule_id": entry.rule_id,
            "user": entry.user,
            "approved": entry.approved,
            "applied_at": entry.applied_at.isoformat(),
            "changes": [
                {
                    "op": c.op if isinstance(c.op, str) else c.op.value,
                    "field": c.field,
                    "value": c.value,
                    "old_value": c.old_value,
                }
                for c in entry.changes
            ],
        }

        try:
            filepath.write_text(yaml.dump(data, allow_unicode=True))
            logger.debug("Saved audit entry to %s", filepath)
        except Exception as e:
            logger.error("Failed to save audit entry: %s", e)

    def apply_batch(
        self,
        patches: list[Patch],
        records: dict[str, dict],
        mode: ApplyMode | str = ApplyMode.DRY_RUN,
        user: str = "system",
    ) -> dict[str, dict]:
        """
        Apply multiple patches to records.

        Args:
            patches: List of patches to apply
            records: Mapping of case_id to record
            mode: Apply mode
            user: User applying the patches

        Returns:
            Mapping of case_id to modified record
        """
        results: dict[str, dict] = {}
        applied_count = 0
        skipped_count = 0

        for patch in patches:
            record = records.get(patch.case_id)

            if record is None:
                logger.warning("No record found for case %s", patch.case_id)
                skipped_count += 1
                continue

            try:
                modified = self.apply(patch, record, mode)
                results[patch.case_id] = modified

                # Create audit entry for committed changes
                if mode == ApplyMode.COMMIT:
                    self.create_audit_entry(patch, user)

                applied_count += 1

            except AutoFixError as e:
                logger.error("Failed to apply patch to %s: %s", patch.case_id, e)
                skipped_count += 1

        logger.info(
            "Batch apply complete: %d applied, %d skipped (mode=%s)",
            applied_count,
            skipped_count,
            mode.value if isinstance(mode, ApplyMode) else mode,
        )

        return results

    def preview_changes(
        self,
        patches: list[Patch],
    ) -> list[dict]:
        """
        Generate preview of changes without applying them.

        Args:
            patches: Patches to preview

        Returns:
            List of change summaries
        """
        previews = []

        for patch in patches:
            preview = {
                "case_id": patch.case_id,
                "rule_id": patch.rule_id,
                "changes": [
                    {
                        "field": c.field,
                        "operation": c.op if isinstance(c.op, str) else c.op.value,
                        "old_value": c.old_value,
                        "new_value": c.value,
                    }
                    for c in patch.changes
                ],
                "rationale": patch.rationale,
                "confidence": f"{patch.confidence:.0%}",
                "is_safe": patch.is_safe,
            }
            previews.append(preview)

        return previews


# =============================================================================
# Convenience Functions
# =============================================================================


def apply_patch(
    patch: Patch,
    record: dict,
    mode: str = "dry-run",
) -> dict:
    """
    Convenience function to apply a single patch.

    Args:
        patch: Patch to apply
        record: Target record
        mode: "dry-run" or "commit"

    Returns:
        Modified record
    """
    applier = PatchApplier()
    return applier.apply(patch, record, ApplyMode(mode))


def export_patches_yaml(
    patches: list[Patch],
    output_path: Path,
) -> None:
    """
    Export patches to YAML file.

    Args:
        patches: Patches to export
        output_path: Output file path
    """
    data = {
        "patches": [p.to_yaml_dict() for p in patches],
        "generated_at": datetime.now().isoformat(),
        "count": len(patches),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))

    logger.info("Exported %d patches to %s", len(patches), output_path)
