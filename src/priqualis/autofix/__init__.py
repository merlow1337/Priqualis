"""
AutoFix module for Priqualis.

Generates and applies patches to fix rule violations.
"""

from priqualis.autofix.applier import (
    ApplyMode,
    PatchApplier,
    apply_patch,
    export_patches_yaml,
)
from priqualis.autofix.generator import (
    AuditEntry,
    DEFAULT_VALUES,
    Patch,
    PatchGenerator,
    PatchOperation,
    SUGGESTED_FIXES,
)

__all__ = [
    # Models
    "Patch",
    "PatchOperation",
    "AuditEntry",
    # Generator
    "PatchGenerator",
    "DEFAULT_VALUES",
    "SUGGESTED_FIXES",
    # Applier
    "PatchApplier",
    "ApplyMode",
    "apply_patch",
    "export_patches_yaml",
]