"""
First-Pass Acceptance (FPA) tracking.

FPA = (Accepted claims / Total submitted claims) × 100%

This module:
1. Imports NFZ rejection data (CSV/XML)
2. Matches rejections to original claims
3. Calculates FPA per batch, rule, department
4. Tracks FPA trends over time
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class RejectionRecord(BaseModel):
    """Single NFZ rejection record."""

    case_id: str = Field(..., description="Original case ID")
    rejection_date: date = Field(..., description="Date of rejection")
    error_code: str = Field(..., description="NFZ error code (e.g., CWV_001)")
    error_message: str = Field(..., description="NFZ error message")
    rule_id: str | None = Field(None, description="Mapped Priqualis rule ID")
    batch_id: str | None = Field(None, description="Original submission batch")


class FPAReport(BaseModel):
    """FPA metrics for a time period."""

    period_start: date
    period_end: date
    total_submitted: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    fpa_rate: float = Field(0.0, ge=0.0, le=1.0, description="FPA rate 0.0-1.0")
    fpa_by_rule: dict[str, float] = Field(default_factory=dict)
    fpa_by_department: dict[str, float] = Field(default_factory=dict)
    top_rejection_reasons: list[tuple[str, int]] = Field(default_factory=list)

    @property
    def fpa_percent(self) -> str:
        """FPA as percentage string."""
        return f"{self.fpa_rate:.1%}"


@dataclass(slots=True)
class FPATrend:
    """FPA trend over time."""

    dates: list[date] = field(default_factory=list)
    fpa_values: list[float] = field(default_factory=list)
    moving_average: list[float] = field(default_factory=list)


# =============================================================================
# Error Code Mapping (NFZ → Priqualis)
# =============================================================================


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
# Rejection Importer
# =============================================================================


class RejectionImporter:
    """Import NFZ rejection data."""

    def __init__(self, error_mapping: dict[str, str] | None = None):
        """
        Initialize importer.

        Args:
            error_mapping: NFZ code → Priqualis rule mapping
        """
        self.error_mapping = error_mapping or NFZ_ERROR_MAPPING

    def import_csv(self, path: Path) -> list[RejectionRecord]:
        """
        Import rejections from CSV file.

        Expected columns: case_id, rejection_date, error_code, error_message
        """
        df = pl.read_csv(path)
        return self.import_from_df(df)

    def import_from_df(self, df: pl.DataFrame) -> list[RejectionRecord]:
        """
        Import rejections from Polars DataFrame.

        Expected columns: case_id, rejection_date, error_code, error_message

        Args:
            df: Polars DataFrame with rejection data

        Returns:
            List of RejectionRecord objects
        """
        from datetime import datetime as dt

        records = []
        for row in df.iter_rows(named=True):
            try:
                # Handle date parsing - może być string lub date
                rejection_date = row.get("rejection_date")
                if isinstance(rejection_date, str):
                    # Try multiple date formats
                    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
                        try:
                            rejection_date = dt.strptime(rejection_date, fmt).date()
                            break
                        except ValueError:
                            continue
                elif rejection_date is None:
                    rejection_date = date.today()

                record = RejectionRecord(
                    case_id=str(row["case_id"]),
                    rejection_date=rejection_date,
                    error_code=str(row.get("error_code", "UNKNOWN")),
                    error_message=str(row.get("error_message", "")),
                    rule_id=self.error_mapping.get(str(row.get("error_code", ""))),
                    batch_id=str(row.get("batch_id")) if row.get("batch_id") else None,
                )
                records.append(record)
            except Exception as e:
                logger.warning("Failed to parse rejection row: %s - %s", row, e)
                continue

        logger.info("Imported %d rejection records from DataFrame", len(records))
        return records


# =============================================================================
# FPA Tracker
# =============================================================================


class FPATracker:
    """
    Track First-Pass Acceptance rates.

    Stores submission and rejection history to calculate FPA metrics.
    """

    def __init__(self, storage_path: Path | None = None):
        """
        Initialize tracker.

        Args:
            storage_path: Optional path to persist data
        """
        self.storage_path = storage_path
        self._submissions: list[dict] = []
        self._rejections: list[RejectionRecord] = []

    def record_submission(
        self,
        batch_id: str,
        case_ids: list[str],
        submission_date: date | None = None,
    ) -> None:
        """
        Record a batch submission for FPA tracking.

        Args:
            batch_id: Unique batch identifier
            case_ids: List of submitted case IDs
            submission_date: Submission date (default: today)
        """
        self._submissions.append({
            "batch_id": batch_id,
            "case_ids": case_ids,
            "count": len(case_ids),
            "date": submission_date or date.today(),
        })
        logger.debug("Recorded submission: %s with %d cases", batch_id, len(case_ids))

    def record_rejections(self, rejections: list[RejectionRecord]) -> None:
        """
        Record rejections received from NFZ.

        Args:
            rejections: List of rejection records
        """
        self._rejections.extend(rejections)
        logger.debug("Recorded %d rejections", len(rejections))

    def calculate_fpa(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> FPAReport:
        """
        Calculate FPA for a period.

        Args:
            start_date: Period start (default: 30 days ago)
            end_date: Period end (default: today)

        Returns:
            FPAReport with metrics
        """
        from datetime import timedelta

        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Filter submissions in period
        total_submitted = sum(
            s["count"]
            for s in self._submissions
            if start_date <= s["date"] <= end_date
        )

        # Filter rejections in period
        rejected_ids = {
            r.case_id
            for r in self._rejections
            if start_date <= r.rejection_date <= end_date
        }
        total_rejected = len(rejected_ids)
        total_accepted = total_submitted - total_rejected

        # Calculate FPA
        fpa_rate = total_accepted / total_submitted if total_submitted > 0 else 1.0

        # FPA by rule
        fpa_by_rule: dict[str, float] = {}
        for rule_id in set(r.rule_id for r in self._rejections if r.rule_id):
            rule_rejections = sum(
                1 for r in self._rejections
                if r.rule_id == rule_id and start_date <= r.rejection_date <= end_date
            )
            if total_submitted > 0:
                fpa_by_rule[rule_id] = 1 - (rule_rejections / total_submitted)

        # Top rejection reasons
        from collections import Counter
        reason_counts = Counter(
            r.error_code
            for r in self._rejections
            if start_date <= r.rejection_date <= end_date
        )
        top_reasons = reason_counts.most_common(5)

        return FPAReport(
            period_start=start_date,
            period_end=end_date,
            total_submitted=total_submitted,
            total_accepted=total_accepted,
            total_rejected=total_rejected,
            fpa_rate=fpa_rate,
            fpa_by_rule=fpa_by_rule,
            top_rejection_reasons=top_reasons,
        )

    def get_trend(self, days: int = 30, window: int = 7) -> FPATrend:
        """
        Get FPA trend with moving average.

        Args:
            days: Number of days to include
            window: Moving average window

        Returns:
            FPATrend with daily FPA values
        """
        from datetime import timedelta

        end_date = date.today()
        dates = []
        fpa_values = []

        for i in range(days):
            d = end_date - timedelta(days=days - 1 - i)
            report = self.calculate_fpa(d, d)
            dates.append(d)
            fpa_values.append(report.fpa_rate)

        # Calculate moving average
        moving_avg = []
        for i in range(len(fpa_values)):
            start = max(0, i - window + 1)
            avg = sum(fpa_values[start : i + 1]) / (i - start + 1)
            moving_avg.append(avg)

        return FPATrend(
            dates=dates,
            fpa_values=fpa_values,
            moving_average=moving_avg,
        )

    @property
    def summary(self) -> dict[str, Any]:
        """Get quick summary stats."""
        return {
            "total_submissions": len(self._submissions),
            "total_cases_submitted": sum(s["count"] for s in self._submissions),
            "total_rejections": len(self._rejections),
        }
