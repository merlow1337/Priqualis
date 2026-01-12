"""
Anomaly Detection and Alerting for Priqualis.

Detects unusual patterns in validation results and generates alerts.
Uses Z-score based detection on historical rule violation counts.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class Alert(BaseModel):
    """Anomaly alert model."""

    alert_id: str = Field(..., description="Unique alert ID")
    alert_type: Literal["zscore", "trend", "threshold"] = Field(
        ..., description="Type of anomaly detection"
    )
    severity: Literal["info", "warning", "critical"] = Field(
        ..., description="Alert severity"
    )
    rule_id: str = Field(..., description="Related rule ID")
    message: str = Field(..., description="Human-readable alert message")
    current_value: int = Field(..., description="Current count")
    threshold: float = Field(..., description="Threshold that triggered alert")
    z_score: float | None = Field(None, description="Z-score if applicable")
    detected_at: datetime = Field(default_factory=datetime.now)

    @property
    def icon(self) -> str:
        """Get severity icon."""
        icons = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
        return icons.get(self.severity, "⚪")


@dataclass(slots=True, frozen=True)
class AlertConfig:
    """
    Configuration for anomaly detection.
    
    Attributes:
        zscore_threshold: Z-score above which to trigger alert
        min_history_days: Minimum days of history needed
        critical_zscore: Z-score for critical severity
        enable_trend_detection: Enable trend-based alerts
    """

    zscore_threshold: float = 2.0
    min_history_days: int = 7
    critical_zscore: float = 3.0
    enable_trend_detection: bool = True


DEFAULT_ALERT_CONFIG = AlertConfig()


# =============================================================================
# Anomaly Detector
# =============================================================================


class AnomalyDetector:
    """
    Detects anomalies in validation results.
    
    Uses statistical methods (Z-score) to identify unusual patterns
    in rule violation counts compared to historical averages.
    """

    def __init__(
        self,
        config: AlertConfig | None = None,
        history: dict[str, list[int]] | None = None,
    ):
        """
        Initialize detector.

        Args:
            config: Alert configuration
            history: Historical rule counts (rule_id -> list of daily counts)
        """
        self.config = config or DEFAULT_ALERT_CONFIG
        self._history: dict[str, list[int]] = history or {}

    def record_batch(self, rule_counts: dict[str, int], batch_date: date | None = None) -> None:
        """
        Record batch results for history.

        Args:
            rule_counts: Dictionary of rule_id -> violation count
            batch_date: Date of batch (default: today)
        """
        batch_date = batch_date or date.today()
        
        for rule_id, count in rule_counts.items():
            if rule_id not in self._history:
                self._history[rule_id] = []
            self._history[rule_id].append(count)
            
            # Keep only last 90 days
            if len(self._history[rule_id]) > 90:
                self._history[rule_id] = self._history[rule_id][-90:]

        logger.debug("Recorded batch with %d rules", len(rule_counts))

    def detect_zscore(
        self,
        rule_id: str,
        current_count: int,
    ) -> Alert | None:
        """
        Detect anomaly using Z-score.

        Args:
            rule_id: Rule identifier
            current_count: Current violation count

        Returns:
            Alert if anomaly detected, None otherwise
        """
        historical = self._history.get(rule_id, [])
        
        if len(historical) < self.config.min_history_days:
            logger.debug("Not enough history for %s (%d days)", rule_id, len(historical))
            return None

        # Calculate statistics
        mean = sum(historical) / len(historical)
        variance = sum((x - mean) ** 2 for x in historical) / len(historical)
        std = variance ** 0.5

        if std == 0:
            # No variation in history
            if current_count > mean:
                z_score = float("inf")
            else:
                return None
        else:
            z_score = (current_count - mean) / std

        # Check threshold
        if z_score < self.config.zscore_threshold:
            return None

        # Determine severity
        if z_score >= self.config.critical_zscore:
            severity = "critical"
        elif z_score >= self.config.zscore_threshold:
            severity = "warning"
        else:
            severity = "info"

        return Alert(
            alert_id=f"ALERT_{rule_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            alert_type="zscore",
            severity=severity,
            rule_id=rule_id,
            message=(
                f"Rule {rule_id} violation count ({current_count}) is "
                f"{z_score:.1f} standard deviations above average ({mean:.0f})"
            ),
            current_value=current_count,
            threshold=self.config.zscore_threshold,
            z_score=z_score,
        )

    def detect_trend(
        self,
        rule_id: str,
        window: int = 7,
    ) -> Alert | None:
        """
        Detect increasing trend in violations.

        Args:
            rule_id: Rule identifier
            window: Number of days to analyze

        Returns:
            Alert if upward trend detected, None otherwise
        """
        if not self.config.enable_trend_detection:
            return None

        historical = self._history.get(rule_id, [])
        
        if len(historical) < window:
            return None

        recent = historical[-window:]
        older = historical[-window*2:-window] if len(historical) >= window * 2 else historical[:window]

        if not older:
            return None

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        # Check for significant increase (50%+)
        if older_avg > 0 and recent_avg > older_avg * 1.5:
            increase_pct = ((recent_avg - older_avg) / older_avg) * 100

            return Alert(
                alert_id=f"TREND_{rule_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                alert_type="trend",
                severity="warning",
                rule_id=rule_id,
                message=(
                    f"Rule {rule_id} shows {increase_pct:.0f}% increase "
                    f"over last {window} days (avg {recent_avg:.0f} vs {older_avg:.0f})"
                ),
                current_value=int(recent_avg),
                threshold=1.5,
            )

        return None

    def check_batch(
        self,
        rule_counts: dict[str, int],
        record: bool = True,
    ) -> list[Alert]:
        """
        Check batch for anomalies.

        Args:
            rule_counts: Dictionary of rule_id -> violation count
            record: Whether to record this batch in history

        Returns:
            List of alerts detected
        """
        alerts: list[Alert] = []

        for rule_id, count in rule_counts.items():
            # Z-score check
            zscore_alert = self.detect_zscore(rule_id, count)
            if zscore_alert:
                alerts.append(zscore_alert)

            # Trend check
            trend_alert = self.detect_trend(rule_id)
            if trend_alert:
                alerts.append(trend_alert)

        # Record after checking
        if record:
            self.record_batch(rule_counts)

        if alerts:
            logger.warning("Detected %d anomaly alerts", len(alerts))

        return alerts

    def get_history(self, rule_id: str) -> list[int]:
        """Get historical counts for a rule."""
        return self._history.get(rule_id, [])

    def get_statistics(self, rule_id: str) -> dict[str, Any]:
        """Get statistical summary for a rule."""
        historical = self._history.get(rule_id, [])
        
        if not historical:
            return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0}

        mean = sum(historical) / len(historical)
        variance = sum((x - mean) ** 2 for x in historical) / len(historical)
        
        return {
            "count": len(historical),
            "mean": mean,
            "std": variance ** 0.5,
            "min": min(historical),
            "max": max(historical),
        }


# =============================================================================
# Alert Manager (optional notifications)
# =============================================================================


class AlertManager:
    """
    Manages alerts and optional notifications.
    
    Stores alerts and can send notifications via various channels.
    """

    def __init__(self):
        """Initialize manager."""
        self._alerts: list[Alert] = []

    def add_alert(self, alert: Alert) -> None:
        """Add alert to store."""
        self._alerts.append(alert)
        logger.info("Alert added: %s - %s", alert.alert_id, alert.message)

    def add_alerts(self, alerts: list[Alert]) -> None:
        """Add multiple alerts."""
        for alert in alerts:
            self.add_alert(alert)

    def get_alerts(
        self,
        severity: str | None = None,
        days: int = 7,
    ) -> list[Alert]:
        """
        Get recent alerts.

        Args:
            severity: Filter by severity (optional)
            days: Number of days to include

        Returns:
            List of matching alerts
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        alerts = [a for a in self._alerts if a.detected_at >= cutoff]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.detected_at, reverse=True)

    def get_critical_alerts(self, days: int = 7) -> list[Alert]:
        """Get critical alerts from last N days."""
        return self.get_alerts(severity="critical", days=days)

    def clear_old_alerts(self, days: int = 30) -> int:
        """
        Clear alerts older than N days.

        Returns:
            Number of alerts cleared
        """
        cutoff = datetime.now() - timedelta(days=days)
        before_count = len(self._alerts)
        
        self._alerts = [a for a in self._alerts if a.detected_at >= cutoff]
        
        cleared = before_count - len(self._alerts)
        logger.info("Cleared %d old alerts", cleared)
        return cleared

    @property
    def summary(self) -> dict[str, int]:
        """Get alert summary."""
        return {
            "total": len(self._alerts),
            "critical": len([a for a in self._alerts if a.severity == "critical"]),
            "warning": len([a for a in self._alerts if a.severity == "warning"]),
            "info": len([a for a in self._alerts if a.severity == "info"]),
        }
