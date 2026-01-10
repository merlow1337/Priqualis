"""
Report Generator for Priqualis.

Generates validation reports in Markdown, PDF, and JSON formats.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from priqualis.rules.models import ValidationReport, RuleResult

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(slots=True, frozen=True)
class ReportConfig:
    """
    Configuration for report generation.
    
    Attributes:
        title: Report title
        include_violations_detail: Include per-case violation details
        include_recommendations: Generate AI recommendations
        max_violations_shown: Limit violations in detail section
        language: Report language (pl/en)
    """

    title: str = "Priqualis Validation Report"
    include_violations_detail: bool = True
    include_recommendations: bool = True
    max_violations_shown: int = 50
    language: Literal["pl", "en"] = "en"


DEFAULT_CONFIG = ReportConfig()


# =============================================================================
# Report Generator
# =============================================================================


class ReportGenerator:
    """
    Generates validation reports in multiple formats.
    
    Supports Markdown, PDF (via weasyprint), and JSON output.
    """

    def __init__(self, config: ReportConfig | None = None):
        """
        Initialize generator.

        Args:
            config: Report configuration
        """
        self.config = config or DEFAULT_CONFIG

    def generate_markdown(
        self,
        report: ValidationReport,
        batch_id: str | None = None,
    ) -> str:
        """
        Generate Markdown report.

        Args:
            report: Validation report from RuleEngine
            batch_id: Optional batch identifier

        Returns:
            Markdown string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Calculate metrics
        pass_rate = report.pass_rate * 100
        violation_rate = (1 - report.pass_rate) * 100

        # Count by rule
        from collections import Counter
        rule_counts = Counter(v.rule_id for v in report.violations)
        top_rules = rule_counts.most_common(5)

        # Build markdown
        lines = [
            f"# {self.config.title}",
            "",
            f"**Batch ID:** `{batch_id}`",
            f"**Generated:** {timestamp}",
            "",
            "---",
            "",
            "## 📊 Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Claims | {report.total_records:,} |",
            f"| ✅ Passed | {report.total_records - report.violation_count:,} ({pass_rate:.1f}%) |",
            f"| ❌ Violations | {report.violation_count:,} ({violation_rate:.1f}%) |",
            f"| ⚠️ Warnings | {report.warning_count:,} |",
            "",
        ]

        # Top violation rules
        if top_rules:
            lines.extend([
                "## 🔝 Top Violation Rules",
                "",
                "| Rank | Rule ID | Count | % of Violations |",
                "|------|---------|-------|-----------------|",
            ])
            for rank, (rule_id, count) in enumerate(top_rules, 1):
                pct = count / report.violation_count * 100 if report.violation_count else 0
                lines.append(f"| {rank} | `{rule_id}` | {count:,} | {pct:.1f}% |")
            lines.append("")

        # Violations detail
        if self.config.include_violations_detail and report.violations:
            lines.extend([
                "## ❌ Violations Detail",
                "",
            ])
            
            shown = min(len(report.violations), self.config.max_violations_shown)
            for v in report.violations[:shown]:
                lines.extend([
                    f"### Case: `{v.case_id}`",
                    f"- **Rule:** `{v.rule_id}`",
                    f"- **Message:** {v.message or 'N/A'}",
                    f"- **State:** `{v.state}`",
                    "",
                ])
            
            if len(report.violations) > shown:
                lines.append(f"*... and {len(report.violations) - shown} more violations*")
                lines.append("")

        # Recommendations
        if self.config.include_recommendations:
            recommendations = self._generate_recommendations(report, top_rules)
            if recommendations:
                lines.extend([
                    "## 💡 Recommendations",
                    "",
                ])
                for i, rec in enumerate(recommendations, 1):
                    lines.append(f"{i}. {rec}")
                lines.append("")

        # Footer
        lines.extend([
            "---",
            "",
            "*Generated by Priqualis v0.1.0*",
        ])

        return "\n".join(lines)

    def generate_json(
        self,
        report: ValidationReport,
        batch_id: str | None = None,
    ) -> dict:
        """
        Generate JSON report.

        Args:
            report: Validation report
            batch_id: Optional batch identifier

        Returns:
            Dictionary suitable for JSON serialization
        """
        from collections import Counter
        rule_counts = Counter(v.rule_id for v in report.violations)

        return {
            "batch_id": batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_records": report.total_records,
                "violations": report.violation_count,
                "warnings": report.warning_count,
                "pass_rate": report.pass_rate,
            },
            "violations_by_rule": dict(rule_counts),
            "top_rules": rule_counts.most_common(5),
            "violations_detail": [
                {
                    "case_id": v.case_id,
                    "rule_id": v.rule_id,
                    "message": v.message,
                    "state": v.state if isinstance(v.state, str) else v.state.value if hasattr(v.state, 'value') else str(v.state),
                }
                for v in report.violations[:self.config.max_violations_shown]
            ],
        }

    def generate_pdf(
        self,
        report: ValidationReport,
        output_path: Path,
        batch_id: str | None = None,
    ) -> Path:
        """
        Generate PDF report.

        Requires weasyprint to be installed.

        Args:
            report: Validation report
            output_path: Path to save PDF
            batch_id: Optional batch identifier

        Returns:
            Path to generated PDF
        """
        try:
            import markdown
            from weasyprint import HTML
        except ImportError as e:
            logger.error("PDF generation requires 'weasyprint' and 'markdown': %s", e)
            raise ImportError(
                "PDF generation requires 'weasyprint' and 'markdown'. "
                "Install with: pip install weasyprint markdown"
            ) from e

        # Generate markdown first
        md_content = self.generate_markdown(report, batch_id)

        # Convert to HTML
        html_body = markdown.markdown(
            md_content,
            extensions=["tables", "fenced_code"],
        )

        # Wrap in styled template
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            color: #1a5f7a;
            border-bottom: 2px solid #1a5f7a;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2a7f9a;
            margin-top: 30px;
        }}
        h3 {{
            color: #444;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px 12px;
            text-align: left;
        }}
        th {{
            background: #1a5f7a;
            color: white;
        }}
        tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', monospace;
        }}
        hr {{
            border: none;
            border-top: 1px solid #ddd;
            margin: 30px 0;
        }}
        em {{
            color: #666;
        }}
    </style>
</head>
<body>
{{html_body}}
</body>
</html>
"""

        # Generate PDF
        output_path = Path(output_path)
        HTML(string=html_template).write_pdf(output_path)
        
        logger.info("Generated PDF report: %s", output_path)
        return output_path

    def _generate_recommendations(
        self,
        report: ValidationReport,
        top_rules: list[tuple[str, int]],
    ) -> list[str]:
        """
        Generate automated recommendations based on validation results.

        Args:
            report: Validation report
            top_rules: Top violation rules with counts

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # High violation rate
        if report.pass_rate < 0.9:
            recommendations.append(
                f"**High violation rate ({(1-report.pass_rate)*100:.1f}%)** - "
                "Review data entry procedures and consider additional training."
            )

        # Dominant rule
        if top_rules and report.violation_count > 0:
            top_rule, top_count = top_rules[0]
            top_pct = top_count / report.violation_count * 100
            if top_pct > 30:
                recommendations.append(
                    f"**Rule `{top_rule}`** accounts for {top_pct:.0f}% of violations - "
                    "Focus remediation efforts on this specific issue."
                )

        # Many warnings
        if report.warning_count > report.violation_count:
            recommendations.append(
                "**High warning count** - Review warning-level rules to prevent "
                "potential future rejections."
            )

        # Good pass rate
        if report.pass_rate >= 0.95:
            recommendations.append(
                "✅ **Excellent compliance** - Pass rate above 95%. "
                "Continue current data quality practices."
            )

        return recommendations


# =============================================================================
# Convenience Function
# =============================================================================


def generate_batch_report(
    report: ValidationReport,
    output_dir: Path | str,
    batch_id: str | None = None,
    formats: list[Literal["markdown", "json", "pdf"]] | None = None,
    config: ReportConfig | None = None,
) -> dict[str, Path]:
    """
    Generate batch report in multiple formats.

    Convenience function that generates all requested format files.

    Args:
        report: Validation report from RuleEngine
        output_dir: Directory to save reports
        batch_id: Optional batch identifier
        formats: List of formats to generate (default: ["markdown", "json"])
        config: Report configuration

    Returns:
        Dictionary mapping format name to output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = formats or ["markdown", "json"]
    batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    generator = ReportGenerator(config)
    outputs: dict[str, Path] = {}

    if "markdown" in formats:
        md_content = generator.generate_markdown(report, batch_id)
        md_path = output_dir / f"{batch_id}.md"
        md_path.write_text(md_content, encoding="utf-8")
        outputs["markdown"] = md_path
        logger.info("Generated Markdown report: %s", md_path)

    if "json" in formats:
        import json
        json_content = generator.generate_json(report, batch_id)
        json_path = output_dir / f"{batch_id}.json"
        json_path.write_text(json.dumps(json_content, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs["json"] = json_path
        logger.info("Generated JSON report: %s", json_path)

    if "pdf" in formats:
        pdf_path = output_dir / f"{batch_id}.pdf"
        generator.generate_pdf(report, pdf_path, batch_id)
        outputs["pdf"] = pdf_path

    return outputs
