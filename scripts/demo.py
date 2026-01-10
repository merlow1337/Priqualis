#!/usr/bin/env python3
"""
Priqualis Demo - Full ETL + Validation Pipeline

Run with: python scripts/demo.py
"""

from pathlib import Path

from priqualis.etl import ClaimImporter
from priqualis.rules import RuleEngine


def main():
    print("=" * 60)
    print("🏥 Priqualis Demo - Claim Validation Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n📂 Loading claims...")
    importer = ClaimImporter()
    batch = importer.import_file("data/raw/claims.parquet")
    print(f"   ✅ Loaded {batch.count} claims")
    print(f"   ✅ Valid records: {batch.valid_count}")
    print(f"   ✅ Records with pre-existing errors: {batch.error_count}")

    # 2. Validate with rules
    print("\n📋 Validating with rule engine...")
    engine = RuleEngine(Path("config/rules"))
    print(f"   ✅ Loaded {len(engine.rules)} rules:")
    for rule in engine.rules:
        print(f"      - {rule.rule_id}: {rule.name}")

    report = engine.validate(batch)

    # 3. Report
    print("\n📊 Validation Results:")
    print(f"   Total records: {report.total_records}")
    print(f"   Total rule checks: {len(report.results)}")
    print(f"   Violations: {report.violation_count}")
    print(f"   Warnings: {report.warning_count}")
    print(f"   Pass rate: {report.pass_rate:.1%}")

    # 4. Top violations
    if report.violations:
        print("\n🔴 Top 10 Violations:")
        for v in report.violations[:10]:
            msg = v.message[:60] + "..." if len(v.message) > 60 else v.message
            print(f"   {v.rule_id} | {v.case_id}: {msg}")

    # 5. Summary by rule
    print("\n📈 Violations by Rule:")
    from collections import Counter
    rule_counts = Counter(v.rule_id for v in report.violations)
    for rule_id, count in rule_counts.most_common():
        rule = engine.get_rule(rule_id)
        name = rule.name if rule else "Unknown"
        print(f"   {rule_id} ({name}): {count}")

    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)

    # 6. AutoFix Demo
    print("\n🔧 AutoFix Demo:")
    from priqualis.autofix import PatchGenerator, PatchApplier

    gen = PatchGenerator()
    records_map = {r.case_id: r.model_dump() for r in batch.records}
    patches = gen.generate_batch(report.violations, records_map)
    print(f"   Generated {len(patches)} patches")

    if patches:
        # Show sample patch
        p = patches[0]
        print(f"\n   Sample patch for {p.case_id}:")
        print(f"      Rule: {p.rule_id}")
        print(f"      Confidence: {p.confidence:.0%}")
        for c in p.changes:
            print(f"      Change: {c.field} = {c.value}")
        print(f"      Rationale: {p.rationale}")

        # Apply in dry-run mode
        applier = PatchApplier()
        record = records_map[p.case_id]
        fixed = applier.apply(p, record, mode="dry-run")
        print(f"\n   After dry-run fix:")
        print(f"      {p.changes[0].field}: {record.get(p.changes[0].field)} → {fixed.get(p.changes[0].field)}")


if __name__ == "__main__":
    main()
