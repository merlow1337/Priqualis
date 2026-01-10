# Priqualis

**Pre-submission compliance validator for healthcare claim batches**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-74%20passed-brightgreen.svg)]()

Priqualis validates healthcare billing packages before submission to NFZ (Polish National Health Fund), reducing rejections and accelerating reimbursement. It combines rule-based validation with hybrid similarity search to surface similar approved cases and generate safe auto-fix suggestions.

## Features

- **Rule Engine** — YAML-based DSL with three-state outcomes (SAT/VIOL/WARN) and impact scoring
- **Hybrid Similarity** — BM25 + vector ANN retrieval with optional cross-encoder re-rank
- **AutoFix** — Generates patches with auditable field-level corrections
- **Shadow Mode** — Import payer rejections to track First-Pass Acceptance (FPA) over time
- **PII Masking** — Deterministic hashing ensures joinable masked data without PII leaks

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  CSV/       │────▶│  ETL +      │────▶│  Rule Engine    │
│  Parquet    │     │  PII Mask   │     │  (7 rules)      │
└─────────────┘     └─────────────┘     └────────┬────────┘
                                                 │
                    ┌─────────────┐              ▼
                    │  AutoFix    │◀────┌─────────────────┐
                    │  Generator  │     │  Hybrid Search  │
                    └─────────────┘     │  BM25 + Vector  │
                                        └─────────────────┘
                                                 │
                    ┌─────────────┐              ▼
                    │  FastAPI    │◀────┌─────────────────┐
                    │  /api/v1    │     │  FPA Tracker    │
                    └─────────────┘     │  (Shadow Mode)  │
                                        └─────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Processing | Polars, Pydantic v2 |
| Search | bm25s (sparse), Qdrant (vector) |
| Embeddings | intfloat/multilingual-e5-small |
| Reranking | sentence-transformers CrossEncoder |
| API | FastAPI + Uvicorn |
| UI | Streamlit |

## Quick Start

```bash
# Clone and install
git clone https://github.com/SirSail/Priqualis.git
cd Priqualis-bigdata
pip install -e ".[dev]"

# Generate synthetic data
python scripts/generate_synthetic.py --count 10000 --output data/raw/claims.parquet

# Run demo (ETL + validation + autofix)
python scripts/demo.py

# Run API
python -m uvicorn api.main:app --reload

# Run UI
python -m streamlit run ui/app.py
```

## Project Structure

```
Priqualis-bigdata/
├── api/                     # FastAPI application
│   ├── main.py              # App entry point
│   ├── deps.py              # Dependency injection
│   └── routes/              # API routers
│       ├── validate.py      # POST /validate, /validate/file
│       ├── similar.py       # POST /similar, /similar/batch
│       ├── autofix.py       # POST /autofix/generate, /apply
│       └── reports.py       # GET /reports/kpis
├── config/
│   └── rules/               # YAML validation rules
│       ├── base.yaml        # R001-R005: core rules
│       └── jgp_validation.yaml  # R006-R007: JGP-specific
├── data/
│   ├── raw/                 # Input data (claims.parquet)
│   ├── processed/           # ETL output + approved claims
│   └── fixtures/            # Test samples (20 records)
├── scripts/
│   ├── demo.py              # Full pipeline demo
│   └── generate_synthetic.py # Synthetic data generator
├── src/priqualis/
│   ├── core/                # Config, exceptions
│   ├── etl/                 # Importers, schemas, PII masking
│   ├── rules/               # Engine, models, scoring
│   ├── search/              # BM25, vector, hybrid, rerank
│   ├── autofix/             # Patch generator & applier
│   └── shadow/              # FPA tracker, rejection importer
├── tests/                   # 74 unit tests
│   ├── test_etl/
│   ├── test_rules/
│   └── test_search/
└── ui/
    └── app.py               # Streamlit dashboard
```

## Validation Rules

| Rule | Name | Severity | Description |
|------|------|----------|-------------|
| R001 | Required Main Diagnosis | error | ICD-10 main diagnosis required |
| R002 | Valid Date Range | error | Discharge must be ≥ admission |
| R003 | JGP Code Required | error | DRG classification required |
| R004 | Procedures Required | warning | At least one procedure code |
| R005 | Valid Admission Mode | error | Must be emergency/planned/transfer |
| R006 | Department Code Required | error | NFZ department code required |
| R007 | Positive Tariff Value | warning | Tariff must be > 0 |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/validate` | Validate batch of claims |
| POST | `/api/v1/validate/file` | Upload and validate CSV/Parquet |
| GET | `/api/v1/validate/rules` | List all validation rules |
| POST | `/api/v1/similar` | Find similar approved cases |
| POST | `/api/v1/autofix/generate` | Generate fix patches |
| POST | `/api/v1/autofix/apply` | Apply patches (dry-run/commit) |
| GET | `/api/v1/reports/kpis` | Get FPA and validation metrics |

## Target KPIs

| Metric | Target |
|--------|--------|
| Formal error reduction | 20-30% |
| Violations with AutoFix | ≥40% |
| FPA improvement | +15-25 pp |
| 10k batch processing | ≤60s |
| Similar query P95 | <300 ms |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src/priqualis --cov-report=html

# Quick tests (no search)
python -m pytest tests/test_etl/ tests/test_rules/ -v
```

## License

MIT


