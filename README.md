# Priqualis

**Pre-submission compliance validator for healthcare claim batches (NFZ/JGP)**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

Priqualis validates healthcare billing packages before submission to NFZ (Polish National Health Fund), reducing rejections and speeds up reimbursement. It uses rule-based validation with hybrid similarity search to find similar approved cases and generate safe auto-fix suggestions.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Rule Engine** | YAML-based DSL with three-state outcomes (SAT/VIOL/WARN) and impact scoring |
| **Hybrid Similarity** | BM25 + vector ANN retrieval (Qdrant) with optional cross-encoder re-rank |
| **AutoFix** | Generates `patch.yaml` with auditable field-level corrections |
| **Shadow Mode** | Import payer rejections to track First-Pass Acceptance (FPA) over time |
| **Batch Reports** | Export validation summaries to Markdown, PDF, or JSON |
| **LLM Explain** | AI-generated explanations citing NFZ rule base (CWV/JGP) |
| **Anomaly Alerts** | Z-score based detection when error-codes spike |
| **PII Masking** | Deterministic hashing keeps masked data joinable without PII leaks |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  CSV / Parquet  │────▶│  ETL + PII Mask │────▶│  Rule Engine    │
│  (claims data)  │     │  (importers.py) │     │  (7 rules)      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐              ▼
                        │    AutoFix      │◀────┌─────────────────┐
                        │   Generator     │     │  Hybrid Search  │
                        │ (generator.py)  │     │  BM25 + Vector  │
                        └─────────────────┘     └─────────────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              ▼
│  Streamlit UI   │◀───▶│    FastAPI      │◀────┌─────────────────┐
│   (app.py)      │     │   /api/v1/*     │     │  FPA Tracker    │
└─────────────────┘     └─────────────────┘     │  (Shadow Mode)  │
                                                └─────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|-------|------------|---------|
| **Data Processing** | Polars, Pydantic v2 | ≥1.20, ≥2.10 |
| **Search (Sparse)** | bm25s | ≥0.2 |
| **Search (Dense)** | Qdrant (HNSW) | ≥1.12 |
| **Embeddings** | intfloat/multilingual-e5-small | 384 dims |
| **Reranking** | sentence-transformers CrossEncoder | ms-marco-MiniLM |
| **API** | FastAPI + Uvicorn | ≥0.115 |
| **UI** | Streamlit | ≥1.40 |

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- pip or uv package manager

### Installation

```bash
# Clone repository
git clone https://github.com/SirSail/Priqualis.git
cd Priqualis-bigdata

# Install dependencies
pip install -e ".[dev]"

# Or with uv (faster)
uv pip install -e ".[dev]"
```

### Generate Synthetic Data

```bash
# Generate 10k synthetic claims with ~20% intentional errors
python scripts/generate_synthetic.py --count 10000 --output data/raw/claims.parquet
```

### Run the Application

```bash
# Option 1: Run Streamlit UI (recommended for demo)
python -m streamlit run ui/app.py

# Option 2: Run FastAPI backend
python -m uvicorn api.main:app --reload --port 8000

# Option 3: Run demo script (ETL + validation + autofix)
python scripts/demo.py
```

> **💡 Windows Note:** If `streamlit` or `uvicorn` commands fail with "not recognized", 
> use `python -m streamlit` or `python -m uvicorn` as shown above.

---

## 📁 Project Structure

```
Priqualis-bigdata/
├── ui/
│   └── app.py               # 🖥️ Streamlit UI (main entry point)
├── api/
│   ├── main.py              # FastAPI application
│   ├── deps.py              # Dependency injection
│   └── routes/              # API endpoints
│       ├── validate.py      # POST /api/v1/validate
│       ├── similar.py       # POST /api/v1/similar
│       ├── autofix.py       # POST /api/v1/autofix/*
│       └── reports.py       # GET /api/v1/reports/*
├── config/
│   └── rules/               # YAML validation rules
│       ├── base.yaml        # R001-R005: core rules
│       └── jgp_validation.yaml  # R006-R007: JGP-specific
├── data/
│   ├── raw/                 # Input data (claims.parquet)
│   ├── processed/           # ETL output
│   └── fixtures/            # Sample test data
├── scripts/
│   ├── demo.py              # Full pipeline demo
│   ├── generate_synthetic.py    # Synthetic data generator
│   └── benchmark_fpa_search.py  # Performance benchmark
├── src/priqualis/           # 📦 Main package
│   ├── core/                # Config, exceptions
│   │   ├── config.py        # Settings (pydantic-settings)
│   │   └── exceptions.py    # Custom exceptions
│   ├── etl/                 # Data processing
│   │   ├── importers.py     # CSV/Parquet loading
│   │   ├── schemas.py       # Pydantic models (ClaimRecord, ClaimBatch)
│   │   ├── pii_masking.py   # PESEL/name masking
│   │   └── processor.py     # ETL pipeline
│   ├── rules/               # Validation engine
│   │   ├── engine.py        # RuleEngine, YAML parser
│   │   ├── models.py        # RuleResult, ValidationReport
│   │   └── scoring.py       # Impact score calculation
│   ├── search/              # Similarity search
│   │   ├── bm25.py          # BM25 sparse retrieval
│   │   ├── vector.py        # Qdrant vector store
│   │   ├── hybrid.py        # RRF/Linear fusion
│   │   ├── rerank.py        # Cross-encoder reranking
│   │   └── service.py       # SimilarityService
│   ├── autofix/             # Patch generation
│   │   ├── generator.py     # Patch generation from violations
│   │   └── applier.py       # Patch application
│   ├── shadow/              # FPA tracking
│   │   ├── fpa.py           # FPA tracker, rejection import
│   │   └── alerts.py        # Anomaly detection (Z-score)
│   ├── llm/                 # AI explanations
│   │   ├── explainer.py     # Violation explanations
│   │   └── rag.py           # RAG store for NFZ rules
│   └── reports/             # Report generation
│       └── generator.py     # Markdown/PDF/JSON reports
├── tests/                   # 🧪 Unit tests (74 tests)
│   ├── test_etl/
│   ├── test_rules/
│   └── test_search/
├── pyproject.toml           # Dependencies & build config
└── README.md
```

---

## 📋 Validation Rules

| Rule | Name | Severity | AutoFix | Description |
|------|------|----------|---------|-------------|
| R001 | Required Main Diagnosis | error | ✅ | ICD-10 main diagnosis required |
| R002 | Valid Date Range | error | ✅ | Discharge must be ≥ admission |
| R003 | JGP Code Required | error | ✅ | DRG classification required |
| R004 | Procedures Required | warning | ❌ | At least one procedure code |
| R005 | Valid Admission Mode | error | ✅ | Must be emergency/planned/transfer |
| R006 | Department Code Required | error | ✅ | NFZ department code required |
| R007 | Positive Tariff Value | warning | ✅ | Tariff must be > 0 |

**AutoFix Coverage:** 6/7 rules (86%)

---

## 🖥️ UI Pages

### 1. Dashboard
- Overview metrics (claims validated, violations, pass rate)
- Recent validation history
- Quick navigation

### 2. Triage (Main Workflow)
- Upload CSV/Parquet files
- Run batch validation
- View violations by rule
- **AutoFix**: Generate patches, preview (dry-run), apply
- Export reports (Markdown/PDF/JSON)
- LLM explanations for violations

### 3. Similar Cases
- Find similar approved cases for violations
- Attribute diff visualization
- Generate patches from similar cases

### 4. KPIs
- First-Pass Acceptance (FPA) rate
- Error distribution by rule
- Trend charts
- **Shadow Mode**: Import NFZ rejections
- Anomaly alerts

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file in project root:

```env
# General
PRIQUALIS_ENV=development
LOG_LEVEL=INFO

# Paths
DATA_RAW_PATH=./data/raw
DATA_PROCESSED_PATH=./data/processed
RULES_CONFIG_PATH=./config/rules

# Qdrant (vector store)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=claims_embeddings

# Embeddings
EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_DEVICE=cpu

# Search
BM25_K1=1.5
BM25_B=0.75
HYBRID_ALPHA=0.5
SEARCH_TOP_K=50
RERANK_ENABLED=false

# API
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 📊 Performance

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **10k batch processing** | ≤60s | **1.5s** | ✅ 40x faster |
| **Error detection** | 20-30% | **100%** | ✅ All injected errors caught |
| **AutoFix coverage** | ≥40% | **86%** | ✅ 6/7 rules |
| **Similar query P95** | <300ms | **1.3ms** | ✅ 225x faster |
| **FPA tracking** | Functional | **85%** | ✅ Complete |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html

# Run specific module tests
pytest tests/test_etl/ -v
pytest tests/test_rules/ -v
pytest tests/test_search/ -v

# Run benchmark
python scripts/benchmark_fpa_search.py
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Slow validation (5+ minutes for 1500 records)**
- Check if validation loop is correct (should be O(n), not O(n²))
- Make sure `engine.validate()` is called ONCE after collecting all records

**2. `AttributeError: 'RejectionImporter' object has no attribute 'import_from_df'`**
- Add `import_from_df()` method to `RejectionImporter` class in `fpa.py`

**3. `KeyError: slice(None, 10, None)` on dict**
- Dict comprehension doesn't support slicing `[:10]`
- Use `dict(list(d.items())[:10])` instead

**4. Qdrant connection refused**
- Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
- Or use in-memory mode: `VectorStore(in_memory=True)`

**5. Embedding model download slow**
- First run downloads ~100MB model
- Cache stored in `~/.cache/huggingface/`

---

## 📚 Domain Context (NFZ/Poland)

| Term | Description |
|------|-------------|
| **NFZ** | Narodowy Fundusz Zdrowia (National Health Fund) - central public payer |
| **JGP** | Jednorodne Grupy Pacjentów (DRG) - diagnosis-related groups for billing |
| **CWV** | Centralne Warunki Walidacji - central validation conditions |
| **CRW** | Centralne Reguły Weryfikacji - central verification rules |
| **SWIAD** | XML message format for claim submissions |
| **PESEL** | Polish national ID number (11 digits) |

---

## 🗺️ Roadmap

- [x] ETL + PII Masking
- [x] Rule Engine (7 rules)
- [x] AutoFix Generator + Applier
- [x] Hybrid Search (BM25 + Vector)
- [x] Streamlit UI
- [x] FPA Tracking (Shadow Mode)
- [x] Anomaly Alerts
- [x] LLM Explanations (RAG)
- [ ] FastAPI endpoints (partial)
- [ ] PDF export (requires weasyprint)
- [ ] Cross-encoder reranking (optional)
- [ ] Multi-language support

---

## 👥 Authors

- **Jakub Zeglinski** - [GitHub](https://github.com/SirSail)
- **Alexander Fichtenberg**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 References

1. [NFZ - Walidacje i weryfikacje](https://www.nfz.gov.pl/dla-swiadczeniodawcy/sprawozdawczosc-elektroniczna/walidacje-i-weryfikacje/)
2. [NFZ - CWV/CRW zestawienie zbiorcze](https://www.nfz.gov.pl/dla-swiadczeniodawcy/sprawozdawczosc-elektroniczna/walidacje-i-weryfikacje/zestawienie-zbiorcze,6464.html)
3. [Opis algorytmu grupera JGP 2024](https://www.nfz.gov.pl/download/gfx/nfz/pl/defaultaktualnosci/354/52/1/opis_algorytmu_grupera_2024.docx)