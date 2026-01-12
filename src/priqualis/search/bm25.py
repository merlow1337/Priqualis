"""
BM25 Index for Priqualis.

Sparse retrieval using bm25s library.
"""

import logging
from pathlib import Path
from typing import Protocol

import bm25s

from priqualis.core.exceptions import SearchError

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class TextTokenizer(Protocol):
    """Protocol for text tokenization."""

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms."""
        ...


# =============================================================================
# Simple Tokenizer
# =============================================================================


class SimpleTokenizer:
    """
    Simple whitespace tokenizer with lowercasing.

    Suitable for structured medical codes (ICD-10, JGP, procedures).
    """

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text by whitespace."""
        if self.lowercase:
            text = text.lower()
        return text.split()

    def tokenize_batch(self, texts: list[str]) -> list[list[str]]:
        """Tokenize multiple texts."""
        return [self.tokenize(t) for t in texts]


# =============================================================================
# BM25 Index
# =============================================================================


class BM25Index:
    """
    BM25 sparse retrieval index.

    Uses bm25s library for fast indexing and retrieval.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: TextTokenizer | None = None,
    ):
        """
        Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            tokenizer: Text tokenizer (uses SimpleTokenizer if None)
        """
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.index: bm25s.BM25 | None = None
        self.corpus_ids: list[str] = []
        self._is_built = False

    @property
    def is_built(self) -> bool:
        """Check if index is built."""
        return self._is_built and self.index is not None

    def build(self, documents: list[tuple[str, str]]) -> None:
        """
        Build index from (case_id, text) pairs.

        Args:
            documents: List of (case_id, searchable_text) tuples
        """
        if not documents:
            logger.warning("No documents to index")
            return

        self.corpus_ids = [doc[0] for doc in documents]
        texts = [doc[1] for doc in documents]

        # Tokenize
        logger.debug("Tokenizing %d documents", len(texts))
        corpus_tokens = bm25s.tokenize(texts, stopwords=None)

        # Build index
        logger.debug("Building BM25 index with k1=%.2f, b=%.2f", self.k1, self.b)
        self.index = bm25s.BM25()
        self.index.index(corpus_tokens)

        self._is_built = True
        logger.info("Built BM25 index with %d documents", len(documents))

    def search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        """
        Search index for top-k results.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of (case_id, score) tuples sorted by score descending
        """
        # Guard: index not built
        if not self.is_built:
            raise SearchError("BM25 index not built. Call build() first.")

        # Tokenize query
        query_tokens = bm25s.tokenize([query], stopwords=None)

        # Search (index is guaranteed to be not None after is_built check)
        assert self.index is not None
        results, scores = self.index.retrieve(query_tokens, k=min(top_k, len(self.corpus_ids)))

        # Convert to (case_id, score) pairs
        output: list[tuple[str, float]] = []
        for idx, score in zip(results[0], scores[0]):
            if idx < len(self.corpus_ids):
                output.append((self.corpus_ids[idx], float(score)))

        return output

    def save(self, path: Path) -> None:
        """
        Save index to disk.

        Args:
            path: Directory to save index
        """
        if not self.is_built:
            raise SearchError("Cannot save: index not built")

        path.mkdir(parents=True, exist_ok=True)

        # Save bm25s index
        assert self.index is not None  # guaranteed by is_built check
        self.index.save(str(path / "bm25_index"))

        # Save corpus IDs
        import json
        (path / "corpus_ids.json").write_text(json.dumps(self.corpus_ids))

        logger.info("Saved BM25 index to %s", path)

    def load(self, path: Path) -> None:
        """
        Load index from disk.

        Args:
            path: Directory containing saved index
        """
        if not path.exists():
            raise SearchError(f"Index path not found: {path}")

        # Load bm25s index
        self.index = bm25s.BM25.load(str(path / "bm25_index"), load_corpus=False)

        # Load corpus IDs
        import json
        self.corpus_ids = json.loads((path / "corpus_ids.json").read_text())

        self._is_built = True
        logger.info("Loaded BM25 index from %s (%d documents)", path, len(self.corpus_ids))

    def __len__(self) -> int:
        """Number of indexed documents."""
        return len(self.corpus_ids)
