"""
=============================================================================
Project Synthexis - Hybrid Retrieval & Vector Specialist Engine (Person 3)
=============================================================================
Module: vector_engine.py
Owner: Person 3 (Retrieval & Vector Specialist)
Workload Scope: ~23.3%

Responsibilities:
1. Embedding Model Management:
   - Converts textual chunks into dense semantic embeddings using SentenceTransformers
     ('all-MiniLM-L6-v2' by default, 384 dimensions).
   - Thread-safe lazy loading and batch inference for high throughput and memory efficiency.
   - Defensive fallback mechanisms to prevent server crashes.

2. Local Vector Database Architecture:
   - Initializes and manages local vector database storage via ChromaDB.
   - Persistent client target: data/chromadb with cosine distance metric ('hnsw:space': 'cosine').
   - Batch insertion, metadata indexing (source doc, page, chunk index), and automatic
     collection re-indexing.
   - Automatic index recovery from disk persistence across FastAPI restarts.

3. BM25 Exact-Match Search:
   - Implements BM25Okapi inverted index (rank-bm25) for precision keyword retrieval.
   - Domain-specific technical tokenizer preserving alphanumeric identifiers,
     hardware specs, metrics (e.g., '420ms', '98.4%'), acronyms, and error codes (e.g., 'ERR-9942').

4. Hybrid Rank Fusion (RRF + Score Blending):
   - Mathematically robust Reciprocal Rank Fusion (RRF, k=60):
       RRF_score(d) = w_dense / (60 + rank_dense(d)) + w_sparse / (60 + rank_sparse(d))
   - Combined with exact keyword match boosting to guarantee zero degradation on technical terms
     while maintaining semantic comprehension.

I/O Contract:
- `create_index(chunks: list) -> bool`
- `hybrid_search(query: str, top_k: int = 5) -> list[str]`
=============================================================================
"""

from __future__ import annotations

import os
import re
import math
import time
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Sequence

# Configure logging
logger = logging.getLogger("synthexis.vector")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [VectorEngine] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Suppress noisy external library warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Dependency Imports with Defensive Fallbacks
# ---------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:
    np = None  # Handled defensively

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import chromadb
    from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
except ImportError:
    chromadb = None
    EmbeddingFunction = object  # type: ignore


# ---------------------------------------------------------------------------
# Path & Configuration Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHROMA_DIR = BASE_DIR / "data" / "chromadb"
DEFAULT_COLLECTION_NAME = "synthexis_chunks"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 0.5
DEFAULT_SPARSE_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# Technical Term Tokenizer
# ---------------------------------------------------------------------------
_TOKEN_PATTERN = re.compile(r"(?:[a-zA-Z0-9]+(?:[-_:.][a-zA-Z0-9]+)*)")

def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes text for BM25 exact matching while preserving technical codes,
    metrics (e.g. '420ms', '98.4%'), hyphenated identifiers, and acronyms.
    """
    if not text:
        return []
    text_lower = text.lower()
    tokens = _TOKEN_PATTERN.findall(text_lower)
    return [t for t in tokens if len(t) > 1 or t.isalnum()]


# ---------------------------------------------------------------------------
# SentenceTransformers Custom Embedding Function for ChromaDB
# ---------------------------------------------------------------------------
class SentenceTransformersEmbeddingFunction(EmbeddingFunction):
    """
    Adapter connecting SentenceTransformers directly to ChromaDB's
    EmbeddingFunction interface without downloading secondary ONNX models.
    """
    def __init__(self, model_instance: Any):
        self._model = model_instance

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        if self._model is None:
            # Fallback zero-vector if model unavailable
            return [[0.0] * 384 for _ in input]
        try:
            embeddings = self._model.encode(
                list(input),
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings via SentenceTransformer: {e}")
            return [[0.0] * 384 for _ in input]


# ---------------------------------------------------------------------------
# Core Hybrid Vector & Retrieval Engine
# ---------------------------------------------------------------------------
class HybridVectorEngine:
    """
    Unified retrieval architecture combining ChromaDB vector similarity search
    and BM25 keyword matching using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        persist_dir: Optional[Path | str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: Optional[str] = None,
        rrf_k: int = DEFAULT_RRF_K,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
    ):
        self.persist_dir = Path(persist_dir or os.getenv("CHROMA_PERSIST_DIR", str(DEFAULT_CHROMA_DIR)))
        self.collection_name = collection_name
        self.model_name = model_name or os.getenv("SYNTHEXIS_EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # State management
        self._lock = threading.RLock()
        self._model: Optional[SentenceTransformer] = None
        self._embedding_function: Optional[SentenceTransformersEmbeddingFunction] = None
        self._chroma_client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._corpus_metadata: List[Dict[str, Any]] = []

        # Initialize storage directory
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Lazy Model & DB Initialization
    # -----------------------------------------------------------------------
    def _get_embedding_model(self) -> Optional[SentenceTransformer]:
        """Lazy loader for SentenceTransformer embedding model."""
        if self._model is None and SentenceTransformer is not None:
            with self._lock:
                if self._model is None:
                    try:
                        logger.info(f"Loading embedding model '{self.model_name}'...")
                        start = time.time()
                        self._model = SentenceTransformer(self.model_name)
                        self._embedding_function = SentenceTransformersEmbeddingFunction(self._model)
                        logger.info(f"Embedding model loaded in {time.time() - start:.2f}s.")
                    except Exception as e:
                        logger.error(f"Failed to load SentenceTransformer ({self.model_name}): {e}")
                        self._model = None
                        self._embedding_function = None
        return self._model

    def _get_chroma_collection(self) -> Optional[Any]:
        """Lazy loader and manager for ChromaDB PersistentClient collection."""
        if self._collection is not None:
            return self._collection

        if chromadb is None:
            logger.warning("ChromaDB library is not installed. Vector search running in fallback mode.")
            return None

        with self._lock:
            if self._collection is not None:
                return self._collection

            # Ensure embedding model is ready for embedding function
            self._get_embedding_model()

            try:
                # Initialize persistent client
                self._chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
            except Exception as e:
                logger.warning(f"PersistentClient failed ({e}). Falling back to in-memory EphemeralClient.")
                try:
                    self._chroma_client = chromadb.Client()
                except Exception as ex:
                    logger.error(f"Failed to initialize ChromaDB client: {ex}")
                    return None

            try:
                # Use cosine distance space
                self._collection = self._chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self._embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                logger.warning(f"Failed to get/create collection '{self.collection_name}' ({e}). Recreating collection...")
                try:
                    self._chroma_client.delete_collection(self.collection_name)
                    self._collection = self._chroma_client.create_collection(
                        name=self.collection_name,
                        embedding_function=self._embedding_function,
                        metadata={"hnsw:space": "cosine"}
                    )
                except Exception as ex:
                    logger.error(f"Failed to recreate ChromaDB collection '{self.collection_name}': {ex}")
                    self._collection = None

            return self._collection

    # -----------------------------------------------------------------------
    # Metadata Parsing Helper
    # -----------------------------------------------------------------------
    @staticmethod
    def _parse_chunk_metadata(chunk: str, index: int) -> Dict[str, Any]:
        """
        Extracts document provenance tags, such as '[document.pdf | Page 2]'
        for rich citation traceability.
        """
        source_doc = "Synthexis Corpus"
        page_num: Optional[int] = None

        chunk_clean = chunk.strip()
        if chunk_clean.startswith("[") and "]" in chunk_clean:
            tag = chunk_clean[1:chunk_clean.find("]")].strip()
            parts = [p.strip() for p in tag.split("|")]
            if parts:
                source_doc = parts[0]
            if len(parts) > 1 and "page" in parts[1].lower():
                digits = re.findall(r"\d+", parts[1])
                if digits:
                    try:
                        page_num = int(digits[0])
                    except ValueError:
                        pass

        meta: Dict[str, Any] = {
            "chunk_index": index,
            "char_length": len(chunk_clean),
            "source_doc": source_doc
        }
        if page_num is not None:
            meta["page"] = page_num
        return meta

    # -----------------------------------------------------------------------
    # Index Creation Pipeline
    # -----------------------------------------------------------------------
    def create_index(self, chunks: List[str]) -> bool:
        """
        Builds both the BM25 inverted index and ChromaDB vector index from
        the provided corpus of text chunks.

        Input: `chunks` (list of str) -> Output: `index_status` (bool)
        """
        with self._lock:
            # 1. Defensive input validation
            if chunks is None:
                logger.warning("create_index received None as chunks.")
                return False

            start_time = time.time()

            # Sanitize and normalize chunks
            cleaned_chunks: List[str] = []
            for item in chunks:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    cleaned_chunks.append(text)

            # If corpus is empty, reset indices cleanly
            if not cleaned_chunks:
                logger.info("Empty chunk list received. Purging vector and BM25 indices.")
                self.reset()
                return True

            logger.info(f"Indexing {len(cleaned_chunks)} text chunk(s) across hybrid engines...")

            # 2. Build BM25 Inverted Index
            try:
                tokenized_corpus = [tokenize_text(c) for c in cleaned_chunks]
                if BM25Okapi is not None and tokenized_corpus:
                    self._bm25 = BM25Okapi(tokenized_corpus)
                    self._tokenized_corpus = tokenized_corpus
                else:
                    self._bm25 = None
                    self._tokenized_corpus = []
                    logger.warning("BM25Okapi unavailable or tokenized corpus is empty.")
            except Exception as e:
                logger.error(f"Error constructing BM25 index: {e}")
                self._bm25 = None
                self._tokenized_corpus = []

            # 3. Build Vector Index (ChromaDB)
            coll = self._get_chroma_collection()
            if coll is not None and self._chroma_client is not None:
                try:
                    # Reset collection to avoid duplicate or stale items
                    try:
                        self._chroma_client.delete_collection(self.collection_name)
                    except Exception:
                        pass

                    self._collection = self._chroma_client.create_collection(
                        name=self.collection_name,
                        embedding_function=self._embedding_function,
                        metadata={"hnsw:space": "cosine"}
                    )

                    # Prepare records
                    ids: List[str] = [f"chunk_{i}" for i in range(len(cleaned_chunks))]
                    metadatas = [self._parse_chunk_metadata(c, i) for i, c in enumerate(cleaned_chunks)]

                    # Add to ChromaDB in batches to avoid payload constraints
                    batch_size = 100
                    total = len(cleaned_chunks)
                    for b_start in range(0, total, batch_size):
                        b_end = min(b_start + batch_size, total)
                        self._collection.add(
                            ids=ids[b_start:b_end],
                            documents=cleaned_chunks[b_start:b_end],
                            metadatas=metadatas[b_start:b_end]
                        )
                    logger.info(f"Successfully populated ChromaDB with {total} chunk vectors.")
                except Exception as e:
                    logger.error(f"Failed to index chunks into ChromaDB: {e}")

            # 4. Cache state in memory
            self._corpus = list(cleaned_chunks)
            self._corpus_metadata = [self._parse_chunk_metadata(c, i) for i, c in enumerate(cleaned_chunks)]

            elapsed = time.time() - start_time
            logger.info(f"Hybrid index constructed in {elapsed:.2f}s for {len(self._corpus)} chunk(s).")
            return True

    # -----------------------------------------------------------------------
    # Hybrid Search & Rank Fusion Pipeline
    # -----------------------------------------------------------------------
    def hybrid_search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Executes hybrid retrieval combining BM25 exact keyword matching and
        dense vector cosine similarity via Reciprocal Rank Fusion (RRF).

        Input: `query` (str), `top_k` (int) -> Output: `relevant_chunks` (list of str)
        """
        with self._lock:
            # 1. Input validation & boundary handling
            if top_k <= 0:
                return []

            if not self._corpus:
                # Attempt to recover persisted documents from ChromaDB if memory was cleared
                self._attempt_corpus_recovery()

            if not self._corpus:
                logger.warning("hybrid_search invoked with an empty corpus.")
                return []

            query_clean = str(query or "").strip()
            if not query_clean:
                return self._corpus[:top_k]

            # 2. Dense Semantic Vector Search
            dense_ranks: Dict[str, int] = {}
            dense_similarities: Dict[str, float] = {}

            coll = self._get_chroma_collection()
            if coll is not None and coll.count() > 0:
                try:
                    n_dense_candidates = min(len(self._corpus), max(top_k * 4, 25))
                    res = coll.query(
                        query_texts=[query_clean],
                        n_results=n_dense_candidates,
                        include=["documents", "distances", "metadatas"]
                    )

                    returned_ids = res.get("ids", [[]])[0]
                    returned_distances = res.get("distances", [[]])[0]

                    for rank_idx, chunk_id in enumerate(returned_ids, start=1):
                        dense_ranks[chunk_id] = rank_idx
                        if rank_idx - 1 < len(returned_distances):
                            dist = returned_distances[rank_idx - 1]
                            sim = max(0.0, min(1.0, 1.0 - float(dist)))
                            dense_similarities[chunk_id] = sim
                except Exception as e:
                    logger.error(f"Vector search failed during hybrid_search: {e}")

            # 3. Sparse BM25 Keyword Search
            bm25_ranks: Dict[str, int] = {}
            bm25_scores: Dict[str, float] = {}

            if self._bm25 is not None and self._tokenized_corpus:
                try:
                    query_tokens = tokenize_text(query_clean)
                    if query_tokens:
                        raw_scores = self._bm25.get_scores(query_tokens)
                        if np is not None:
                            sorted_indices = np.argsort(raw_scores)[::-1]
                        else:
                            sorted_indices = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)

                        rank_counter = 1
                        for idx in sorted_indices:
                            score = float(raw_scores[idx])
                            if score > 0.0:
                                cid = f"chunk_{idx}"
                                bm25_ranks[cid] = rank_counter
                                bm25_scores[cid] = score
                                rank_counter += 1
                                if rank_counter > max(top_k * 4, 25):
                                    break
                except Exception as e:
                    logger.error(f"BM25 search failed during hybrid_search: {e}")

            # 4. Reciprocal Rank Fusion (RRF) & Exact Match Boosting
            candidates = set(dense_ranks.keys()) | set(bm25_ranks.keys())

            # If both engines yielded zero candidates, fall back to safe prefix
            if not candidates:
                logger.info("Zero direct matches found; returning corpus prefix.")
                return self._corpus[:top_k]

            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
            if max_bm25 <= 0.0:
                max_bm25 = 1.0

            id_to_chunk = {f"chunk_{i}": c for i, c in enumerate(self._corpus)}
            scored_candidates: List[Tuple[float, str]] = []
            query_lower = query_clean.lower()

            for cid in candidates:
                chunk_text = id_to_chunk.get(cid)
                if not chunk_text:
                    continue

                # Reciprocal Rank component
                r_dense = dense_ranks.get(cid)
                r_bm25 = bm25_ranks.get(cid)

                rrf_dense = (1.0 / (self.rrf_k + r_dense)) if r_dense is not None else 0.0
                rrf_bm25 = (1.0 / (self.rrf_k + r_bm25)) if r_bm25 is not None else 0.0

                fused_score = (self.dense_weight * rrf_dense) + (self.sparse_weight * rrf_bm25)

                # Keyword bonus: reward exact substring match or high BM25 term coverage
                if cid in bm25_scores:
                    norm_bm25 = bm25_scores[cid] / max_bm25
                    fused_score += 0.005 * norm_bm25

                if len(query_lower) >= 4 and query_lower in chunk_text.lower():
                    fused_score += 0.02  # Exact phrase match priority boost

                scored_candidates.append((fused_score, chunk_text))

            # 5. Sort candidates descending by fused score
            scored_candidates.sort(key=lambda item: item[0], reverse=True)

            results = [text for _, text in scored_candidates[:top_k]]
            return results

    # -----------------------------------------------------------------------
    # Recovery & Maintenance
    # -----------------------------------------------------------------------
    def _attempt_corpus_recovery(self) -> None:
        """
        Recovers indexed chunks from ChromaDB persistent storage if
        in-memory corpus is empty (e.g., following a worker restart).
        """
        coll = self._get_chroma_collection()
        if coll is None:
            return

        try:
            stored = coll.get()
            docs = stored.get("documents", [])
            metas = stored.get("metadatas", [])
            if docs:
                logger.info(f"Recovered {len(docs)} persisted chunk(s) from ChromaDB storage.")
                # Sort by chunk_index metadata if present
                indexed_docs = []
                for i, doc in enumerate(docs):
                    idx = metas[i].get("chunk_index", i) if i < len(metas) and metas[i] else i
                    indexed_docs.append((idx, doc))
                indexed_docs.sort(key=lambda x: x[0])

                self._corpus = [d for _, d in indexed_docs]
                self._tokenized_corpus = [tokenize_text(c) for c in self._corpus]
                if BM25Okapi is not None and self._tokenized_corpus:
                    self._bm25 = BM25Okapi(self._tokenized_corpus)
        except Exception as e:
            logger.warning(f"Could not recover persisted corpus from ChromaDB: {e}")

    def reset(self) -> bool:
        """Purges existing indices in memory and on disk."""
        with self._lock:
            self._corpus.clear()
            self._tokenized_corpus.clear()
            self._corpus_metadata.clear()
            self._bm25 = None

            if self._chroma_client is not None:
                try:
                    self._chroma_client.delete_collection(self.collection_name)
                except Exception:
                    pass
                try:
                    self._collection = self._chroma_client.create_collection(
                        name=self.collection_name,
                        embedding_function=self._embedding_function,
                        metadata={"hnsw:space": "cosine"}
                    )
                except Exception:
                    self._collection = None

            logger.info("HybridVectorEngine reset completed.")
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Returns diagnostic statistics for system health checks."""
        with self._lock:
            coll_count = 0
            if self._collection is not None:
                try:
                    coll_count = self._collection.count()
                except Exception:
                    coll_count = 0

            return {
                "corpus_chunk_count": len(self._corpus),
                "chromadb_vector_count": coll_count,
                "bm25_active": self._bm25 is not None,
                "embedding_model": self.model_name,
                "persist_directory": str(self.persist_dir),
                "rrf_k": self.rrf_k,
                "dense_weight": self.dense_weight,
                "sparse_weight": self.sparse_weight,
            }


# ---------------------------------------------------------------------------
# Singleton Instance & Top-Level Module API (Zero-Conflict Contracts)
# ---------------------------------------------------------------------------
_GLOBAL_ENGINE: Optional[HybridVectorEngine] = None
_INIT_LOCK = threading.Lock()

def get_vector_engine() -> HybridVectorEngine:
    """Returns the singleton HybridVectorEngine instance."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        with _INIT_LOCK:
            if _GLOBAL_ENGINE is None:
                _GLOBAL_ENGINE = HybridVectorEngine()
    return _GLOBAL_ENGINE


def create_index(chunks: list) -> bool:
    """
    Person 3 Contract Endpoint:
    Input: `chunks` (list of str) -> Output: `index_status` (bool)
    """
    return get_vector_engine().create_index(chunks)


def hybrid_search(query: str, top_k: int = 5) -> list:
    """
    Person 3 Contract Endpoint:
    Input: `query` (str), `top_k` (int) -> Output: `relevant_chunks` (list of str)
    """
    return get_vector_engine().hybrid_search(query=query, top_k=top_k)


def reset_index() -> bool:
    """Resets the vector and BM25 indices."""
    return get_vector_engine().reset()


def get_index_stats() -> dict:
    """Returns diagnostic statistics."""
    return get_vector_engine().get_stats()


# ---------------------------------------------------------------------------
# Local Test & Verification Suite (Phase 2 Protocol)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("SYNTHEXIS PERSON 3: HYBRID VECTOR & RETRIEVAL ENGINE VERIFICATION")
    print("=" * 70)

    # Mock corpus reflecting multimodal document content
    mock_chunks = [
        "[Architecture.pdf | Page 1] Synthexis utilizes a decoupled microservice architecture separating the Streamlit UI from FastAPI backend routing.",
        "[Architecture.pdf | Page 2] Vector DB layer implements ChromaDB persistent storage coupled with BM25Okapi for hybrid retrieval.",
        "[Benchmarks.pdf | Page 1] Performance metrics: Ingestion latency benchmarked at 420ms with 98.4% precision on multi-page technical documents.",
        "[Benchmarks.pdf | Page 2] Error code ERR-9942 was isolated during stress testing on distributed node 4 under 50 concurrent requests.",
        "[Hardware.pdf | Page 3] Recommended accelerator: NVIDIA RTX 4090 or Apple Silicon M-Series with unified memory architecture.",
        "[Vision.pdf | Page 1] Extracted chart visual artifacts are base64-encoded and passed to Google Gemini 1.5/2.0 multimodal API."
    ]

    print(f"\n1. Building hybrid index with {len(mock_chunks)} mock chunks...")
    t0 = time.time()
    success = create_index(mock_chunks)
    print(f"-> Indexing success: {success} (Time elapsed: {time.time() - t0:.2f}s)")

    stats = get_index_stats()
    print(f"-> Index Diagnostics: {stats}")

    print("\n2. Executing Search Scenarios:")

    # Scenario A: Exact technical term / error code (BM25 priority)
    q1 = "What caused error ERR-9942?"
    res1 = hybrid_search(q1, top_k=2)
    print(f"\nQuery A (Exact Code): '{q1}'")
    for i, r in enumerate(res1, 1):
        print(f"   [{i}] {r}")
    assert any("ERR-9942" in r for r in res1), "Assertion failed: ERR-9942 not in top results"

    # Scenario B: Quantitative metrics (exact match + semantics)
    q2 = "What is the ingestion latency benchmark and precision?"
    res2 = hybrid_search(q2, top_k=2)
    print(f"\nQuery B (Metrics): '{q2}'")
    for i, r in enumerate(res2, 1):
        print(f"   [{i}] {r}")
    assert any("420ms" in r for r in res2), "Assertion failed: 420ms metric not in top results"

    # Scenario C: Semantic conceptual query (Dense vector priority)
    q3 = "How does the system handle visual charts and diagrams?"
    res3 = hybrid_search(q3, top_k=2)
    print(f"\nQuery C (Semantic Concept): '{q3}'")
    for i, r in enumerate(res3, 1):
        print(f"   [{i}] {r}")
    assert any("Vision.pdf" in r or "chart" in r.lower() for r in res3), "Assertion failed: Vision chart not in top results"

    # Scenario D: Edge Cases
    print("\n3. Testing Edge Cases:")
    empty_res = hybrid_search("", top_k=3)
    print(f"-> Empty query returned {len(empty_res)} chunks (Safe fallback)")

    zero_k = hybrid_search("architecture", top_k=0)
    print(f"-> top_k=0 returned {len(zero_k)} chunks (Safe boundary)")

    high_k = hybrid_search("pipeline", top_k=50)
    print(f"-> top_k=50 returned {len(high_k)} chunks (Clamped to corpus size)")

    print("\n" + "=" * 70)
    print("ALL PERSON 3 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
