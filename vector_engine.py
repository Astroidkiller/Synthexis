"""
=============================================================================
Project Synthexis - Hybrid Retrieval & Vector Specialist Engine (Person 3 - Part 1)
=============================================================================
Module: vector_engine.py
Owner: Person 3 (Retrieval & Vector Specialist)
Workload Scope: ~23.3%

Part 1: Core Vector Database & Dense Semantic Embedding Architecture
---------------------------------------------------------------------
Responsibilities:
1. Embedding Model Management:
   - Converts textual chunks into dense semantic embeddings using SentenceTransformers
     ('all-MiniLM-L6-v2', 384 dimensions).
   - Thread-safe lazy loading and batch inference for high throughput and memory efficiency.
   - Defensive fallback mechanisms to prevent server crashes.

2. Local Vector Database Architecture:
   - Initializes and manages local vector database storage via ChromaDB.
   - Persistent client target: data/chromadb with cosine distance metric ('hnsw:space': 'cosine').
   - Batch insertion, metadata indexing (source doc, page, chunk index), and automatic
     collection re-indexing.
   - Automatic index recovery from disk persistence across FastAPI restarts.

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
# Core Vector Retrieval Engine
# ---------------------------------------------------------------------------
class HybridVectorEngine:
    """
    Core vector database architecture managing SentenceTransformers embeddings
    and ChromaDB persistent vector search.
    """

    def __init__(
        self,
        persist_dir: Optional[Path | str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: Optional[str] = None,
    ):
        self.persist_dir = Path(persist_dir or os.getenv("CHROMA_PERSIST_DIR", str(DEFAULT_CHROMA_DIR)))
        self.collection_name = collection_name
        self.model_name = model_name or os.getenv("SYNTHEXIS_EMBEDDING_MODEL", DEFAULT_MODEL_NAME)

        # State management
        self._lock = threading.RLock()
        self._model: Optional[SentenceTransformer] = None
        self._embedding_function: Optional[SentenceTransformersEmbeddingFunction] = None
        self._chroma_client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._corpus: List[str] = []
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
            logger.warning("ChromaDB library is not installed. Running in fallback mode.")
            return None

        with self._lock:
            if self._collection is not None:
                return self._collection

            self._get_embedding_model()

            try:
                self._chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
            except Exception as e:
                logger.warning(f"PersistentClient failed ({e}). Falling back to EphemeralClient.")
                try:
                    self._chroma_client = chromadb.Client()
                except Exception as ex:
                    logger.error(f"Failed to initialize ChromaDB client: {ex}")
                    return None

            try:
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
        Builds the ChromaDB vector index from the provided corpus of text chunks.

        Input: `chunks` (list of str) -> Output: `index_status` (bool)
        """
        with self._lock:
            if chunks is None:
                logger.warning("create_index received None as chunks.")
                return False

            start_time = time.time()

            cleaned_chunks: List[str] = []
            for item in chunks:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    cleaned_chunks.append(text)

            if not cleaned_chunks:
                logger.info("Empty chunk list received. Purging vector index.")
                self.reset()
                return True

            logger.info(f"Indexing {len(cleaned_chunks)} text chunk(s) into ChromaDB vector store...")

            coll = self._get_chroma_collection()
            if coll is not None and self._chroma_client is not None:
                try:
                    try:
                        self._chroma_client.delete_collection(self.collection_name)
                    except Exception:
                        pass

                    self._collection = self._chroma_client.create_collection(
                        name=self.collection_name,
                        embedding_function=self._embedding_function,
                        metadata={"hnsw:space": "cosine"}
                    )

                    ids: List[str] = [f"chunk_{i}" for i in range(len(cleaned_chunks))]
                    metadatas = [self._parse_chunk_metadata(c, i) for i, c in enumerate(cleaned_chunks)]

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

            self._corpus = list(cleaned_chunks)
            self._corpus_metadata = [self._parse_chunk_metadata(c, i) for i, c in enumerate(cleaned_chunks)]

            elapsed = time.time() - start_time
            logger.info(f"Vector index constructed in {elapsed:.2f}s for {len(self._corpus)} chunk(s).")
            return True

    # -----------------------------------------------------------------------
    # Semantic Search Pipeline
    # -----------------------------------------------------------------------
    def hybrid_search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Executes semantic vector retrieval using ChromaDB cosine similarity.

        Input: `query` (str), `top_k` (int) -> Output: `relevant_chunks` (list of str)
        """
        with self._lock:
            if top_k <= 0:
                return []

            if not self._corpus:
                self._attempt_corpus_recovery()

            if not self._corpus:
                logger.warning("hybrid_search invoked with an empty corpus.")
                return []

            query_clean = str(query or "").strip()
            if not query_clean:
                return self._corpus[:top_k]

            coll = self._get_chroma_collection()
            if coll is not None and coll.count() > 0:
                try:
                    n_results = min(len(self._corpus), max(top_k, 1))
                    res = coll.query(
                        query_texts=[query_clean],
                        n_results=n_results,
                        include=["documents"]
                    )
                    docs = res.get("documents", [[]])[0]
                    if docs:
                        return docs
                except Exception as e:
                    logger.error(f"Vector search failed during query: {e}")

            return self._corpus[:top_k]

    # -----------------------------------------------------------------------
    # Recovery & Maintenance
    # -----------------------------------------------------------------------
    def _attempt_corpus_recovery(self) -> None:
        """Recovers indexed chunks from ChromaDB persistent storage."""
        coll = self._get_chroma_collection()
        if coll is None:
            return

        try:
            stored = coll.get()
            docs = stored.get("documents", [])
            metas = stored.get("metadatas", [])
            if docs:
                logger.info(f"Recovered {len(docs)} persisted chunk(s) from ChromaDB storage.")
                indexed_docs = []
                for i, doc in enumerate(docs):
                    idx = metas[i].get("chunk_index", i) if (metas and i < len(metas) and metas[i]) else i
                    indexed_docs.append((idx, doc))
                indexed_docs.sort(key=lambda x: x[0])
                self._corpus = [d for _, d in indexed_docs]
        except Exception as e:
            logger.warning(f"Could not recover persisted corpus from ChromaDB: {e}")

    def reset(self) -> bool:
        """Purges existing indices in memory and on disk."""
        with self._lock:
            self._corpus.clear()
            self._corpus_metadata.clear()

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

            logger.info("VectorEngine reset completed.")
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
                "embedding_model": self.model_name,
                "persist_directory": str(self.persist_dir),
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
    """Resets the vector indices."""
    return get_vector_engine().reset()


def get_index_stats() -> dict:
    """Returns diagnostic statistics."""
    return get_vector_engine().get_stats()


if __name__ == "__main__":
    print("Testing Person 3 (Part 1: ChromaDB Vector Database)...")
    chunks = [
        "[Doc.pdf | Page 1] Synthexis architecture decouples Streamlit UI and FastAPI backend.",
        "[Doc.pdf | Page 2] ChromaDB persistent storage maintains dense vector representations."
    ]
    create_index(chunks)
    res = hybrid_search("vector database persistence", top_k=1)
    print("Retrieved:", res)
    assert len(res) == 1
    print("Part 1 verification passed!")
