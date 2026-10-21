# SYNTHEXIS - PROJECT CONTEXT FOR AI AGENTS
## Architecture Overview
This is a decoupled Retrieval-Augmented Generation (RAG) system called 'Synthexis'.
The system allows users to upload PDFs (text and charts) and perform multimodal comparison queries.

- **Frontend:** Streamlit (Dark mode, Glassmorphism UI)
- **Backend:** FastAPI (Microservice connecting modules)
- **Data Ingestion (Person 2):** `extractors.py` (Extracts text and images from PDFs)
- **Vector Engine (Person 3):** `vector_engine.py` (BM25 + Vector hybrid search)
- **LLM Engine (Person 4):** `llm_agent.py` (LLM synthesis and prompt formatting)

## Collaboration Rules & AI Agent Instructions
- **STRICT ISOLATION:** Each team member's AI agent must ONLY edit their assigned `.py` file. Do not attempt to refactor the entire repository.
- **I/O CONTRACTS:** Input and output schemas must be strictly respected to ensure seamless integration via the FastAPI backend.
