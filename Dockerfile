# Synthexis Dockerfile - Decoupled Multimodal RAG
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (build-essential, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create persistent storage directories
RUN mkdir -p /app/data/uploads /app/data/chromadb /app/data/extracted_images

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Default command runs the FastAPI backend
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
