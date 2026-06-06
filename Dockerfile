FROM python:3.10-slim

WORKDIR /app

# System dependencies for sentence-transformers / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

RUN mkdir -p output

EXPOSE 5000

# Build vector index on first start, then launch Flask.
# The HF model cache and qdrant_db are mounted as volumes so they
# survive container restarts — only the first run is slow.
CMD ["bash", "-c", \
     "python run_retrieval_pipeline.py && python app.py"]
