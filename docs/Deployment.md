# Deployment & Execution Guide

This document provides setup instructions, execution steps, and scaling guidelines.

## 1. Quick Start (Local Launch)

Follow these steps to run the RAG API and Streamlit UI locally:

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Copy `.env.example` to `.env` and enter your OpenAI API key:
```bash
cp .env.example .env
```

### Step 3: Run the Ingest/Seeding Script
This pre-populates ChromaDB with health insurance data so you can run queries immediately:
```bash
$env:PYTHONPATH="."
python tests/eval_rag.py
```

### Step 4: Start the FastAPI Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to view the interactive Swagger documentation.

### Step 5: Start the Streamlit Frontend Dashboard
In a separate terminal window:
```bash
streamlit run frontend/app.py
```
Open [http://localhost:8501](http://localhost:8501) to interact with the frontend interface.

---

## 2. Production Scaling Recommendations

When scaling the platform to handle higher traffic volumes:

1. **Migrate ChromaDB to a Managed Vector Database:** While local persistent ChromaDB is excellent for development, production workloads should migrate to a dedicated, high-availability cluster (such as pgvector on managed RDS PostgreSQL, or Qdrant).
2. **Implement Async Crawlers:** For production crawling of large domains, swap the requests-based crawler with an asynchronous framework (like Scrapy or Crawlee) to download pages in parallel.
3. **Decouple Ingestion via Message Queues:** Offload crawled pages and PDF documents to a message broker (like RabbitMQ or Amazon SQS). Worker processes can then handle parsing and embedding tasks asynchronously, preventing API server bottlenecks.
