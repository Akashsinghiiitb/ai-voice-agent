# Production-Ready Health Insurance RAG Knowledge Base

This repository contains a modular, production-grade Retrieval-Augmented Generation (RAG) platform tailored for health insurance policies. The system combines domain crawling (`uiic.co.in`), PyMuPDF-based PDF parsing, text cleaning, PII redacting, ChromaDB indexing, FastAPI backend services, and a Streamlit frontend UI.

---

## 🛠️ Tech Stack

- **Backend:** FastAPI (Python 3.11) serving queries and ingestion.
- **Frontend Dashboard:** Streamlit UI for crawl parameters, PDF uploads, logs, and query streams.
- **Vector DB:** ChromaDB for index persistence.
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`) generating 384-dimensional dense vectors.
- **HTML & PDF Parsing:** BeautifulSoup & PyMuPDF (`fitz`).

---

## 📂 Project Directory Structure

- **`crawler/`** — Website crawling engine discovering internal links and PDF download urls.
- **`pdf_parser/`** — PyMuPDF parsing logic reading text, headers, and page references.
- **`utils/`** — Data cleaners correcting whitespace and redacting sensitive PII (SSNs, emails, phones).
- **`vector_store/`** — ChromaDB client management and SentenceTransformers embedding calculations.
- **`ingestion/`** — Pipeline orchestrating crawl -> parse -> clean -> chunk (size 500, overlap 100) -> vector DB.
- **`voice_agent/`** — Speech-to-Text transcription (Whisper), Conversation Manager routing (objections, human escalations), and Text-to-Speech synthesis (gTTS).
- **`backend/`** — FastAPI web router with search priority filtering and new voice API endpoints.
- **`frontend/`** — Streamlit app exposing visual RAG search and a new "🎙️ Voice Insurance Assistant" panel.
- **`tests/`** — Automated unit tests for RAG (`test_rag.py`) and Voice (`test_voice.py`), and evaluation benchmarks (`eval_rag.py`).
- **`demo/`** — Call conversation logs (`call_1_normal.txt`, `call_2_objection.txt`, `call_3_escalation.txt`) showing realistic health insurance call paths.
- **`docs/`** — Comprehensive architectural and system documentation manuals.

---

## 🚀 Setup & Execution Guide

### 1. Install Project Dependencies
```bash
pip install -r requirements.txt
```
*Note: Using Whisper ASR locally requires [FFmpeg](https://ffmpeg.org) installed on your system PATH. If FFmpeg or Whisper is missing, the system will automatically fall back to an offline metadata-based simulation.*

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and insert your OpenAI API Key:
```bash
cp .env.example .env
```

### 3. Run the Evaluation Benchmark (Offline Seeding)
Pre-populate your ChromaDB vector index with standard health policy guidelines and print an accuracy scorecard for 20 query parameters:
```bash
$env:PYTHONPATH="."
python tests/eval_rag.py
```

### 4. Run the Automated Test Suite
Ensure all unit tests compile and verify successfully:
```bash
python -m unittest tests/test_rag.py
python -m unittest tests/test_voice.py
```

### 5. Start the FastAPI Backend Service
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Swagger UI docs will be available at [http://localhost:8000/docs](http://localhost:8000/docs) (exposing `/voice/transcribe`, `/voice/chat`, `/voice/synthesize`).

### 6. Launch the Streamlit Frontend UI
In a separate terminal window:
```bash
streamlit run frontend/app.py
```
Open [http://localhost:8501](http://localhost:8501) and switch to the **🎙️ Voice Insurance Assistant** tab to record your question live using the browser microphone (primary) or upload an audio file (fallback) and listen to synthesized, grounded RAG responses played back automatically.

---

## 📚 Architectural Documentation

Complete specifications are located inside the `docs/` folder:
- [Architecture.md](file:///c:/Users/ap775/OneDrive/Desktop/darwix_assesment/docs/Architecture.md) — System flowcharts and crawler mechanics.
- [FolderStructure.md](file:///c:/Users/ap775/OneDrive/Desktop/darwix_assesment/docs/FolderStructure.md) — Map of module files and responsibilities.
- [Testing.md](file:///c:/Users/ap775/OneDrive/Desktop/darwix_assesment/docs/Testing.md) — Unit test specifications and evaluation lists.
- [KnowledgeBase.md](file:///c:/Users/ap775/OneDrive/Desktop/darwix_assesment/docs/KnowledgeBase.md) — Ingestion fields and metadata schema rules.
- [API.md](file:///c:/Users/ap775/OneDrive/Desktop/darwix_assesment/docs/API.md) — FastAPI input/output specifications.
- [Deployment.md](file:///c:/Users/ap775/OneDrive/Desktop/darwix_assesment/docs/Deployment.md) — Running guides and scaling parameters.
