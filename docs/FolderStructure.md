# Project Folder Structure Guide

Here is a breakdown of the modular folder structure and responsibilities of each file:

## Catalog Map

```
darwix_assesment/
├── requirements.txt         # Package dependency specifications
├── .env.example             # Configuration environment variable layout
├── README.md                # System user manual
├── crawler/
│   ├── __init__.py
│   └── crawler.py           # Domain web crawler (link collection & HTML crawling)
├── pdf_parser/
│   ├── __init__.py
│   └── parser.py            # PyMuPDF text reader & PDF downloader
├── vector_store/
│   ├── __init__.py
│   └── store.py             # ChromaDB persistent indexes and all-MiniLM-L6-v2 embeddings
├── ingestion/
│   ├── __init__.py
│   └── pipeline.py          # Ingestion coordinator (crawl -> clean -> chunk -> index)
├── backend/
│   ├── __init__.py
│   └── main.py              # FastAPI endpoints (/ask query resolver)
├── frontend/
│   └── app.py               # Streamlit web user interface
├── utils/
│   ├── __init__.py
│   └── helpers.py           # Text regex cleanup, PII redactions, HTML parsers
├── tests/
│   ├── test_rag.py          # Unit and integrations testing suite
│   └── eval_rag.py          # Evaluation script running 20 benchmark tests
└── docs/
    ├── Architecture.md      # Mermaid diagrams and flowcharts
    ├── FolderStructure.md   # [This file] Directory documentation
    ├── Testing.md           # Instructions on running evaluations
    ├── KnowledgeBase.md     # Ingestion schema rules
    ├── API.md               # API endpoint specifications
    └── Deployment.md        # Run guides and deployment setup
```

## Folder Responsibilities

- **`crawler/`**  
  *Responsibility:* Scopes target websites to extract raw web pages and isolates downloadable document links. Stays inside the domain boundary.
  
- **`pdf_parser/`**  
  *Responsibility:* Converts PDF binary payloads to clear text, tracks page numbers, and maps document section headings.
  
- **`utils/`**  
  *Responsibility:* Standardizes text outputs and redacts PII elements to ensure data hygiene.
  
- **`vector_store/`**  
  *Responsibility:* Translates text paragraphs to semantic numerical matrices and saves them into ChromaDB storage.
  
- **`ingestion/`**  
  *Responsibility:* Drives the chunking logic (500 size, 100 overlap) and merges metadata variables into unified database records.
  
- **`backend/`**  
  *Responsibility:* Runs the FastAPI API server, serving queries with search priority filters and anti-hallucination guardrails.
  
- **`frontend/`**  
  *Responsibility:* Streamlit web interface exposing crawl commands, uploads, and search bars.
  
- **`tests/`**  
  *Responsibility:* Validates execution accuracy across all code components.
