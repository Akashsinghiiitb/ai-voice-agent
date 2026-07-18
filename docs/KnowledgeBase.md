# Knowledge Base & Metadata Specification

This document details the database schema and metadata structure used to index health policy chunks.

## Ingestion Record Schema

Every parsed text segment is converted into a structured Knowledge Record before being indexed in ChromaDB. 

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **`record_id`** | String | Unique string hash identifying the document chunk | `kb_4f2a1b_chunk_0` |
| **`title`** | String | Human-readable document or web page name | `PDF: official_uiic_terms.pdf` |
| **`content`** | String | Cleaned and PII-redacted text paragraph | `Maternity coverage has a 9-month waiting period...` |
| **`category`** | String | Policy classification category (used for Reranking Priority) | `Policy PDF` |
| **`source`** | String | Original source filename or website indicator | `official_uiic_terms.pdf` |
| **`page`** | String | PDF page number where text was found (defaults to 1 for web) | `3` |
| **`section`** | String | Section heading or structural title of text | `Coverage & Exclusions` |
| **`url`** | String | Original web URL or download link reference | `https://uiic.co.in/downloads` |
| **`version`** | String | Document version number to track updates | `1.0` |
| **`timestamp`** | String | UTC Ingestion timestamp | `2026-07-17T21:12:00` |

---

## Chunking & Embedding Strategy

1. **Text Splitting:** Uses `RecursiveCharacterTextSplitter` configured with `chunk_size=500` characters and `chunk_overlap=100` characters. This preserves paragraph boundaries and ensures contextual continuity across splits.
2. **Dense Vector Space:** Uses the Sentence Transformers **`all-MiniLM-L6-v2`** model. This translates each chunk into a 384-dimensional vector representation.
3. **Similarity Metric:** ChromaDB indexes vectors using **Cosine Similarity**, calculating relevance scores from 0.0 (completely dissimilar) to 1.0 (exact match).
