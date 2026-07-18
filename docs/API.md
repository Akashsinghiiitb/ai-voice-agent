# API Documentation

This document describes the FastAPI RAG service endpoints.

## Base URL
When running locally, the server is available at:
`http://localhost:8000`

---

## 1. Ask Question (`POST /ask`)

Resolves a policy query against the vector database, applying source priority guidelines and anti-hallucination limits.

### Request Headers
- `Content-Type: application/json`

### Request Body
```json
{
  "question": "What is the waiting period for cataract surgery?"
}
```

### Response Body (200 OK)
```json
{
  "answer": "Cataract surgery is subject to a 24-month (2 years) waiting period, with a maximum limit of Rs. 20,000 per eye.",
  "confidence": 0.892,
  "source": "official_uiic_terms.pdf",
  "page": "4",
  "url": "https://uiic.co.in/web/downloadforms/downloads",
  "retrieved_chunks": [
    {
      "id": "kb_e2f1a2_chunk_3",
      "content": "Cataract surgery is subject to a 24-month (2 years) waiting period, with a maximum limit of Rs. 20,000 per eye.",
      "metadata": {
        "title": "UIIC Official Health Policy Guidelines",
        "category": "Policy PDF",
        "source": "official_uiic_terms.pdf",
        "page": "4",
        "section": "Coverage & Exclusions",
        "url": "https://uiic.co.in/web/downloadforms/downloads"
      },
      "score": 0.892
    }
  ]
}
```

### Error Responses
- **422 Unprocessable Entity:** Invalid payload format (e.g. missing `question` key).
- **500 Internal Server Error:** Database connection failure or runtime exceptions.
