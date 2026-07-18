# Testing & Evaluation Guide

This guide explains how to run automated unit tests and execute the 20-question health insurance RAG evaluation suite.

## 1. Running Unit Tests

We use Python's built-in `unittest` library to test all code modules (crawler, cleaner, PII scrubbers, ChromaDB queries, and FastAPI endpoints):

To run the unit tests:
```bash
python -m unittest tests/test_rag.py
```

### What is tested:
- **Text Cleaner:** Verifies that copyright notices, repeated navigation links, and blank lines are correctly stripped.
- **PII Scrubbing:** Verifies that email addresses and phone numbers are redacted with `[MASKED_EMAIL]` and `[MASKED_PHONE]`.
- **HTML Parser:** Verifies heading and paragraph tag isolation.
- **Crawler Boundaries:** Verifies that external links are ignored while internal links are crawled.
- **Vector DB Queries:** Confirms that ChromaDB successfully saves, indexes, and queries mock document chunks.
- **API Endpoint:** Verifies that the FastAPI server starts and resolves `/ask` requests.

---

## 2. Running the RAG Evaluation Benchmark

We created a custom evaluation script in [eval_rag.py](file:///c:/Users/ap775/OneDrive/Desktop/darwix_assesment/tests/eval_rag.py) that tests the accuracy and groundedness of the RAG pipeline using **20 specific health insurance queries**:

To execute the evaluation script:
```bash
# Set Python path to project root
$env:PYTHONPATH="."

# Run the benchmark
python tests/eval_rag.py
```

### Evaluation Criteria:
- **Questions:** Covers 20 key health insurance topics including pre-existing conditions, maternity, claim filing limits, cataract limits, grace periods, IVF exclusions, room rent caps, AYUSH coverage, co-payments, and portability requirements.
- **Seeded Data:** The script automatically seeds ChromaDB with the official policy guidelines to run offline testing successfully.
- **Verification:** For each query, the script evaluates the retrieved chunks and checks if the synthesized response contains key required facts (e.g. "4 years" for pre-existing disease limits, "Rs. 20,000" for cataract caps).
- **Grounded Verification (Pass/Fail):** If the answer matches the expected criteria OR returns the correct hallucination fallback (*"I don't have enough information"*), it outputs **PASS**. If a factual error is returned, it outputs **FAIL**.
