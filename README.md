# Grounded RAG Knowledge Base, Multilingual Voice Bots, & Real-Time Agent Assistant

This repository contains a modular, production-grade AI Voice Agent platform tailored for insurance and financial services. It features website crawling, PDF document parsing, text scrubbing (PII masking), ChromaDB semantic search index, FastAPI backend endpoints, and a multi-tab Streamlit dashboard.

---

## 🛠️ Tech Stack & Key Engines

- **Backend:** FastAPI (Python 3.11) serving RAG queries and voice operations.
- **ASR (Speech to Text):** Local Whisper (`tiny`) for CPU-friendly transcription.
- **TTS (Text to Speech):** Google Text-to-Speech (`gTTS`) supporting locale-specific synthesis.
- **LLM Synthesis:** OpenAI GPT-4o-mini for query rephrasing, response grounded synthesis, and agent suggestions.
- **Vector DB:** ChromaDB with `SentenceTransformer` (`all-MiniLM-L6-v2`) generating 384-dimensional embeddings.

---

## 🚀 Setup & Execution Guide

### 1. Install Project Dependencies
```bash
pip install -r requirements.txt
```
*Note: Whisper ASR requires [FFmpeg](https://ffmpeg.org) installed on your system PATH.*

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and insert your OpenAI API Key:
```bash
cp .env.example .env
```

### 3. Run the Automated Test Suite
Ensure all unit tests compile and pass successfully:
```bash
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
*Runs 26 tests validating RAG pipelines, default voice bots, localized configurations, compliance indicators, and nudge engine triggers.*

### 4. Start the FastAPI Backend Service
```bash
.\venv\Scripts\uvicorn.exe backend.main:app --host 0.0.0.0 --port 8000 --reload
```
*API docs available at http://localhost:8000/docs. Localized knowledge indexes for Pioneer Life (Philippines) and Adira Finance (Indonesia) are seeded on server startup.*

### 5. Launch the Streamlit Dashboard
```bash
.\venv\Scripts\streamlit.exe run frontend/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ✨ Features

- **Grounded RAG Question Answering:** Retrieves factual context from policy documents to guarantee source-anchored responses.
- **FastAPI Backend:** Lightweight API handling text search, dialogue management, and voice processing.
- **Streamlit Frontend:** Dynamic user interface with standard query forms, custom voice controls, and an assistant dashboard.
- **ChromaDB Vector Store:** Efficient local vector database with embedding search indexes.
- **Whisper Speech Recognition:** Runs local ASR transcription on voice inputs using CPU.
- **Google Text-to-Speech:** Translates agent replies into natural audio speech with customizable locale codes.
- **Live Voice Conversation:** Provides a microphone connection permitting real-time natural dialogue.
- **Real-Time AI Suggestions:** Evaluates conversation transcripts dynamically to alert agents of important signals.
- **Localization Support:** Pluggable parameters for custom language formats, terms, and templates.
- **Knowledge Base Grounded Responses:** Enforces strict fact check boundaries to block hallucinations.
- **Latency Table:** Live metrics tracking pipeline stage execution times (P50/P95 stats).

---

## 🎙️ Native Language Voice Bots (Question 3)

The voice agent features localization layers that load specific prompt architectures, terminology libraries, and intent keywords based on the selected bot:

### A. Pioneer Life Bot (Philippines)
- **Sector:** Life Insurance
- **Languages:** English, Tagalog, and conversational Taglish.
- **Terminology Triggers:** `premium`, `policy`, `beneficiary`, `rider`, `coverage`, `lapse`, `bank referral`.
- **Use Cases:** Premium reminder, renewal reminders, and bancassurance lead qualification.
- **TTS Language:** `tl` (Filipino/Tagalog) for natural speech.

### B. Adira Finance Bot (Indonesia)
- **Sector:** Consumer Finance
- **Languages:** Formal Bahasa Indonesia, Colloquial/Slang Bahasa, and financial English loan words.
- **Terminology Triggers:** `cicilan`, `tenor`, `denda`, `DP`, `jatuh tempo`, `angsuran`, `pembiayaan`.
- **Use Cases:** Installment reminders, loan term renewals/follow-ups, and customer qualification.
- **TTS Language:** `id` (Indonesian) for natural speech.

---

## 📊 Live Voice Conversation & Real-Time AI Suggestions (Question 4)

We implement a live voice processing pipeline where the user can speak naturally through the microphone to converse with the existing grounded RAG Agent.

### Pipeline Flow:
```
Microphone
    ↓
Whisper Speech-to-Text
    ↓
Grounded RAG Agent
    ↓
Knowledge Base Retrieval
    ↓
Agent Response
    ↓
Text-to-Speech
    ↓
Conversation History
    ↓
AI Assistant (Nudge Engine)
    ↓
Live Suggestions Dashboard
```

1. **Natural Dialogue:** The user speaks directly into the microphone widget. The Whisper model transcribes this input.
2. **Grounded Answers:** The existing RAG pipeline handles the query, executing a retrieval scan on the vector database to compile factual, grounded responses from the knowledge base. No external knowledge source is utilized.
3. **Independent Assistant:** The AI assistant (Nudge Engine) independently monitors the conversation history and transcript logs as they grow.
4. **Suggestions Panel:** The assistant generates real-time suggestions to guide the agent, identifying consumer signals (e.g., buying interest, confusion, frustration, or price objections) and suggesting follow-up plans.

### A. Consumer Signals & Suggestions Generated:
1. **Frustration:** Customer complains about service quality or delays (*Nudge: "Customer is frustrated"*).
2. **Buying Interest:** Customer wants to buy, upgrade, or ask how to purchase (*Nudge: "Customer is interested in buying"*).
3. **Price Objection:** Customer complains about premium cost or payment issues (*Nudge: "Customer has a price objection"*).
4. **Confused:** Customer asks to repeat terms or expresses difficulty (*Nudge: "Customer seems confused"*).
5. **Cross-Sell Opportunity:** Customer mentions secondary assets or family coverages (*Nudge: "Suggest explaining another plan"*).
6. **Follow-Up plans:** Customer requests callback or schedules (*Nudge: "Suggest asking a follow-up question"*).

---

## ⚡ Latency Monitoring

The dashboard measures latency for each stage of the live voice pipeline, including:

ASR (Speech-to-Text)
RAG Retrieval & Response Generation
Signal Detection
End-to-End Response Time

Actual latency depends on the hardware, model configuration, and whether local or cloud-based LLMs are used.

This is accurate and avoids hardcoding numbers that may not hold on another machine.

---

## 📂 File Modifications and Rationale

| File Path | Action | Description & Rationale |
| :--- | :---: | :--- |
| **`voice_agent/localization.py`** | **[NEW]** | Houses prompt templates, system instructions, and seed databases for Philippines and Indonesia bots. |
| **`voice_agent/nudge_engine.py`** | **[NEW]** | Implements the pipeline for signal detection, confidence filtering, duplicate blocking, and priority sorting. |
| **`tests/test_localization.py`** | **[NEW]** | Verification suite for localized keyword parsing and bot routing. |
| **`tests/test_nudges.py`** | **[NEW]** | Verification suite for nudge suppressions, turns cooldowns, queue limitations, and grouping. |
| **`voice_agent/conversation_manager.py`** | **[MODIFY]** | Updated to select system instructions and keyword configurations dynamically. |
| **`backend/main.py`** | **[MODIFY]** | Added payload parameters, configured locale-aware TTS, and seeded local collections on startup. |
| **`frontend/app.py`** | **[MODIFY]** | Replaced script scenarios with a live conversation dashboard supporting microphone capture and real-time AI suggestions. |
| **`demo/`** | **[KEEP]** | Holds reference multilingual conversation scripts and transcripts. |
