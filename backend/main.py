import os
import sys

# Ensure project root is in python path for internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import openai
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from vector_store.store import ChromaVectorStore

# Import voice agent modules
from voice_agent.speech_to_text import SpeechToText
from voice_agent.text_to_speech import TextToSpeech
from voice_agent.conversation_manager import ConversationManager
from voice_agent.localization import LOCALIZATION_CONFIGS

app = FastAPI(
    title="Health Insurance Grounded RAG API",
    description="FastAPI service serving grounded health policy query resolutions with strict citations.",
    version="1.0.0"
)

# Initialize vector database persistence
store = ChromaVectorStore()

class QueryPayload(BaseModel):
    question: str = Field(..., description="The user query regarding health insurance rules")

class ChunkResponse(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float

class AskResponse(BaseModel):
    answer: str
    confidence: float
    source: str
    page: str
    url: str
    retrieved_chunks: List[ChunkResponse]

def get_priority_score(category: str) -> int:
    """
    Returns search priority:
    1 Policy PDF
    2 Official Website FAQ
    3 Official Product Page
    4 Brochure
    """
    cat_lower = category.lower()
    if "policy pdf" in cat_lower or "pdf" in cat_lower:
        return 1
    elif "website faq" in cat_lower or "faq" in cat_lower:
        return 2
    elif "product page" in cat_lower or "page" in cat_lower:
        return 3
    elif "brochure" in cat_lower:
        return 4
    return 5

def resolve_grounded_query(question: str, store_instance: ChromaVectorStore, history: list = None, bot_type: str = "default") -> dict:
    """
    Unified RAG retrieval and synthesis helper. Enforces priority rules and thresholds.
    Ensures text queries and voice queries run through identical execution paths.
    Reformulates follow-up queries using session history before searching ChromaDB.
    Supports localization layer parameters for target system prompts.
    """
    search_query = question
    config = LOCALIZATION_CONFIGS.get(bot_type, LOCALIZATION_CONFIGS["default"])
    
    # 1. Query Reformulation step for multi-turn contextual queries
    if history and len(history) > 0:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and not openai_key.startswith("your_"):
            try:
                history_str = "\n".join([f"{item['role'].capitalize()}: {item['text']}" for item in history])
                rephrase_prompt = (
                    "Given the conversation history and the latest user query, "
                    "rephrase the query into a standalone, search-friendly question. "
                    "Only output the rephrased query string itself without explanations or preambles."
                )
                openai.api_key = openai_key
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": rephrase_prompt},
                        {"role": "user", "content": f"History:\n{history_str}\n\nLatest Query: {question}"}
                    ],
                    temperature=0.0
                )
                search_query = response.choices[0].message.content.strip()
                print(f"[RAG Reformulation]: Original: '{question}' -> Standalone: '{search_query}'")
            except Exception as e:
                print("Failed to rephrase query, searching using original question:", e)

    # 2. Fetch top candidate chunks (fetch 8 to allow priority reranking)
    raw_matches = store_instance.query(search_query, limit=8)
    
    # Filter matches to match the specific bot's domain if necessary
    # (e.g. Pioneer life docs contain pioneer_life_terms, Adira contains adira_finance_terms)
    filtered_matches = []
    if bot_type == "philippines":
        filtered_matches = [m for m in raw_matches if "pioneer" in m["metadata"].get("source", "").lower() or "ph_policy" in m["id"]]
    elif bot_type == "indonesia":
        filtered_matches = [m for m in raw_matches if "adira" in m["metadata"].get("source", "").lower() or "id_policy" in m["id"]]
    else:
        # Default health insurance should ignore PH/ID docs
        filtered_matches = [m for m in raw_matches if "pioneer" not in m["metadata"].get("source", "").lower() and "adira" not in m["metadata"].get("source", "").lower()]
        
    # If filtering leaves nothing, fall back to raw matches
    if not filtered_matches:
        filtered_matches = raw_matches

    if not filtered_matches:
        return {
            "answer": "I don't have enough information in the knowledge base.",
            "confidence": 0.0,
            "source": "None",
            "page": "N/A",
            "url": "N/A",
            "retrieved_chunks": []
        }

    # 3. Sort matches based on Search Priority rules
    sorted_matches = sorted(
        filtered_matches,
        key=lambda x: (get_priority_score(x["metadata"].get("category", "General")), -x["score"])
    )

    # Pick the top 5 chunks for LLM context injection
    final_chunks = sorted_matches[:5]
    
    # Check if the best match meets similarity threshold limits (0.65)
    best_score = final_chunks[0]["score"]
    
    # In fallback modes (offline, or no API key, or custom bot) we might be lenient
    # But let's enforce 0.65 as a soft threshold, or fallback gracefully
    if best_score < 0.60:
        if bot_type == "default":
            return {
                "answer": "I don't have enough information in the knowledge base.",
                "confidence": float(best_score),
                "source": "None",
                "page": "N/A",
                "url": "N/A",
                "retrieved_chunks": final_chunks
            }
        else:
            # Localized bots can speak using prompt-based guidelines if RAG score is low
            pass

    # 4. Formulate answer synthesis
    context_str = "\n\n".join([
        f"Source: {c['metadata'].get('source')} (Page {c['metadata'].get('page')})\n"
        f"Category: {c['metadata'].get('category')}\n"
        f"Content: {c['content']}"
        for c in final_chunks
    ])

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and not openai_key.startswith("your_"):
        try:
            openai.api_key = openai_key
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": config["system_prompt"]
                    },
                    {"role": "user", "content": f"Context from Knowledge Base:\n{context_str}\n\nUser Statement: {search_query}"}
                ],
                temperature=0.0
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            print("OpenAI API invocation failed:", e)
            answer = f"Extraction Fallback (API Error): {final_chunks[0]['content'][:200]}..."
    else:
        # Static local context matching synthesis when LLM key is absent
        # For localized bots, synthesize in native languages
        if bot_type == "philippines":
            answer = f"Ayon sa policy guidelines: {final_chunks[0]['content']}..."
        elif bot_type == "indonesia":
            answer = f"Berdasarkan ketentuan polis: {final_chunks[0]['content']}..."
        else:
            answer = f"According to the policy: {final_chunks[0]['content'][:250]}..."

    if "I don't have enough information" in answer and bot_type == "default":
        return {
            "answer": "I don't have enough information in the knowledge base.",
            "confidence": float(best_score),
            "source": "None",
            "page": "N/A",
            "url": "N/A",
            "retrieved_chunks": final_chunks
        }

    # 5. Formulate citation references from the top matching chunk
    top_meta = final_chunks[0]["metadata"]
    
    return {
        "answer": answer,
        "confidence": float(best_score),
        "source": top_meta.get("source", "Unknown"),
        "page": top_meta.get("page", "1"),
        "url": top_meta.get("url", ""),
        "retrieved_chunks": final_chunks
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: QueryPayload):
    """
    Resolves a user question using grounded semantic retrieval, sorting by priority guidelines.
    """
    try:
        res = resolve_grounded_query(payload.question, store)
        # Convert chunk dicts to Pydantic responses
        chunks = []
        for c in res["retrieved_chunks"]:
            score = c.get("score", 0.0)
            chunks.append(ChunkResponse(
                id=c.get("id", "unknown"),
                content=c.get("content", ""),
                metadata=c.get("metadata", {}),
                score=float(score)
            ))
        return AskResponse(
            answer=res["answer"],
            confidence=res["confidence"],
            source=res["source"],
            page=res["page"],
            url=res["url"],
            retrieved_chunks=chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# --- Voice Agent Extension ---

# Initialize Voice Modules with shared store and query resolver injection
stt_model = SpeechToText()
tts_model = TextToSpeech()
conversation_mgr = ConversationManager(store, resolve_grounded_query)

class VoiceChatPayload(BaseModel):
    question: str = Field(..., description="The user statement transcribed from voice input")
    session_id: Optional[str] = Field(None, description="In-memory conversation session tracking ID")
    bot_type: Optional[str] = Field("default", description="Locale type: default, philippines, indonesia")

class VoiceChatResponse(BaseModel):
    answer: str
    confidence: float
    source: str
    page: str
    url: str
    intent: str
    session_id: str
    active: bool

class VoiceSynthesizePayload(BaseModel):
    text: str = Field(..., description="Grounded response text to translate to voice output")
    session_id: Optional[str] = Field(None, description="Optional session tracking ID for latency profiling")
    bot_type: Optional[str] = Field("default", description="Locale type for TTS voice synthesis selection")

@app.post("/voice/transcribe")
async def voice_transcribe(file: UploadFile = File(...)):
    """
    STT Endpoint: Transcribes an uploaded audio file into plain text.
    """
    os.makedirs("./tmp_audio", exist_ok=True)
    temp_path = f"./tmp_audio/{uuid.uuid4()}_{file.filename}"
    
    try:
        # Save temporary uploaded file
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        # Transcribe
        transcript = stt_model.transcribe(temp_path)
        return {"transcript": transcript}
    except ImportError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Clean up local temp files
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.post("/voice/chat", response_model=VoiceChatResponse)
async def voice_chat(payload: VoiceChatPayload):
    """
    Chat Router Endpoint: Evaluates intent, checks objections/escalations, 
    and queries ChromaDB using conversation history memory and localization parameters.
    """
    import time
    start_time = time.time()
    
    try:
        # Run dialog manager passing session id and bot type
        res = conversation_mgr.process_message(payload.question, session_id=payload.session_id, bot_type=payload.bot_type)
        llm_latency = time.time() - start_time
        
        session_id = res["session_id"]
        session = conversation_mgr.sessions.get(session_id)
        if session:
            session["llm_latency"] = llm_latency
            session["latest_query"] = payload.question
            session["latest_intent"] = res["intent"]
            session["latest_source"] = res.get("source", "Unknown")
            session["latest_confidence"] = res.get("confidence", 0.0)
            session["latest_chunks"] = res.get("retrieved_chunks", [])
            session["bot_type"] = payload.bot_type

        # Format retrieved chunks for logging checks
        chunks = res.get("retrieved_chunks", [])
        chunk_texts = [c.get("content", "")[:75] + "..." for c in chunks] if isinstance(chunks, list) else []
        scores = [c.get("score", 0.0) for c in chunks] if isinstance(chunks, list) else []
        
        print("\n================================")
        print(f"VOICE SESSION (CHAT ROUTING) - LOCALE: {payload.bot_type.upper()}")
        print(f"Session ID: {session_id}")
        print(f"Transcript: {payload.question}")
        print(f"Detected Intent: {res['intent'].upper()}")
        print(f"Retrieved Chunks: {chunk_texts}")
        print(f"Similarity Scores: {scores}")
        print(f"Chosen Source: {res.get('source')}")
        print(f"Final Answer: {res.get('answer')}")
        print(f"LLM Latency: {llm_latency*1000:.1f}ms")
        print("Speech Generation Time: Pending synthesis endpoint call...")
        print("================================\n")
        
        return VoiceChatResponse(
            answer=res["answer"],
            confidence=res["confidence"],
            source=res["source"],
            page=res["page"],
            url=res["url"],
            intent=res["intent"],
            session_id=session_id,
            active=res["active"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/synthesize")
async def voice_synthesize(payload: VoiceSynthesizePayload):
    """
    TTS Endpoint: Synthesizes text into an MP3 file using the locale language, returning a FileResponse.
    Tracks and prints consolidated conversation logs.
    """
    import time
    tts_start = time.time()
    os.makedirs("./tmp_audio", exist_ok=True)
    temp_out_path = f"./tmp_audio/tts_out_{uuid.uuid4()}.mp3"
    
    # Determine bot type and correct language code
    bot_type = payload.bot_type or "default"
    if payload.session_id:
        session = conversation_mgr.sessions.get(payload.session_id)
        if session and session.get("bot_type"):
            bot_type = session["bot_type"]
            
    config = LOCALIZATION_CONFIGS.get(bot_type, LOCALIZATION_CONFIGS["default"])
    tts_lang = config.get("tts_lang", "en")
    
    try:
        # Instantiate a locale-aware TTS synthesizer
        local_tts = TextToSpeech(lang=tts_lang)
        local_tts.synthesize(payload.text, temp_out_path)
        
        if not os.path.exists(temp_out_path) or os.path.getsize(temp_out_path) == 0:
            raise HTTPException(status_code=500, detail="TTS synthesis returned an empty file.")
            
        tts_latency = time.time() - tts_start
        
        # Look up session info if session_id is provided
        llm_latency = 0.0
        intent = "UNKNOWN"
        query = "N/A"
        chunks_txt = []
        scores = []
        source = "Unknown"
        
        if payload.session_id:
            session = conversation_mgr.sessions.get(payload.session_id)
            if session:
                llm_latency = session.get("llm_latency", 0.0)
                query = session.get("latest_query", "N/A")
                intent = session.get("latest_intent", "UNKNOWN")
                source = session.get("latest_source", "Unknown")
                
                raw_chunks = session.get("latest_chunks", [])
                chunks_txt = [c.get("content", "")[:75] + "..." for c in raw_chunks] if isinstance(raw_chunks, list) else []
                scores = [c.get("score", 0.0) for c in raw_chunks] if isinstance(raw_chunks, list) else []
        
        total_time = llm_latency + tts_latency
        
        print("\n================================")
        print(f"VOICE SESSION COMPLETE AUDIT - LOCALE: {bot_type.upper()}")
        print(f"* Session ID: {payload.session_id or 'N/A'}")
        print(f"* Transcript: {query}")
        print(f"* Detected Intent: {intent.upper()}")
        print(f"* Retrieved Chunks: {chunks_txt}")
        print(f"* Similarity Scores: {scores}")
        print(f"* Selected Citation: {source}")
        print(f"* LLM Latency: {llm_latency*1000:.1f}ms")
        print(f"* TTS Latency: {tts_latency*1000:.1f}ms")
        print(f"* Total Processing Time: {total_time*1000:.1f}ms")
        print("================================\n")
        
        return FileResponse(
            path=temp_out_path,
            media_type="audio/mpeg",
            filename="response.mp3"
        )
    except ImportError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS compilation failed: {str(e)}")


# --- Startup Seeding of Localized Knowledge Chunks ---
@app.on_event("startup")
async def seed_localized_data():
    """
    Pre-populates ChromaDB on startup with localized FAQ rules for Pioneer Life and Adira Finance.
    """
    try:
        # Check if Pioneer documents exist
        results_ph = store.query("Pioneer Life Insurance Policy Guidelines", limit=1)
        has_ph = results_ph and any("pioneer" in str(r["metadata"].get("source", "")).lower() for r in results_ph)
        
        # Check if Adira documents exist
        results_id = store.query("Adira Finance Terms of Service", limit=1)
        has_id = results_id and any("adira" in str(r["metadata"].get("source", "")).lower() for r in results_id)
        
        if not has_ph:
            print("Seeding Pioneer Life (Philippines) knowledge chunks...")
            ph_docs = LOCALIZATION_CONFIGS["philippines"]["seed_data"]
            store.add_documents([{
                "record_id": f"ph_policy_{idx}",
                "title": d["title"],
                "content": d["content"],
                "category": d["category"],
                "source": d["source"],
                "page": d["page"],
                "section": d["section"],
                "url": d["url"],
                "version": "1.0",
                "timestamp": "2026-07-18"
            } for idx, d in enumerate(ph_docs)])
            
        if not has_id:
            print("Seeding Adira Finance (Indonesia) knowledge chunks...")
            id_docs = LOCALIZATION_CONFIGS["indonesia"]["seed_data"]
            store.add_documents([{
                "record_id": f"id_policy_{idx}",
                "title": d["title"],
                "content": d["content"],
                "category": d["category"],
                "source": d["source"],
                "page": d["page"],
                "section": d["section"],
                "url": d["url"],
                "version": "1.0",
                "timestamp": "2026-07-18"
            } for idx, d in enumerate(id_docs)])
            
        print("Localized DB seeding checks complete.")
    except Exception as e:
        print(f"Warning during localized seeding check: {e}")
