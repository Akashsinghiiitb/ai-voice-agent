import os
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

def resolve_grounded_query(question: str, store_instance: ChromaVectorStore, history: list = None) -> dict:
    """
    Unified RAG retrieval and synthesis helper. Enforces priority rules and thresholds.
    Ensures text queries and voice queries run through identical execution paths.
    Reformulates follow-up queries using session history before searching ChromaDB.
    """
    search_query = question
    
    # 1. Query Reformulation step for multi-turn contextual queries (e.g. "What about knee replacement?")
    if history and len(history) > 0:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and not openai_key.startswith("your_"):
            try:
                history_str = "\n".join([f"{item['role'].capitalize()}: {item['text']}" for item in history])
                rephrase_prompt = (
                    "Given the conversation history and the latest user query, "
                    "rephrase the query into a standalone, search-friendly health insurance question. "
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
    
    if not raw_matches:
        return {
            "answer": "I don't have enough information in the knowledge base.",
            "confidence": 0.0,
            "source": "None",
            "page": "N/A",
            "url": "N/A",
            "retrieved_chunks": []
        }

    # 3. Sort matches based on Search Priority rules
    # Sort key: primary = get_priority_score (lower is better), secondary = vector distance score (higher is better)
    sorted_matches = sorted(
        raw_matches,
        key=lambda x: (get_priority_score(x["metadata"].get("category", "General")), -x["score"])
    )

    # Pick the top 5 chunks for LLM context injection
    final_chunks = sorted_matches[:5]
    
    # Check if the best match meets similarity threshold limits
    best_score = final_chunks[0]["score"]
    if best_score < 0.65:
        # Fallback to prevent hallucination
        return {
            "answer": "I don't have enough information in the knowledge base.",
            "confidence": float(best_score),
            "source": "None",
            "page": "N/A",
            "url": "N/A",
            "retrieved_chunks": final_chunks
        }

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
                        "content": (
                            "You are a strict, grounded Health Insurance assistant. "
                            "Synthesize an answer for the user query based ONLY on the provided context. "
                            "If the context does not contain enough information to resolve the question, "
                            "reply exactly with: 'I don't have enough information in the knowledge base.'\n"
                            "Do not make assumptions, do not use outside knowledge, and do not hallucinate."
                        )
                    },
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {search_query}"}
                ],
                temperature=0.0
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            print("OpenAI API invocation failed:", e)
            answer = f"Extraction Fallback (API Error): {final_chunks[0]['content'][:200]}..."
    else:
        # Static local context matching synthesis when LLM key is absent
        answer = f"According to the policy: {final_chunks[0]['content'][:250]}..."

    if "I don't have enough information" in answer:
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
            # Ensure safety of float score conversion
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
    and queries ChromaDB for normal health insurance topics using conversation history memory.
    """
    import time
    start_time = time.time()
    
    try:
        # Run dialog manager passing session id to load/save conversation history
        res = conversation_mgr.process_message(payload.question, session_id=payload.session_id)
        llm_latency = time.time() - start_time
        
        # Save temporary latency details in session data to allow consolidated logs during tts synthesis
        session_id = res["session_id"]
        session = conversation_mgr.sessions.get(session_id)
        if session:
            session["llm_latency"] = llm_latency
            session["latest_query"] = payload.question
            session["latest_intent"] = res["intent"]
            session["latest_source"] = res.get("source", "Unknown")
            session["latest_confidence"] = res.get("confidence", 0.0)
            session["latest_chunks"] = res.get("retrieved_chunks", [])

        # Format retrieved chunks for logging checks
        chunks = res.get("retrieved_chunks", [])
        chunk_texts = [c.get("content", "")[:75] + "..." for c in chunks] if isinstance(chunks, list) else []
        scores = [c.get("score", 0.0) for c in chunks] if isinstance(chunks, list) else []
        
        print("\n================================")
        print("VOICE SESSION (CHAT ROUTING)")
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
    TTS Endpoint: Synthesizes text into an MP3 file, returning it as a FileResponse.
    Tracks and prints consolidated multi-turn conversation logs upon completion.
    """
    import time
    tts_start = time.time()
    os.makedirs("./tmp_audio", exist_ok=True)
    temp_out_path = f"./tmp_audio/tts_out_{uuid.uuid4()}.mp3"
    
    try:
        tts_model.synthesize(payload.text, temp_out_path)
        if not os.path.exists(temp_out_path) or os.path.getsize(temp_out_path) == 0:
            raise HTTPException(status_code=500, detail="TTS synthesis returned an empty file.")
            
        tts_latency = time.time() - tts_start
        
        # Look up session info if session_id is provided to construct consolidated log outputs
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
        
        # Print mandatory assessment session log
        print("\n================================")
        print("VOICE SESSION COMPLETE AUDIT")
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


