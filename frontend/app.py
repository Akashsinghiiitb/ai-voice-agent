import os
import streamlit as st
from vector_store.store import ChromaVectorStore
from ingestion.pipeline import IngestionPipeline
import requests

# Page layout & styling
st.set_page_config(
    page_title="UIIC Health Insurance RAG Knowledge Base",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core initialization
if "logs" not in st.session_state:
    st.session_state.logs = ["System initialized. Ready to ingest documents."]

def log_message(msg: str):
    st.session_state.logs.append(msg)
    print(msg) # print to stdout as well

# Sidebar parameters
st.sidebar.title("Configuration Panel")
openai_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

st.sidebar.markdown("---")
st.sidebar.subheader("System Information")
st.sidebar.info(
    "This RAG platform queries crawled website contents from UIIC (uiic.co.in) "
    "and uploaded health insurance policy PDFs. It uses SentenceTransformers "
    "(all-MiniLM-L6-v2) and ChromaDB."
)

st.title("🛡️ Health Insurance Grounded RAG Knowledge Base")
st.markdown("Build and query a grounded knowledge base for health insurance terms, policy coverage limits, and waiting periods.")

# Layout splits
col_left, col_right = st.columns([1, 1])

with col_left:
    st.header("1. Populate Knowledge Base")
    
    # URL Input
    web_url = st.text_input("Website URL to crawl", value="https://uiic.co.in")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Health Policy PDFs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    build_kb = st.button("Build Knowledge Base", type="primary")
    
    if build_kb:
        log_message("Starting knowledge base compilation...")
        progress_bar = st.progress(0)
        
        # Save files locally to parse
        local_paths = []
        if uploaded_files:
            os.makedirs("./tmp_uploads", exist_ok=True)
            for f in uploaded_files:
                path = os.path.join("./tmp_uploads", f.name)
                with open(path, "wb") as dest:
                    dest.write(f.getbuffer())
                local_paths.append(path)
                log_message(f"Saved uploaded file: {f.name}")
                
        progress_bar.progress(30)
        
        # Run Ingestion
        try:
            # Instantiate components
            vector_store = ChromaVectorStore()
            pipeline = IngestionPipeline(vector_store)
            
            log_message("Starting crawling & parsing cycles...")
            progress_bar.progress(50)
            
            res = pipeline.run(
                start_url=web_url if web_url else None,
                local_pdf_paths=local_paths if local_paths else None
            )
            
            progress_bar.progress(100)
            log_message(f"Ingestion successful! Details: {res}")
            st.success("Knowledge Base successfully compiled!")
            
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
            log_message(f"Error during execution: {str(e)}")

    # Log Terminal Window
    st.markdown("### Execution Logs")
    log_text = "\n".join(st.session_state.logs)
    st.text_area("Pipeline Console Output", value=log_text, height=200, disabled=True)

with col_right:
    tab_text, tab_voice = st.tabs(["🔍 Search & Ask Queries", "🎙️ Voice Insurance Assistant"])
    
    with tab_text:
        st.header("Search & Ask Queries")
        user_query = st.text_input("Enter your health policy question", placeholder="What is the waiting period for cataract surgery?")
        ask_btn = st.button("Ask Question", type="secondary")
        
        if ask_btn and user_query:
            try:
                with st.spinner("Retrieving facts and synthesizing answers..."):
                    response = None
                    try:
                        res = requests.post("http://localhost:8000/ask", json={"question": user_query}, timeout=10)
                        if res.status_code == 200:
                            response = res.json()
                    except Exception:
                        pass
                    
                    if not response:
                        v_store = ChromaVectorStore()
                        raw_matches = v_store.query(user_query, limit=5)
                        if not raw_matches:
                            response = {
                                "answer": "I don't have enough information in the knowledge base.",
                                "confidence": 0.0,
                                "source": "None",
                                "page": "N/A",
                                "url": "N/A",
                                "retrieved_chunks": []
                            }
                        else:
                            response = {
                                "answer": f"Retrieval match (Local Fallback): {raw_matches[0]['content']}",
                                "confidence": raw_matches[0]["score"],
                                "source": raw_matches[0]["metadata"].get("source", "Local"),
                                "page": raw_matches[0]["metadata"].get("page", "1"),
                                "url": raw_matches[0]["metadata"].get("url", ""),
                                "retrieved_chunks": raw_matches
                            }
                    
                    st.markdown("### Answer")
                    st.info(response["answer"])
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Confidence Score", f"{response['confidence'] * 100:.1f}%")
                    with c2:
                        st.metric("Reference Source", response["source"])
                    with c3:
                        st.metric("Reference Page", response["page"])
                        
                    if response["url"] and response["url"] != "N/A":
                        st.markdown(f"**Document Link:** [{response['source']}]({response['url']})")
                    
                    st.markdown("---")
                    st.markdown("### Grounded Evidence (Retrieved Chunks)")
                    for idx, chunk in enumerate(response["retrieved_chunks"]):
                        score = chunk.get("score", 0.0)
                        meta = chunk.get("metadata", {})
                        with st.expander(f"Evidence Chunk #{idx+1} (Relevance: {score * 100:.1f}%)"):
                            st.write(chunk["content"])
                            st.caption(f"Source: {meta.get('source')} | Page: {meta.get('page')} | Category: {meta.get('category')} | URL: {meta.get('url')}")
                            
            except Exception as e:
                st.error(f"Failed to query knowledge base: {e}")

    with tab_voice:
        st.header("Voice Insurance Assistant")
        st.markdown("Record your voice using the microphone below to query the grounded knowledge base.")

        # Initialize session state variables
        if "voice_active" not in st.session_state:
            st.session_state.voice_active = False
        if "voice_session_id" not in st.session_state:
            st.session_state.voice_session_id = ""
        if "voice_history" not in st.session_state:
            st.session_state.voice_history = []
        if "turn_count" not in st.session_state:
            st.session_state.turn_count = 0
        if "latest_audio_bytes" not in st.session_state:
            st.session_state.latest_audio_bytes = None

        # Control panel buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🎤 Start Conversation", use_container_width=True, type="primary"):
                import time
                st.session_state.voice_active = True
                st.session_state.voice_session_id = f"session_{int(time.time())}"
                st.session_state.voice_history = []
                st.session_state.turn_count = 0
                st.session_state.latest_audio_bytes = None
                st.rerun()
                
        with col_btn2:
            if st.button("■ End Conversation", use_container_width=True):
                st.session_state.voice_active = False
                st.session_state.latest_audio_bytes = None
                st.rerun()

        # Session status display
        if st.session_state.voice_active:
            st.success(f"Conversation Active (ID: {st.session_state.voice_session_id})")
        else:
            st.info("Conversation Ended. Click 'Start Conversation' to begin.")

        audio_file = None
        
        # Render mic input if conversation is active
        if st.session_state.voice_active:
            if hasattr(st, "audio_input"):
                audio_file = st.audio_input(
                    "Speak your question...", 
                    key=f"mic_{st.session_state.turn_count}"
                )
                st.caption("Click the mic to record, speak, and click stop. The system will auto-process and listen again.")
            else:
                st.warning("st.audio_input is not supported in this version. Using file upload fallback.")
                audio_file = st.file_uploader(
                    "Upload WAV or MP3 recording",
                    type=["wav", "mp3", "m4a"],
                    key=f"upload_{st.session_state.turn_count}"
                )

        # Autoplay the latest audio response if it exists
        if st.session_state.voice_active and st.session_state.latest_audio_bytes:
            st.markdown("🗣️ **Playing Agent Response...**")
            try:
                st.audio(st.session_state.latest_audio_bytes, format="audio/mp3", autoplay=True)
            except TypeError:
                st.audio(st.session_state.latest_audio_bytes, format="audio/mp3")

        # Automatically process voice inputs when captured
        if st.session_state.voice_active and audio_file:
            try:
                status_box = st.empty()
                status_box.status("🎙️ Voice captured, preparing transmission...")
                
                # 1. Transcribe Audio (ASR)
                status_box.status("Processing Speech-to-Text Transcription...")
                files = {"file": ("input.wav", audio_file.getvalue(), "audio/wav")}
                transcribe_res = requests.post("http://localhost:8000/voice/transcribe", files=files, timeout=30)
                
                if transcribe_res.status_code != 200:
                    status_box.empty()
                    st.error(f"ASR transcription failed: {transcribe_res.text}")
                    st.session_state.turn_count += 1
                    st.rerun()
                    
                transcript = transcribe_res.json().get("transcript", "").strip()
                
                if not transcript:
                    status_box.empty()
                    st.warning("No speech detected. Please try speaking again.")
                    st.session_state.turn_count += 1
                    st.rerun()
                
                # 2. Run Grounded Conversation Manager (Chat)
                status_box.status("Thinking...")
                chat_payload = {
                    "question": transcript,
                    "session_id": st.session_state.voice_session_id
                }
                chat_res = requests.post("http://localhost:8000/voice/chat", json=chat_payload, timeout=20)
                
                if chat_res.status_code != 200:
                    status_box.empty()
                    st.error(f"Dialogue routing failed: {chat_res.text}")
                    st.session_state.turn_count += 1
                    st.rerun()
                    
                chat_data = chat_res.json()
                
                # Record exchange to session state history
                st.session_state.voice_history.append({"role": "user", "text": transcript})
                st.session_state.voice_history.append({"role": "assistant", "text": chat_data["answer"]})
                
                # Terminate active conversation status if session is marked inactive by agent
                if not chat_data.get("active", True):
                    st.session_state.voice_active = False
                
                # 3. Synthesize Speech Output (TTS)
                status_box.status("Speaking...")
                synth_payload = {
                    "text": chat_data["answer"],
                    "session_id": st.session_state.voice_session_id
                }
                synth_res = requests.post("http://localhost:8000/voice/synthesize", json=synth_payload, timeout=20)
                
                status_box.empty()
                
                if synth_res.status_code == 200:
                    st.session_state.latest_audio_bytes = synth_res.content
                else:
                    st.session_state.latest_audio_bytes = None
                
                # Increment turn count to refresh mic widget keys
                st.session_state.turn_count += 1
                st.rerun()
                
            except Exception as e:
                st.error(f"Voice processing error: {e}")

        # Render conversation history feed
        if len(st.session_state.voice_history) > 0:
            st.markdown("### Conversation History")
            for msg in st.session_state.voice_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["text"])


