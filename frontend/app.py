import os
import sys

# Ensure project root is in python path for internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from vector_store.store import ChromaVectorStore
from ingestion.pipeline import IngestionPipeline
from voice_agent.nudge_engine import NudgeEngine
from voice_agent.localization import LOCALIZATION_CONFIGS
import requests
import time
import uuid
import numpy as np
import re

# Page layout & styling
st.set_page_config(
    page_title="UIIC & Multilingual Grounded Voice Assistants",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling for Dark Mode and Neon accents
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        text-align: left;
    }
    
    .sub-header {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .premium-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    .nudge-panel {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        border: 1px solid rgba(236, 72, 153, 0.2);
        border-left: 6px solid #ec4899;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
    }
    
    .nudge-title {
        color: #ec4899;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 8px;
    }
    
    .metric-card {
        background: rgba(31, 41, 55, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #a855f7;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    .transcript-bubble {
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    
    .bubble-user {
        background-color: #3b82f6;
        color: white;
        margin-left: auto;
    }
    
    .bubble-agent {
        background-color: #1f2937;
        color: #f3f4f6;
        margin-right: auto;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .signal-badge {
        background-color: #ef4444;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    .badge-critical { background-color: #ef4444; }
    .badge-high { background-color: #f97316; }
    .badge-medium { background-color: #eab308; }
    .badge-low { background-color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

# Core initialization
if "logs" not in st.session_state:
    st.session_state.logs = ["System initialized. Ready to ingest documents."]

def log_message(msg: str):
    st.session_state.logs.append(msg)
    print(msg)

# Initialize nudge engine state
if "nudge_engine" not in st.session_state:
    st.session_state.nudge_engine = NudgeEngine()

# Sidebar parameters
st.sidebar.title("Configuration Panel")
openai_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

st.sidebar.markdown("---")
st.sidebar.subheader("System Information")
st.sidebar.info(
    "Multilingual Grounded RAG & Real-Time Nudge System.\n"
    "ASR: Local Whisper (tiny)\n"
    "TTS: Google Text-to-Speech (gTTS)\n"
    "LLM: OpenAI gpt-4o-mini"
)

# Header block
st.markdown('<div class="main-header">🛡️ Multilingual Voice Bots & Real-Time Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Localized agents for Life Insurance and Consumer Finance featuring real-time compliance, compliance signals, and agent nudges.</div>', unsafe_allow_html=True)

# Layout splits
col_left, col_right = st.columns([2, 3])

with col_left:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.header("1. Populate Knowledge Base")
    
    # URL Input
    web_url = st.text_input("Website URL to crawl", value="https://uiic.co.in")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Health/Life/Finance Policy PDFs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    build_kb = st.button("Build Knowledge Base", type="primary", use_container_width=True)
    
    if build_kb:
        log_message("Starting knowledge base compilation...")
        progress_bar = st.progress(0)
        
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
        
        try:
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
    st.text_area("Pipeline Console Output", value=log_text, height=180, disabled=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    tab_text, tab_voice, tab_nudge = st.tabs([
        "🔍 RAG Queries", 
        "🎙️ Localized Voice Bots", 
        "📊 Real-Time Assistant & Nudges"
    ])
    
    # --- RAG Queries Tab ---
    with tab_text:
        st.header("Search & Ask Queries")
        user_query = st.text_input("Enter your policy question", placeholder="Enter your policy Question.")
        ask_btn = st.button("Ask Question", type="primary")
        
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

    # --- Localized Voice Bots Tab ---
    with tab_voice:
        st.header("Multilingual Voice Assistants")
        st.markdown("Select a target locale/sector to converse with a custom voice assistant.")

        # Bot Locale Selector
        bot_locale = st.selectbox(
            "Select Localized Bot",
            options=["default", "philippines", "indonesia"],
            format_func=lambda x: {
                "default": "Default Bot (UIIC Health Insurance - English)",
                "philippines": "Philippines Bot (Pioneer Life - Taglish/Filipino/English)",
                "indonesia": "Indonesia Bot (Adira Finance - Bahasa/Colloquial/Loan words)"
            }[x],
            key="selected_locale"
        )
        
        # Display bot profile card
        locale_conf = LOCALIZATION_CONFIGS[bot_locale]
        st.markdown(f"""
        <div class="premium-card">
            <strong>Locale Profile:</strong> {locale_conf["name"]}<br/>
            <strong>Sector:</strong> {locale_conf["sector"]}<br/>
            <strong>Supported Languages:</strong> {', '.join(locale_conf["languages"])}<br/>
            <strong>Objection Trigger:</strong> <i>{locale_conf["objection_response"]}</i>
        </div>
        """, unsafe_allow_html=True)

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

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🎤 Start Conversation", key="start_voice_convo", use_container_width=True, type="primary"):
                st.session_state.voice_active = True
                st.session_state.voice_session_id = f"session_{int(time.time())}"
                st.session_state.voice_history = []
                st.session_state.turn_count = 0
                st.session_state.latest_audio_bytes = None
                st.rerun()
                
        with col_btn2:
            if st.button("■ End Conversation", key="end_voice_convo", use_container_width=True, type="primary"):
                st.session_state.voice_active = False
                st.session_state.latest_audio_bytes = None
                st.rerun()

        # Session status display
        if st.session_state.voice_active:
            st.success(f"Conversation Active (ID: {st.session_state.voice_session_id})")
        else:
            st.info("Conversation Ended. Click 'Start' to begin.")

        audio_file = None
        if st.session_state.voice_active:
            if hasattr(st, "audio_input"):
                audio_file = st.audio_input(
                    "Speak your question...", 
                    key=f"mic_{st.session_state.turn_count}"
                )
            else:
                audio_file = st.file_uploader(
                    "Upload WAV or MP3 recording",
                    type=["wav", "mp3", "m4a"],
                    key=f"upload_{st.session_state.turn_count}"
                )

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
                    "session_id": st.session_state.voice_session_id,
                    "bot_type": st.session_state.selected_locale
                }
                chat_res = requests.post("http://localhost:8000/voice/chat", json=chat_payload, timeout=20)
                
                if chat_res.status_code != 200:
                    status_box.empty()
                    st.error(f"Dialogue routing failed: {chat_res.text}")
                    st.session_state.turn_count += 1
                    st.rerun()
                    
                chat_data = chat_res.json()
                
                st.session_state.voice_history.append({"role": "user", "text": transcript})
                st.session_state.voice_history.append({"role": "assistant", "text": chat_data["answer"]})
                
                if not chat_data.get("active", True):
                    st.session_state.voice_active = False
                
                # 3. Synthesize Speech Output (TTS)
                status_box.status("Speaking...")
                synth_payload = {
                    "text": chat_data["answer"],
                    "session_id": st.session_state.voice_session_id,
                    "bot_type": st.session_state.selected_locale
                }
                synth_res = requests.post("http://localhost:8000/voice/synthesize", json=synth_payload, timeout=20)
                
                status_box.empty()
                
                if synth_res.status_code == 200:
                    st.session_state.latest_audio_bytes = synth_res.content
                else:
                    st.session_state.latest_audio_bytes = None
                
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

    # --- Real-Time Assistant & Nudges Tab ---
    with tab_nudge:
        st.header("📊 Live Conversation & AI Assistant")
        st.markdown(
            "Speak naturally with the grounded RAG agent over the microphone. "
            "The AI assistant continuously listens, transcribes your voice, retrieves grounding facts, "
            "and generates real-time suggestions (nudges) on the sidebar panel."
        )

        # Initialize session state variables for live conversation
        if "live_active" not in st.session_state:
            st.session_state.live_active = False
        if "live_session_id" not in st.session_state:
            st.session_state.live_session_id = ""
        if "live_history" not in st.session_state:
            st.session_state.live_history = []
        if "live_turn" not in st.session_state:
            st.session_state.live_turn = 0
        if "live_audio_bytes" not in st.session_state:
            st.session_state.live_audio_bytes = None
        if "live_latencies" not in st.session_state:
            st.session_state.live_latencies = {"asr": [], "llm": [], "signal": [], "dashboard": [], "e2e": []}
        if "live_nudges" not in st.session_state:
            st.session_state.live_nudges = []
        if "live_latest_nudge" not in st.session_state:
            st.session_state.live_latest_nudge = {
                "type": "None", "confidence": 0.0, "recommendation": "No active suggestions.", 
                "priority": "Low", "reason": "None", "timestamp": "--:--:--"
            }
        if "live_sentiment" not in st.session_state:
            st.session_state.live_sentiment = "Neutral"
        if "live_intent" not in st.session_state:
            st.session_state.live_intent = "Inquiry"
        if "live_logs" not in st.session_state:
            st.session_state.live_logs = []

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            if st.button("🎤 Start Live Conversation", key="start_live_voice", use_container_width=True, type="primary"):
                st.session_state.live_active = True
                st.session_state.live_session_id = f"live_{int(time.time())}"
                st.session_state.live_history = []
                st.session_state.live_turn = 0
                st.session_state.live_audio_bytes = None
                st.session_state.live_nudges = []
                st.session_state.live_latest_nudge = {
                    "type": "None", "confidence": 0.0, "recommendation": "No active suggestions.", 
                    "priority": "Low", "reason": "None", "timestamp": "--:--:--"
                }
                st.session_state.live_sentiment = "Neutral"
                st.session_state.live_intent = "Inquiry"
                st.session_state.live_logs = []
                st.session_state.live_latencies = {"asr": [], "llm": [], "signal": [], "dashboard": [], "e2e": []}
                st.session_state.nudge_engine = NudgeEngine() # Reset engine state
                st.rerun()
                
        with col_l2:
            if st.button("■ End Live Conversation", key="end_live_voice", use_container_width=True, type="primary"):
                st.session_state.live_active = False
                st.session_state.live_audio_bytes = None
                st.rerun()

        # Session status display
        if st.session_state.live_active:
            st.success(f"Live Mic Session Active (ID: {st.session_state.live_session_id}) - Language: English")
        else:
            st.info("Live Session Ended. Click 'Start Live Conversation' to begin.")

        # Grid splits: Left = Live Transcript & Mic, Right = Live Suggestions & Latency
        g_left, g_right = st.columns([1, 1])

        with g_left:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("🗣️ Live Conversation Transcript")
            
            # Render chat bubble transcripts
            transcript_html = ""
            for item in st.session_state.live_history:
                bubble_class = "bubble-user" if item["role"] == "user" else "bubble-agent"
                role_label = "Customer" if item["role"] == "user" else "Agent"
                transcript_html += f"""
                <div class="transcript-bubble {bubble_class}">
                    <strong>{role_label}:</strong> {item["text"]}
                </div>
                """
            st.markdown(transcript_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Audio Input controls inside left panel
            if st.session_state.live_active:
                st.markdown("### 🎙️ Speak Now")
                audio_file = None
                if hasattr(st, "audio_input"):
                    audio_file = st.audio_input(
                        "Click to speak...", 
                        key=f"live_mic_{st.session_state.live_turn}"
                    )
                else:
                    audio_file = st.file_uploader(
                        "Upload audio recording",
                        type=["wav", "mp3", "m4a"],
                        key=f"live_upload_{st.session_state.live_turn}"
                    )
                
                if audio_file:
                    try:
                        e2e_start = time.time()
                        status_box = st.empty()
                        
                        # 1. Speech Recognition (ASR)
                        status_box.status("Transcribing voice input...")
                        asr_start = time.time()
                        files = {"file": ("input.wav", audio_file.getvalue(), "audio/wav")}
                        transcribe_res = requests.post("http://localhost:8000/voice/transcribe", files=files, timeout=30)
                        
                        if transcribe_res.status_code == 200:
                            transcript_text = transcribe_res.json().get("transcript", "").strip()
                        else:
                            transcript_text = ""
                            
                        asr_time = time.time() - asr_start
                        st.session_state.live_latencies["asr"].append(asr_time)
                        
                        if transcript_text:
                            # 2. Dialogue RAG Query
                            status_box.status("Querying RAG agent...")
                            llm_start = time.time()
                            chat_payload = {
                                "question": transcript_text,
                                "session_id": st.session_state.live_session_id,
                                "bot_type": "default"
                            }
                            chat_res = requests.post("http://localhost:8000/voice/chat", json=chat_payload, timeout=20)
                            agent_answer = chat_res.json().get("answer", "I didn't catch that.")
                            llm_time = time.time() - llm_start
                            st.session_state.live_latencies["llm"].append(llm_time)
                            
                            # Append history
                            st.session_state.live_history.append({"role": "user", "text": transcript_text})
                            st.session_state.live_history.append({"role": "assistant", "text": agent_answer})
                            
                            # 3. Text to Speech synthesis (TTS)
                            status_box.status("Synthesizing voice reply...")
                            synth_payload = {
                                "text": agent_answer,
                                "session_id": st.session_state.live_session_id,
                                "bot_type": "default"
                            }
                            synth_res = requests.post("http://localhost:8000/voice/synthesize", json=synth_payload, timeout=20)
                            if synth_res.status_code == 200:
                                st.session_state.live_audio_bytes = synth_res.content
                            else:
                                st.session_state.live_audio_bytes = None
                                
                            # 4. Continuous Nudge Engine Analysis
                            status_box.status("Analyzing conversation signals...")
                            sig_start = time.time()
                            
                            # Build growing transcript context
                            full_transcript_str = ""
                            for h_item in st.session_state.live_history:
                                prefix = "Customer: " if h_item["role"] == "user" else "Agent: "
                                full_transcript_str += prefix + h_item["text"] + " "
                                
                            nudge_result = st.session_state.nudge_engine.process_transcript(
                                full_transcript_str, st.session_state.live_session_id, st.session_state.live_turn
                            )
                            sig_time = time.time() - sig_start
                            st.session_state.live_latencies["signal"].append(sig_time)
                            
                            # Update nudge values
                            st.session_state.live_nudges = nudge_result["active_nudges"]
                            st.session_state.live_latest_nudge = nudge_result["latest_nudge"]
                            st.session_state.live_sentiment = nudge_result["sentiment"]
                            st.session_state.live_intent = nudge_result["intent"]
                            
                            # Log events
                            st.session_state.live_logs.append({
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "turn": st.session_state.live_turn,
                                "customer_text": transcript_text,
                                "agent_text": agent_answer,
                                "nudges": len(nudge_result["active_nudges"]),
                                "latest_nudge": nudge_result["latest_nudge"].get("recommendation", "None")
                            })
                            
                            # Measure Display Render
                            dash_start = time.time()
                            dash_time = time.time() - dash_start
                            st.session_state.live_latencies["dashboard"].append(dash_time)
                            
                            # Total E2E Latency
                            st.session_state.live_latencies["e2e"].append(time.time() - e2e_start)
                            
                        status_box.empty()
                        st.session_state.live_turn += 1
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing audio turn: {e}")

            if st.session_state.live_audio_bytes:
                st.markdown("🔊 **Playing Agent Response...**")
                try:
                    st.audio(st.session_state.live_audio_bytes, format="audio/mp3", autoplay=True)
                except TypeError:
                    st.audio(st.session_state.live_audio_bytes, format="audio/mp3")

        with g_right:
            # Active suggestions card
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("💡 Real-Time AI Suggestions (Nudges)")
            
            latest = st.session_state.live_latest_nudge
            active_list = st.session_state.live_nudges
            
            html = ""
            if latest and latest["type"] != "None":
                badge_color = "badge-critical" if latest.get("priority") == "Critical" else ("badge-high" if latest.get("priority") == "High" else ("badge-medium" if latest.get("priority") == "Medium" else "badge-low"))
                html += f"""
                <div class="nudge-panel">
                    <div class="signal-badge {badge_color}">{latest.get("priority", "Medium").upper()} ({latest.get("confidence", 0)*100:.0f}%)</div>
                    <div class="nudge-title">AI Assistant Suggestion:</div>
                    <p style="font-size:1.15rem; line-height:1.4; font-weight:600; margin:5px 0;">"{latest.get("recommendation")}"</p>
                    <div style="font-size:0.8rem; color:#f43f5e; margin-top:8px;">Reason: {latest.get("reason")}</div>
                    <div style="font-size:0.75rem; color:#9ca3af; margin-top:4px;">Triggered at: {latest.get("timestamp")}</div>
                </div>
                <hr style="border: 0; height: 1px; background: rgba(255,255,255,0.08); margin: 15px 0;"/>
                """
            
            if not active_list:
                html += "<p style='color:#9ca3af;'>No active suggestions. Speak through the microphone to generate nudges.</p>"
            else:
                html += "<div style='max-height: 250px; overflow-y: auto;'>"
                for n in active_list:
                    badge_color = "badge-critical" if n["priority"] == "Critical" else ("badge-high" if n["priority"] == "High" else ("badge-medium" if n["priority"] == "Medium" else "badge-low"))
                    html += f"""
                    <div style="padding: 10px; margin-bottom: 8px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px;">
                        <span class="signal-badge {badge_color}" style="margin-bottom: 4px;">{n["priority"].upper()}</span>
                        <span style="font-size:0.75rem; color:#9ca3af; float:right;">Confidence: {n["confidence"]*100:.0f}%</span>
                        <div style="font-size: 0.95rem; font-weight: 600;">{n["recommendation"]}</div>
                        <div style="font-size: 0.75rem; color: #9ca3af; margin-top: 2px;">Reason: {n["reason"]}</div>
                    </div>
                    """
                html += "</div>"
            
            st.markdown(html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Metadata Display
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("📋 Session Indicators")
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; text-align: center;">
                <div class="metric-card">
                    <div style="font-size:1.15rem; font-weight:700; color:#10b981;">{st.session_state.live_intent}</div>
                    <div style="font-size:0.75rem; color:#9ca3af; text-transform:uppercase;">Customer Intent</div>
                </div>
                <div class="metric-card">
                    <div style="font-size:1.15rem; font-weight:700; color:#ef4444;">{st.session_state.live_sentiment}</div>
                    <div style="font-size:0.75rem; color:#9ca3af; text-transform:uppercase;">Sentiment</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Latency statistics
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("⚡ Latency Profile")
            
            # Latency calculations
            def get_lat_stats(key):
                vals = st.session_state.live_latencies.get(key, [])
                if not vals:
                    return "0 ms", "0 ms", "0 ms"
                p50 = np.percentile(vals, 50) * 1000
                p95 = np.percentile(vals, 95) * 1000
                v_avg = np.mean(vals) * 1000
                return f"{p50:.0f} ms", f"{p95:.0f} ms", f"{v_avg:.0f} ms"
            
            asr_p50, asr_p95, asr_avg = get_lat_stats("asr")
            llm_p50, llm_p95, llm_avg = get_lat_stats("llm")
            sig_p50, sig_p95, sig_avg = get_lat_stats("signal")
            e2e_p50, e2e_p95, e2e_avg = get_lat_stats("e2e")
            
            st.markdown(f"""
            <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem;">
                <thead>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.08); color: #9ca3af;">
                        <th style="padding: 6px;">Pipeline Stage</th>
                        <th style="padding: 6px;">P50</th>
                        <th style="padding: 6px;">P95</th>
                        <th style="padding: 6px;">Average</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                        <td style="padding: 6px; font-weight:600;">ASR (Whisper)</td>
                        <td>{asr_p50}</td>
                        <td>{asr_p95}</td>
                        <td>{asr_avg}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                        <td style="padding: 6px; font-weight:600;">LLM RAG</td>
                        <td>{llm_p50}</td>
                        <td>{llm_p95}</td>
                        <td>{llm_avg}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                        <td style="padding: 6px; font-weight:600;">Signal Detection</td>
                        <td>{sig_p50}</td>
                        <td>{sig_p95}</td>
                        <td>{sig_avg}</td>
                    </tr>
                    <tr style="border-top: 1px solid rgba(255,255,255,0.1); background: rgba(99,102,241,0.05); color:#a855f7; font-weight:bold;">
                        <td style="padding: 8px; font-weight:800;">End-to-End Total</td>
                        <td>{e2e_p50}</td>
                        <td>{e2e_p95}</td>
                        <td>{e2e_avg}</td>
                    </tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.live_logs:
            st.markdown("---")
            with st.expander("📝 Real-Time System Log Database"):
                st.json(st.session_state.live_logs)
