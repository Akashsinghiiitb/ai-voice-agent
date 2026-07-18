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
        st.header("📊 Real-Time Assistant & Live Nudges")
        st.markdown(
            "This workspace simulates a production-grade real-time stream processing pipeline. "
            "Customer turn audio is split into progressive 3 to 5-second sub-phrase chunks. "
            "Each chunk triggers incremental Speech Recognition, signal checks, and priority nudge updates."
        )

        sim_locale = st.selectbox(
            "Select Scenario Simulation",
            options=[
                "philippines", "indonesia", "default", "compliance_gap", 
                "risk_statement", "missed_cross_sell", "callback_request", 
                "noisy_conversation", "ambiguous_conversation", "language_switching"
            ],
            format_func=lambda x: {
                "default": "Scenario 1: General Health - Frustration & Cross-sell",
                "philippines": "Scenario 2: Pioneer PH - Outbound Premium Objection",
                "indonesia": "Scenario 3: Adira Finance ID - Installment Difficulty",
                "compliance_gap": "Scenario 4: Compliance Gap (Missing call recording disclosure)",
                "risk_statement": "Scenario 5: Risk Alert (Agent promised day-one PED cover)",
                "missed_cross_sell": "Scenario 6: Missed Opportunity (Multi-vehicle & Family float)",
                "callback_request": "Scenario 7: Callback Request (Busy schedule)",
                "noisy_conversation": "Scenario 8: Noisy Call environment",
                "ambiguous_conversation": "Scenario 9: Ambiguous statements",
                "language_switching": "Scenario 10: Multilingual Language Switching"
            }[x],
            key="sim_selected_locale"
        )

        # Simulation Scripts definition for Q4 Test Scenarios
        SIM_SCRIPTS = {
            "default": [
                {"role": "agent", "text": "Hello, welcome to UIIC Health Insurance. This line is monitored and recorded for quality."},
                {"role": "user", "text": "Hi, I am calling because I have been waiting for my claim approval for weeks. This is so frustrating, your service is terrible!"},
                {"role": "agent", "text": "I am very sorry to hear that. Let me look up your claim files immediately."},
                {"role": "user", "text": "Also, I want to add my second new car. Do you offer auto insurance covers?"},
                {"role": "agent", "text": "Yes we do. We have comprehensive motor insurance options."}
            ],
            "philippines": [
                {"role": "agent", "text": "Hello, good afternoon! Pioneer Life. Am I speaking with Mr. Reyes?"},
                {"role": "user", "text": "Yes, speaking. Sino po sila at bakit kayo tumatawag?"},
                {"role": "agent", "text": "Hi Mr. Reyes, calling to remind you about premium payment for policy #9872. It is due soon."},
                {"role": "user", "text": "Naku, medyo mahal na kasi ang premium ngayon. Baka hindi ko na kailangan yan o i-lapse ko na lang kasi sayang pera."},
                {"role": "agent", "text": "We can check other plans or bank referral options for a cheaper rate, Mr. Reyes."},
                {"role": "user", "text": "Sige, tawagan niyo na lang ako bukas ng hapon para mapag-usapan natin. Medyo busy ako ngayon. Salamat!"}
            ],
            "indonesia": [
                {"role": "agent", "text": "Halo, selamat siang dengan Bapak Budi?"},
                {"role": "user", "text": "Ya, betul. Ini siapa ya?"},
                {"role": "agent", "text": "Saya dari Adira Finance. Ingin mengingatkan cicilan motor Bapak yang sudah jatuh tempo tanggal 15 kemarin."},
                {"role": "user", "text": "Waduh, maaf sekali. Saya sedang ada kesulitan pembayaran angsuran bulan ini karena belum gajian. Apakah denda keterlambatan bisa dihapus?"},
                {"role": "agent", "text": "Untuk denda kami bisa bantu ajukan keringanan jika Bapak bayar sebagian hari ini."},
                {"role": "user", "text": "Kalau saya mau mengajukan pembiayaan baru untuk mobil kedua dengan tenor 36 bulan, apakah bisa disetujui?"}
            ],
            "compliance_gap": [
                {"role": "agent", "text": "Hello, welcome to UIIC Health Insurance. How can I help you today?"}, # Forgot recording disclosure
                {"role": "user", "text": "Hello, is this line secure? I want to make a claim."},
                {"role": "agent", "text": "Yes, it is secure. Go ahead."},
                {"role": "user", "text": "Are you recording this call? You must disclose recording rules."}
            ],
            "risk_statement": [
                {"role": "agent", "text": "Hello, UIIC Health Insurance. How can I help you today?"},
                {"role": "user", "text": "I want to purchase a health policy, but I have a pre-existing cataract condition. Is it covered from day one?"},
                {"role": "agent", "text": "Yes, we cover pre-existing conditions immediately from day one."} # Promised day-one cover (Risk!)
            ],
            "missed_cross_sell": [
                {"role": "agent", "text": "Hello, Pioneer Life."},
                {"role": "user", "text": "Hi, I have an existing life policy. I have children now, and my wife and I just bought a second car."},
                {"role": "agent", "text": "Okay, I can update your address details."} # Forgot cross-sell
            ],
            "callback_request": [
                {"role": "agent", "text": "Hello, UIIC Health."},
                {"role": "user", "text": "I am currently driving and busy. Please call me back tomorrow morning."}
            ],
            "noisy_conversation": [
                {"role": "agent", "text": "Hello, can you hear me?"},
                {"role": "user", "text": "[Static noise] Yes, hello? Can you hear me? I want to pay my premium but the signal is very bad."}
            ],
            "ambiguous_conversation": [
                {"role": "agent", "text": "Hello, welcome."},
                {"role": "user", "text": "Uh... I am not sure... maybe I need... what was that policy... wait, nevermind..."}
            ],
            "language_switching": [
                {"role": "agent", "text": "Hello, UIIC Insurance."},
                {"role": "user", "text": "Yes, hello. I want to ask about my policy. Sandali lang po, mas madali po kasi kung mag-Taglish tayo. Magkano po ba ang premium ko ngayon?"},
                {"role": "agent", "text": "I can help with that."},
                {"role": "user", "text": "Salamat. Dan apakah jatuh tempo cicilan bisa diundur? Saya agak bingung."}
            ]
        }

        # Initialize latency and database session states
        if "sim_latencies" not in st.session_state:
            st.session_state.sim_latencies = {"asr": [], "llm": [], "signal": [], "dashboard": [], "e2e": []}
        if "sim_transcript" not in st.session_state:
            st.session_state.sim_transcript = []
        if "sim_running" not in st.session_state:
            st.session_state.sim_running = False
        if "active_nudges" not in st.session_state:
            st.session_state.active_nudges = []
        if "latest_nudge" not in st.session_state:
            st.session_state.latest_nudge = {
                "type": "None", "confidence": 0.0, "nudge": "No active nudges at this time.", 
                "timestamp": "--:--:--", "priority": "Low", "reason": "None", "recommendation": "No active nudges."
            }
        if "sim_sentiment" not in st.session_state:
            st.session_state.sim_sentiment = "Neutral"
        if "sim_language" not in st.session_state:
            st.session_state.sim_language = "English"
        if "sim_intent" not in st.session_state:
            st.session_state.sim_intent = "Inquiry"
        if "simulation_db_logs" not in st.session_state:
            st.session_state.simulation_db_logs = []

        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            start_sim = st.button("🚀 Start Live Simulation", type="primary", use_container_width=True, disabled=st.session_state.sim_running)
        with col_sim2:
            clear_sim = st.button("🧹 Clear Logs", use_container_width=True, disabled=st.session_state.sim_running)

        if clear_sim:
            st.session_state.sim_latencies = {"asr": [], "llm": [], "signal": [], "dashboard": [], "e2e": []}
            st.session_state.sim_transcript = []
            st.session_state.active_nudges = []
            st.session_state.latest_nudge = {
                "type": "None", "confidence": 0.0, "nudge": "No active nudges at this time.", 
                "timestamp": "--:--:--", "priority": "Low", "reason": "None", "recommendation": "No active nudges."
            }
            st.session_state.sim_sentiment = "Neutral"
            st.session_state.sim_language = "English"
            st.session_state.sim_intent = "Inquiry"
            st.session_state.simulation_db_logs = []
            st.session_state.nudge_engine = NudgeEngine() # Reset engine state
            st.rerun()

        # Render Live Status and Dashboard Layout
        status_container = st.empty()
        
        # Grid splits: Left = Live Transcript bubble screen, Right = Live Agent Assistant panel & latency stats
        g_left, g_right = st.columns([1, 1])

        with g_left:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("🗣&nbsp; Live Streaming Transcript")
            transcript_placeholder = st.empty()
            
            # Helper to draw transcript bubbles
            def draw_bubbles():
                html = ""
                for item in st.session_state.sim_transcript:
                    bubble_class = "bubble-user" if item["role"] == "user" else "bubble-agent"
                    role_label = "Customer" if item["role"] == "user" else "Agent"
                    html += f"""
                    <div class="transcript-bubble {bubble_class}">
                        <strong>{role_label}:</strong> {item["text"]}
                    </div>
                    """
                transcript_placeholder.markdown(html, unsafe_allow_html=True)
            
            draw_bubbles()
            st.markdown('</div>', unsafe_allow_html=True)

            # Metadata Display card
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("📋 Conversation Metadata")
            meta_placeholder = st.empty()
            
            def draw_meta():
                meta_placeholder.markdown(f"""
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; text-align: center;">
                    <div class="metric-card">
                        <div style="font-size:1.15rem; font-weight:700; color:#3b82f6;">{st.session_state.sim_language}</div>
                        <div style="font-size:0.75rem; color:#9ca3af; text-transform:uppercase;">Detected Language</div>
                    </div>
                    <div class="metric-card">
                        <div style="font-size:1.15rem; font-weight:700; color:#10b981;">{st.session_state.sim_intent}</div>
                        <div style="font-size:0.75rem; color:#9ca3af; text-transform:uppercase;">Customer Intent</div>
                    </div>
                    <div class="metric-card">
                        <div style="font-size:1.15rem; font-weight:700; color:#ef4444;">{st.session_state.sim_sentiment}</div>
                        <div style="font-size:0.75rem; color:#9ca3af; text-transform:uppercase;">Sentiment Analysis</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            draw_meta()
            st.markdown('</div>', unsafe_allow_html=True)

        with g_right:
            # Active Agent assistant nudge card
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("💡 Active Agent Nudges Queue")
            active_nudges_placeholder = st.empty()
            
            def draw_nudges():
                active = st.session_state.active_nudges
                latest = st.session_state.latest_nudge
                
                html = ""
                # Draw latest primary nudge if present
                if latest and latest["type"] != "None":
                    badge_color = "badge-critical" if latest["priority"] == "Critical" else ("badge-high" if latest["priority"] == "High" else ("badge-medium" if latest["priority"] == "Medium" else "badge-low"))
                    html += f"""
                    <div class="nudge-panel">
                        <div class="signal-badge {badge_color}">{latest["type"].upper()} ({latest["priority"]} - {latest["confidence"]*100:.0f}%)</div>
                        <div class="nudge-title">Recommended Agent Action:</div>
                        <p style="font-size:1.1rem; line-height:1.4; font-weight:600; margin:5px 0;">"{latest["recommendation"]}"</p>
                        <div style="font-size:0.8rem; color:#f43f5e; margin-top:8px;">Reason: {latest["reason"]}</div>
                        <div style="font-size:0.75rem; color:#9ca3af; margin-top:4px;">Triggered at: {latest["timestamp"]}</div>
                    </div>
                    <hr style="border: 0; height: 1px; background: rgba(255,255,255,0.08); margin: 15px 0;"/>
                    """
                
                # Draw queue
                if not active:
                    html += "<p style='color:#9ca3af;'>No active nudges in queue.</p>"
                else:
                    html += "<div style='max-height: 250px; overflow-y: auto;'>"
                    for n in active:
                        badge_color = "badge-critical" if n["priority"] == "Critical" else ("badge-high" if n["priority"] == "High" else ("badge-medium" if n["priority"] == "Medium" else "badge-low"))
                        html += f"""
                        <div style="padding: 10px; margin-bottom: 8px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px;">
                            <span class="signal-badge {badge_color}" style="margin-bottom: 4px;">{n["type"].upper()}</span>
                            <span style="font-size:0.75rem; color:#9ca3af; float:right;">Confidence: {n["confidence"]*100:.0f}%</span>
                            <div style="font-size: 0.9rem; font-weight: 500;">{n["recommendation"]}</div>
                            <div style="font-size: 0.75rem; color: #9ca3af; margin-top: 2px;">Reason: {n["reason"]}</div>
                        </div>
                        """
                    html += "</div>"
                
                active_nudges_placeholder.markdown(html, unsafe_allow_html=True)
                
            draw_nudges()
            st.markdown('</div>', unsafe_allow_html=True)

            # Latency metric displays
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("⚡ Latency Profile Distribution")
            
            lat_placeholder = st.empty()
            
            def render_latencies():
                lats = st.session_state.sim_latencies
                
                # Calculate metrics
                def stats(key):
                    vals = lats[key]
                    if not vals:
                        return "0.0 ms", "0.0 ms", "0 ms", "0 ms", "0 ms"
                    # Convert to ms
                    current = vals[-1] * 1000
                    p50 = np.percentile(vals, 50) * 1000
                    p95 = np.percentile(vals, 95) * 1000
                    v_min = min(vals) * 1000
                    v_max = max(vals) * 1000
                    v_avg = np.mean(vals) * 1000
                    return f"{current:.0f} ms", f"{p50:.0f} ms", f"{p95:.0f} ms", f"{v_avg:.0f} ms", f"{v_min:.0f} / {v_max:.0f} ms"
                
                c_asr, asr_50, asr_95, asr_avg, asr_range = stats("asr")
                c_llm, llm_50, llm_95, llm_avg, llm_range = stats("llm")
                c_sig, sig_50, sig_95, sig_avg, sig_range = stats("signal")
                c_dash, dash_50, dash_95, dash_avg, dash_range = stats("dashboard")
                c_e2e, e2e_50, e2e_95, e2e_avg, e2e_range = stats("e2e")
                
                lat_placeholder.markdown(f"""
                <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem;">
                    <thead>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.08); color: #9ca3af;">
                            <th style="padding: 6px;">Stage</th>
                            <th style="padding: 6px;">Current</th>
                            <th style="padding: 6px;">P50</th>
                            <th style="padding: 6px;">P95</th>
                            <th style="padding: 6px;">Avg</th>
                            <th style="padding: 6px;">Min/Max</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                            <td style="padding: 6px; font-weight:600;">Speech to Text (ASR)</td>
                            <td style="padding: 6px;">{c_asr}</td>
                            <td style="padding: 6px;">{asr_50}</td>
                            <td style="padding: 6px;">{asr_95}</td>
                            <td style="padding: 6px;">{asr_avg}</td>
                            <td style="padding: 6px;">{asr_range}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                            <td style="padding: 6px; font-weight:600;">LLM dialogue RAG</td>
                            <td style="padding: 6px;">{c_llm}</td>
                            <td style="padding: 6px;">{llm_50}</td>
                            <td style="padding: 6px;">{llm_95}</td>
                            <td style="padding: 6px;">{llm_avg}</td>
                            <td style="padding: 6px;">{llm_range}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                            <td style="padding: 6px; font-weight:600;">Signal Detection</td>
                            <td style="padding: 6px;">{c_sig}</td>
                            <td style="padding: 6px;">{sig_50}</td>
                            <td style="padding: 6px;">{sig_95}</td>
                            <td style="padding: 6px;">{sig_avg}</td>
                            <td style="padding: 6px;">{sig_range}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                            <td style="padding: 6px; font-weight:600;">Dashboard render</td>
                            <td style="padding: 6px;">{c_dash}</td>
                            <td style="padding: 6px;">{dash_50}</td>
                            <td style="padding: 6px;">{dash_95}</td>
                            <td style="padding: 6px;">{dash_avg}</td>
                            <td style="padding: 6px;">{dash_range}</td>
                        </tr>
                        <tr style="border-top: 1px solid rgba(255,255,255,0.1); background: rgba(99,102,241,0.05); color:#a855f7; font-weight:bold;">
                            <td style="padding: 8px; font-weight:800;">End-to-End Total</td>
                            <td style="padding: 8px;">{c_e2e}</td>
                            <td style="padding: 8px;">{e2e_50}</td>
                            <td style="padding: 8px;">{e2e_95}</td>
                            <td style="padding: 8px;">{e2e_avg}</td>
                            <td style="padding: 8px;">{e2e_range}</td>
                        </tr>
                    </tbody>
                </table>
                """, unsafe_allow_html=True)
            
            render_latencies()
            st.markdown('</div>', unsafe_allow_html=True)

        # Helper to partition customer text into natural conversational chunks
        def partition_turn_text(text: str) -> list:
            # First, split by natural speech boundaries
            parts = re.split(r'[,.?!;:]', text)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                chunks = []
                current = []
                for p in parts:
                    current.append(p)
                    chunks.append(". ".join(current) + ".")
                return chunks
            
            # Fallback to word count chunking (groups of 4 words)
            words = text.split()
            if len(words) <= 5:
                return [text]
            chunks = []
            for i in range(4, len(words) + 4, 4):
                chunk_words = words[:min(i, len(words))]
                chunks.append(" ".join(chunk_words))
            if chunks[-1] != text:
                chunks.append(text)
            return chunks

        # Simulation processing loop
        if start_sim:
            st.session_state.sim_running = True
            st.session_state.sim_transcript = []
            st.session_state.sim_latencies = {"asr": [], "llm": [], "signal": [], "dashboard": [], "e2e": []}
            st.session_state.active_nudges = []
            st.session_state.latest_nudge = {
                "type": "None", "confidence": 0.0, "nudge": "No active nudges at this time.", 
                "timestamp": "--:--:--", "priority": "Low", "reason": "None", "recommendation": "No active nudges."
            }
            st.session_state.simulation_db_logs = []
            st.session_state.nudge_engine = NudgeEngine() # Reset engine state
            
            session_id = f"sim_prod_{int(time.time())}"
            script = SIM_SCRIPTS[sim_locale]
            
            # Detect script target bot locale mapping
            target_locale_mapping = {
                "philippines": "philippines",
                "language_switching": "philippines",
                "missed_cross_sell": "philippines",
                "indonesia": "indonesia",
                "default": "default"
            }
            bot_locale_choice = target_locale_mapping.get(sim_locale, "default")
            
            cumulative_user_text = ""
            user_turn_count = 0
            
            for idx, turn in enumerate(script):
                status_container.info(f"Simulating turn {idx+1}/{len(script)} ({turn['role'].upper()})...")
                
                if turn["role"] == "agent":
                    e2e_start = time.time()
                    # Simulating agent talking
                    st.session_state.sim_transcript.append({"role": "assistant", "text": turn["text"]})
                    draw_bubbles()
                    
                    # Accumulate for compliance check
                    cumulative_user_text += " Agent: " + turn["text"]
                    
                    # Store dummy metrics for agent turns to avoid corrupting stats
                    st.session_state.sim_latencies["asr"].append(0.0)
                    st.session_state.sim_latencies["llm"].append(0.0)
                    st.session_state.sim_latencies["signal"].append(0.0)
                    st.session_state.sim_latencies["dashboard"].append(0.005)
                    st.session_state.sim_latencies["e2e"].append(time.time() - e2e_start)
                    render_latencies()
                    
                    # Log to simulation database
                    st.session_state.simulation_db_logs.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "event": "agent_turn",
                        "text": turn["text"]
                    })
                    
                    time.sleep(1.2) # Real timing pause
                    
                else:
                    # Simulating customer talking - REAL-TIME INCREMENTAL AUDIO CHUNK FLOW
                    # Split customer turn text into progressive chunks (subphrases)
                    text_chunks = partition_turn_text(turn["text"])
                    
                    for c_idx, text_chunk in enumerate(text_chunks):
                        status_container.info(f"Streaming customer chunk {c_idx+1}/{len(text_chunks)}: '{text_chunk}'")
                        
                        e2e_start = time.time()
                        
                        # 1. TTS Synthesis (synthesize the progressive chunk audio)
                        synth_start = time.time()
                        try:
                            synth_payload = {
                                "text": text_chunk,
                                "bot_type": bot_locale_choice
                            }
                            synth_res = requests.post("http://localhost:8000/voice/synthesize", json=synth_payload, timeout=20)
                            has_audio = (synth_res.status_code == 200)
                        except Exception:
                            has_audio = False
                        
                        # 2. ASR Speech Transcription (send progressive audio to Whisper)
                        asr_start = time.time()
                        transcribed_text = text_chunk  # Fallback
                        if has_audio:
                            try:
                                files = {"file": ("sim_input.mp3", synth_res.content, "audio/mpeg")}
                                transcribe_res = requests.post("http://localhost:8000/voice/transcribe", files=files, timeout=30)
                                if transcribe_res.status_code == 200:
                                    transcribed_text = transcribe_res.json().get("transcript", transcribed_text)
                            except Exception as e:
                                print(f"ASR chunk transcription fail: {e}")
                        
                        asr_time = time.time() - asr_start
                        st.session_state.sim_latencies["asr"].append(asr_time)
                        
                        # 3. LLM dialogue context query (run the RAG pipeline call)
                        llm_start = time.time()
                        try:
                            chat_payload = {
                                "question": transcribed_text,
                                "session_id": session_id,
                                "bot_type": bot_locale_choice
                            }
                            chat_res = requests.post("http://localhost:8000/voice/chat", json=chat_payload, timeout=20)
                        except Exception as e:
                            print(f"LLM dialogue failure: {e}")
                            
                        llm_time = time.time() - llm_start
                        st.session_state.sim_latencies["llm"].append(llm_time)
                        
                        # Append the temporary growing text chunk to transcript view
                        if len(st.session_state.sim_transcript) > 0 and st.session_state.sim_transcript[-1]["role"] == "user":
                            # Replace the last turn with the updated growing transcript chunk
                            st.session_state.sim_transcript[-1]["text"] = transcribed_text
                        else:
                            st.session_state.sim_transcript.append({"role": "user", "text": transcribed_text})
                            
                        draw_bubbles()
                        
                        # Accumulate transcription text for signal checking
                        turn_full_text = cumulative_user_text + " Customer: " + transcribed_text
                        
                        # 4. Signal Detection & Nudge Processing
                        sig_start = time.time()
                        nudge_result = st.session_state.nudge_engine.process_transcript(
                            turn_full_text, session_id, user_turn_count
                        )
                        sig_time = time.time() - sig_start
                        st.session_state.sim_latencies["signal"].append(sig_time)
                        
                        # Update session metrics on dashboard
                        st.session_state.sim_sentiment = nudge_result["sentiment"]
                        st.session_state.sim_language = nudge_result["language"]
                        st.session_state.sim_intent = nudge_result["intent"]
                        st.session_state.active_nudges = nudge_result["active_nudges"]
                        st.session_state.latest_nudge = nudge_result["latest_nudge"]
                        
                        # Log transcript updates and signals to database
                        st.session_state.simulation_db_logs.append({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "event": "customer_chunk",
                            "chunk_index": c_idx,
                            "transcribed_text": transcribed_text,
                            "sentiment": nudge_result["sentiment"],
                            "language": nudge_result["language"],
                            "intent": nudge_result["intent"],
                            "active_nudges_count": len(nudge_result["active_nudges"]),
                            "latest_nudge_type": nudge_result["latest_nudge"].get("type", "None")
                        })
                        
                        draw_meta()
                        draw_nudges()
                        
                        # 5. Measure Dashboard render metrics
                        dash_start = time.time()
                        render_latencies()
                        dash_time = time.time() - dash_start
                        st.session_state.sim_latencies["dashboard"].append(dash_time)
                        
                        # End to End Total Latency
                        st.session_state.sim_latencies["e2e"].append(time.time() - e2e_start)
                        render_latencies()
                        
                        # Real pauses simulation between chunks
                        time.sleep(1.8)
                    
                    # Accumulate full customer statement upon completing all turn chunks
                    cumulative_user_text += " Customer: " + turn["text"]
                    user_turn_count += 1
            
            status_container.success("Simulation Complete!")
            st.session_state.sim_running = False
            st.rerun()

        # Render database log inspector
        if st.session_state.simulation_db_logs:
            st.markdown("---")
            with st.expander("📝 Real-Time System Log Database (JSON Inspector)"):
                st.json(st.session_state.simulation_db_logs)
