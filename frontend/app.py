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
            if st.button("■ End Conversation", key="end_voice_convo", use_container_width=True):
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
            "This tab simulates a live call and transcribes streaming audio. It passes the incremental transcript "
            "to the Signal Detection LLM and renders real-time recommendations (Nudges) with comprehensive latency audits."
        )

        sim_locale = st.selectbox(
            "Select Scenario Simulation",
            options=["philippines", "indonesia", "default"],
            format_func=lambda x: {
                "default": "General Health - Frustration, Compliance & Cross-sell Script",
                "philippines": "Pioneer Life PH - Outbound Premium Reminder & Objection Script",
                "indonesia": "Adira Finance ID - Installment & Payment Difficulty Script"
            }[x],
            key="sim_selected_locale"
        )

        # Simulation Scripts definition
        SIM_SCRIPTS = {
            "philippines": [
                {"role": "agent", "text": "Hello, good afternoon! Pioneer Life. Am I speaking with Mr. Reyes?"},
                {"role": "user", "text": "Yes, speaking. Sino po sila at bakit kayo tumatawag?"},
                {"role": "agent", "text": "Hi Mr. Reyes, calling to remind you about premium payment for policy #9872. It is due soon."},
                {"role": "user", "text": "Ah, yes. Mag-lapse na ba ang policy ko? Hindi ko kasi alam kung sino ang beneficiary ko at anong riders ang kasama."},
                {"role": "agent", "text": "Your policy will lapse next week. It has a critical illness rider and your wife is the beneficiary."},
                {"role": "user", "text": "Naku, medyo mahal na kasi ang premium ngayon. Baka hindi ko na kailangan yan o i-lapse ko na lang kasi sayang pera."},
                {"role": "agent", "text": "We can do a bank referral to find a cheaper option, or adjust coverage."},
                {"role": "user", "text": "Sige, tawagan niyo na lang ako bukas ng hapon para mapag-usapan natin. Medyo busy ako ngayon. Salamat!"}
            ],
            "indonesia": [
                {"role": "agent", "text": "Halo, selamat siang dengan Bapak Budi?"},
                {"role": "user", "text": "Ya, betul. Ini siapa ya?"},
                {"role": "agent", "text": "Saya dari Adira Finance. Ingin mengingatkan cicilan motor Bapak yang sudah jatuh tempo tanggal 15 kemarin."},
                {"role": "user", "text": "Waduh, maaf sekali. Saya sedang ada kesulitan pembayaran angsuran bulan ini karena belum gajian. Apakah denda keterlambatan bisa dihapus?"},
                {"role": "agent", "text": "Untuk denda kami bisa bantu ajukan keringanan jika Bapak bayar sebagian hari ini."},
                {"role": "user", "text": "Kalau saya mau mengajukan pembiayaan baru untuk mobil kedua dengan tenor 36 bulan, apakah bisa disetujui?"},
                {"role": "agent", "text": "Bisa diajukan jika angsuran sebelumnya diselesaikan. Kami bisa hubungkan ke petugas supervisor."},
                {"role": "user", "text": "Oke baik, tolong hubungkan saya dengan staf supervisor agar bisa bicara langsung sekarang."}
            ],
            "default": [
                {"role": "agent", "text": "Hello, thank you for calling UIIC Health Insurance. How can I help you today?"},
                {"role": "user", "text": "Hi, I am calling because I have been waiting for my claim approval for weeks. This is so frustrating, your service is terrible!"},
                {"role": "agent", "text": "I am very sorry for the delay. Let me check the claim documents for you."},
                {"role": "user", "text": "Yes, I also want to cover another vehicle, my new car. Do you offer auto insurance or should I go somewhere else?"},
                {"role": "agent", "text": "We do offer motor policies. Let me fetch details."},
                {"role": "user", "text": "Great. Please connect me to a human representative or supervisor to solve this claim delay first."}
            ]
        }

        # Initialize latency store in session state
        if "sim_latencies" not in st.session_state:
            st.session_state.sim_latencies = {"asr": [], "llm": [], "signal": [], "dashboard": [], "e2e": []}
        if "sim_transcript" not in st.session_state:
            st.session_state.sim_transcript = []
        if "sim_nudge" not in st.session_state:
            st.session_state.sim_nudge = {"signal": "None", "confidence": 0.0, "nudge": "No active nudges at this time.", "timestamp": "--:--:--"}
        if "sim_running" not in st.session_state:
            st.session_state.sim_running = False

        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            start_sim = st.button("🚀 Start Live Simulation", type="primary", use_container_width=True, disabled=st.session_state.sim_running)
        with col_sim2:
            clear_sim = st.button("🧹 Clear Logs", use_container_width=True, disabled=st.session_state.sim_running)

        if clear_sim:
            st.session_state.sim_latencies = {"asr": [], "llm": [], "signal": [], "dashboard": [], "e2e": []}
            st.session_state.sim_transcript = []
            st.session_state.sim_nudge = {"signal": "None", "confidence": 0.0, "nudge": "No active nudges at this time.", "timestamp": "--:--:--"}
            st.rerun()

        # Render Live Status and Dashboard Layout
        status_container = st.empty()
        
        # Grid splits: Left = Live Transcript bubble screen, Right = Live Agent Assistant panel & latency stats
        g_left, g_right = st.columns([1, 1])

        with g_left:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("🗣️ Live Conversation Transcript")
            transcript_placeholder = st.empty()
            
            # Render existing transcript
            transcript_html = ""
            for item in st.session_state.sim_transcript:
                bubble_class = "bubble-user" if item["role"] == "user" else "bubble-agent"
                role_label = "Customer" if item["role"] == "user" else "Agent"
                transcript_html += f"""
                <div class="transcript-bubble {bubble_class}">
                    <strong>{role_label}:</strong> {item["text"]}
                </div>
                """
            transcript_placeholder.markdown(transcript_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with g_right:
            # Active Agent assistant nudge card
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("💡 Real-Time Agent Assistant (Nudge Board)")
            
            nudge_placeholder = st.empty()
            sig_name = st.session_state.sim_nudge["signal"]
            confidence = st.session_state.sim_nudge["confidence"]
            nudge_text = st.session_state.sim_nudge["nudge"]
            time_stamp = st.session_state.sim_nudge["timestamp"]
            
            badge_html = f'<div class="signal-badge">{sig_name.upper()} ({confidence*100:.0f}%)</div>' if sig_name != "None" else ""
            nudge_placeholder.markdown(f"""
            <div class="nudge-panel">
                {badge_html}
                <div class="nudge-title">Recommended Agent Action:</div>
                <p style="font-size:1.1rem; line-height:1.4;">"{nudge_text}"</p>
                <div style="font-size:0.8rem; color:#9ca3af; margin-top:10px;">Triggered at: {time_stamp}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Latency metric displays
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("⚡ Latency Profile (P50 & P95)")
            
            lat_placeholder = st.empty()
            
            def render_latencies():
                lats = st.session_state.sim_latencies
                
                # Calculate metrics
                def stats(key):
                    vals = lats[key]
                    if not vals:
                        return "0.0 ms", "0.0 ms"
                    # Convert to ms
                    p50 = np.percentile(vals, 50) * 1000
                    p95 = np.percentile(vals, 95) * 1000
                    return f"{p50:.0f} ms", f"{p95:.0f} ms"
                
                asr_50, asr_95 = stats("asr")
                llm_50, llm_95 = stats("llm")
                sig_50, sig_95 = stats("signal")
                dash_50, dash_95 = stats("dashboard")
                e2e_50, e2e_95 = stats("e2e")
                
                lat_placeholder.markdown(f"""
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                    <div class="metric-card">
                        <div class="metric-value">{asr_50} / {asr_95}</div>
                        <div class="metric-label">ASR (P50/P95)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{llm_50} / {llm_95}</div>
                        <div class="metric-label">LLM RAG (P50/P95)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{sig_50} / {sig_95}</div>
                        <div class="metric-label">Signal Detection (P50/P95)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dash_50} / {dash_95}</div>
                        <div class="metric-label">Dashboard Update (P50/P95)</div>
                    </div>
                </div>
                <div class="metric-card" style="margin-top: 12px; border-color: rgba(236,72,153,0.3)">
                    <div class="metric-value" style="color: #ec4899; font-size: 2.1rem;">{e2e_50} / {e2e_95}</div>
                    <div class="metric-label" style="font-weight:700;">End-to-End Latency (P50 / P95)</div>
                </div>
                """, unsafe_allow_html=True)
            
            render_latencies()
            st.markdown('</div>', unsafe_allow_html=True)

        # Execution of Simulation Loop
        if start_sim:
            st.session_state.sim_running = True
            st.session_state.sim_transcript = []
            st.session_state.sim_latencies = {"asr": [], "llm": [], "signal": [], "dashboard": [], "e2e": []}
            
            session_id = f"sim_{int(time.time())}"
            script = SIM_SCRIPTS[sim_locale]
            
            cumulative_user_text = ""
            user_turn_count = 0
            
            for idx, turn in enumerate(script):
                status_container.info(f"Simulating turn {idx+1}/{len(script)} ({turn['role'].upper()})...")
                
                e2e_start = time.time()
                
                if turn["role"] == "agent":
                    # Simulating agent talking (just append to text transcript)
                    st.session_state.sim_transcript.append({"role": "assistant", "text": turn["text"]})
                    
                    # Update view
                    transcript_html = ""
                    for item in st.session_state.sim_transcript:
                        bubble_class = "bubble-user" if item["role"] == "user" else "bubble-agent"
                        role_label = "Customer" if item["role"] == "user" else "Agent"
                        transcript_html += f"""
                        <div class="transcript-bubble {bubble_class}">
                            <strong>{role_label}:</strong> {item["text"]}
                        </div>
                        """
                    transcript_placeholder.markdown(transcript_html, unsafe_allow_html=True)
                    
                    # Add dummy dashboard update latency (very small)
                    st.session_state.sim_latencies["dashboard"].append(0.005)
                    st.session_state.sim_latencies["asr"].append(0.0)
                    st.session_state.sim_latencies["llm"].append(0.0)
                    st.session_state.sim_latencies["signal"].append(0.0)
                    st.session_state.sim_latencies["e2e"].append(time.time() - e2e_start)
                    
                    render_latencies()
                    time.sleep(1.2) # Simulate agent voice duration
                    
                else:
                    # Simulating customer speaking
                    # 1. TTS Synthesis (Synthesize customer voice on-the-fly to test realistic pipeline)
                    # We synthesize using gTTS via the backend synthesizer
                    synth_start = time.time()
                    try:
                        synth_payload = {
                            "text": turn["text"],
                            "bot_type": sim_locale
                        }
                        synth_res = requests.post("http://localhost:8000/voice/synthesize", json=synth_payload, timeout=20)
                        has_audio = (synth_res.status_code == 200)
                    except Exception:
                        has_audio = False
                    
                    # 2. ASR Transcription (Transcribe audio using Whisper model)
                    asr_start = time.time()
                    transcribed_text = turn["text"]  # Fallback
                    if has_audio:
                        try:
                            files = {"file": ("sim_input.mp3", synth_res.content, "audio/mpeg")}
                            transcribe_res = requests.post("http://localhost:8000/voice/transcribe", files=files, timeout=30)
                            if transcribe_res.status_code == 200:
                                transcribed_text = transcribe_res.json().get("transcript", transcribed_text)
                        except Exception as e:
                            print(f"ASR transcription fail in simulation: {e}")
                    
                    asr_time = time.time() - asr_start
                    st.session_state.sim_latencies["asr"].append(asr_time)
                    
                    # 3. LLM dialogue evaluation (Chat RAG query)
                    llm_start = time.time()
                    try:
                        chat_payload = {
                            "question": transcribed_text,
                            "session_id": session_id,
                            "bot_type": sim_locale
                        }
                        # We trigger backend chat, which resolves history and runs LLM
                        chat_res = requests.post("http://localhost:8000/voice/chat", json=chat_payload, timeout=20)
                    except Exception as e:
                        print(f"LLM dialogue failure in simulation: {e}")
                        
                    llm_time = time.time() - llm_start
                    st.session_state.sim_latencies["llm"].append(llm_time)
                    
                    # Append User text to cumulative transcript
                    st.session_state.sim_transcript.append({"role": "user", "text": transcribed_text})
                    
                    # Accumulate transcript for signal processing
                    cumulative_user_text += " " + transcribed_text
                    
                    # Update transcript bubble UI
                    transcript_html = ""
                    for item in st.session_state.sim_transcript:
                        bubble_class = "bubble-user" if item["role"] == "user" else "bubble-agent"
                        role_label = "Customer" if item["role"] == "user" else "Agent"
                        transcript_html += f"""
                        <div class="transcript-bubble {bubble_class}">
                            <strong>{role_label}:</strong> {item["text"]}
                        </div>
                        """
                    transcript_placeholder.markdown(transcript_html, unsafe_allow_html=True)
                    
                    # 4. Signal Detection and Nudge Generation
                    sig_start = time.time()
                    nudge_result = st.session_state.nudge_engine.process_transcript(
                        cumulative_user_text, session_id, user_turn_count
                    )
                    sig_time = time.time() - sig_start
                    st.session_state.sim_latencies["signal"].append(sig_time)
                    
                    user_turn_count += 1
                    
                    # Update nudge panel state
                    if nudge_result and nudge_result["signal"] != "None":
                        st.session_state.sim_nudge = nudge_result
                        
                        # Render new nudge
                        sig_name = nudge_result["signal"]
                        confidence = nudge_result["confidence"]
                        nudge_text = nudge_result["nudge"]
                        time_stamp = nudge_result["timestamp"]
                        badge_html = f'<div class="signal-badge">{sig_name.upper()} ({confidence*100:.0f}%)</div>'
                        nudge_placeholder.markdown(f"""
                        <div class="nudge-panel">
                            {badge_html}
                            <div class="nudge-title">Recommended Agent Action:</div>
                            <p style="font-size:1.1rem; line-height:1.4;">"{nudge_text}"</p>
                            <div style="font-size:0.8rem; color:#9ca3af; margin-top:10px;">Triggered at: {time_stamp}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 5. Measure Dashboard render time
                    dash_start = time.time()
                    render_latencies()
                    dash_time = time.time() - dash_start
                    st.session_state.sim_latencies["dashboard"].append(dash_time)
                    
                    # E2E Latency calculation
                    st.session_state.sim_latencies["e2e"].append(time.time() - e2e_start)
                    
                    render_latencies() # Re-render to show updated e2e
                    time.sleep(1.0) # Pause between turns
            
            status_container.success("Simulation Complete!")
            st.session_state.sim_running = False
            st.rerun()
