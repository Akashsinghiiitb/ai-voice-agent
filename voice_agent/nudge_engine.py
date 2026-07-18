# voice_agent/nudge_engine.py

import os
import re
import time
import json
import openai

class NudgeEngine:
    """
    Analyzes conversation transcripts in real-time to detect consumer signals,
    applying confidence thresholds, duplicate suppression, cooldown rules, and topic grouping.
    Provides an LLM-based analyzer and an offline rule-based fallback.
    """
    def __init__(self, confidence_threshold: float = 0.65, cooldown_seconds: float = 10.0, cooldown_turns: int = 2):
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.cooldown_turns = cooldown_turns
        
        # State tracking: session_id -> { signal_name -> { "timestamp": float, "turn": int, "text": str } }
        self.session_states = {}

        # Signal priority hierarchy (lower number = higher priority)
        self.signal_priorities = {
            "compliance_issue": 1,
            "customer_frustration": 2,
            "payment_difficulty": 3,
            "callback_request": 4,
            "buying_signal": 5,
            "missed_cross_sell": 6,
            "intent_change": 7
        }

        # Local rule-based dictionary for offline fallback
        # Key patterns to match against the transcript text
        self.fallback_rules = [
            {
                "signal": "customer_frustration",
                "patterns": [
                    r"\b(mahal|expensive|waste of money|sayang pera|terrible|bad service|angry|annoyed|frustrated|unfair|denda|lambat|sucks|hate|tidak adil|tidak mau bayar)\b",
                    r"\b(so expensive|too high|costly|lousy|slow|stupid|worst)\b"
                ],
                "nudge": "Recommend empathy statement and active listening. Offer to connect to a supervisor.",
                "confidence": 0.85
            },
            {
                "signal": "buying_signal",
                "patterns": [
                    r"\b(want to buy|sign up|interested|upgrade|avail|purchase|kumuha|beli|saya mau|tertarik|ingin daftar)\b",
                    r"\b(how much is the premium|how to pay|send link|bisa bantu daftar|mau beli)\b"
                ],
                "nudge": "Buying interest detected! Present payment options or explain how to bind coverage immediately.",
                "confidence": 0.90
            },
            {
                "signal": "compliance_issue",
                "patterns": [
                    r"\b(privacy|legal|terms|record|taping|pribado|hukum|syarat|ketentuan|regulasi|compliance|policy details)\b",
                    r"\b(are you recording|not disclose|lawyer|rekanan|resmi)\b"
                ],
                "nudge": "Compliance note: State the call recording disclosure and read policy terms clearly.",
                "confidence": 0.80
            },
            {
                "signal": "missed_cross_sell",
                "patterns": [
                    r"\b(another car|second vehicle|family member|wife|child|motorcycle|kotse|sasakyan|motor|istri|anak|suami|mobil kedua|kendaraan lain)\b"
                ],
                "nudge": "Suggest Multi-Vehicle / Family Float discount offer.",
                "confidence": 0.75
            },
            {
                "signal": "payment_difficulty",
                "patterns": [
                    r"\b(cannot pay|no money|insufficient|next week|later|installment|cicilan|tidak ada uang|bokek|belum gajian|minta tempo|nyicil|tunda)\b",
                    r"\b(susah bayar|telat bayar|tidak sanggup)\b"
                ],
                "nudge": "Suggest callback, payment assistance program, or installment schedule.",
                "confidence": 0.85
            },
            {
                "signal": "callback_request",
                "patterns": [
                    r"\b(call me back|later|busy|tomorrow|tawagan|telepon nanti|telepon kembali|hubungi nanti|sibuk|sedang rapat)\b"
                ],
                "nudge": "Acknowledge immediately, check preferred time, and offer callback.",
                "confidence": 0.80
            },
            {
                "signal": "intent_change",
                "patterns": [
                    r"\b(change my mind|actually|instead|wait|pala|ternyata|eh)\b",
                    r"\b(sebentar|tunggu)\b"
                ],
                "nudge": "Intent shift detected: pivot conversation to address the new customer request.",
                "confidence": 0.70
            }
        ]

    def _get_or_create_state(self, session_id: str) -> dict:
        if session_id not in self.session_states:
            self.session_states[session_id] = {}
        return self.session_states[session_id]

    def run_llm_detection(self, transcript: str) -> list:
        """
        Sends transcript to OpenAI for signal detection and nudge recommendation.
        """
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key.startswith("your_"):
            return []
            
        system_prompt = (
            "You are a real-time speech analytics engine processing customer phone call transcripts. "
            "Your objective is to analyze the incremental transcript and detect any of the following signals:\n"
            "1. customer_frustration: customer is angry, complaining about price or service.\n"
            "2. buying_signal: customer wants to buy, upgrade, or asks how to pay.\n"
            "3. compliance_issue: call lacks recording disclosures, mentions privacy, legal, or terms.\n"
            "4. missed_cross_sell: customer mentions another vehicle, family member, or property but the agent hasn't pitched cover.\n"
            "5. payment_difficulty: customer has no money, needs installments, late due-date waiver.\n"
            "6. callback_request: customer is busy, wants to be called back later.\n"
            "7. intent_change: customer changes topic (e.g. shifts from questions to complaining or buying).\n\n"
            "Format your response as a valid JSON object matching this schema:\n"
            "{\n"
            "  \"detected_signals\": [\n"
            "    {\n"
            "      \"signal\": \"signal_name_here\",\n"
            "      \"confidence\": 0.85,\n"
            "      \"nudge\": \"Brief actionable recommendation for the agent\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Only use the exact keys listed above. If no signals are present, return an empty array for \"detected_signals\"."
        )

        try:
            openai.api_key = openai_key
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Transcript:\n{transcript}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("detected_signals", [])
        except Exception as e:
            print(f"[NudgeEngine LLM Error]: {e}. Falling back to rule-based engine.")
            return []

    def run_fallback_detection(self, transcript: str) -> list:
        """
        Regex keyword pattern match fallback when LLM is unavailable or errors out.
        """
        detected = []
        text_lower = transcript.lower()
        
        for rule in self.fallback_rules:
            matched = False
            for pat in rule["patterns"]:
                if re.search(pat, text_lower):
                    matched = True
                    break
            
            if matched:
                detected.append({
                    "signal": rule["signal"],
                    "confidence": rule["confidence"],
                    "nudge": rule["nudge"]
                })
        return detected

    def process_transcript(self, transcript: str, session_id: str, turn_index: int) -> dict:
        """
        Analyzes the transcript, runs signal detection, filters outcomes, and groups them.
        Returns the top selected nudge and its details, or None.
        """
        start_time = time.time()
        
        # 1. Run detection (LLM first, fallback to regex)
        signals = self.run_llm_detection(transcript)
        if not signals:
            signals = self.run_fallback_detection(transcript)
            
        llm_latency = time.time() - start_time
        
        state = self._get_or_create_state(session_id)
        current_time = time.time()
        
        valid_signals = []
        
        for sig in signals:
            name = sig["signal"]
            conf = sig["confidence"]
            nudge_text = sig["nudge"]
            
            # A. Confidence Threshold check
            if conf < self.confidence_threshold:
                continue
                
            # Retrieve historical state for this signal
            sig_state = state.get(name, {})
            
            # B. Duplicate Suppression check
            if sig_state.get("text") == nudge_text:
                continue
                
            # C. Cooldown Period checks
            last_time = sig_state.get("timestamp", 0.0)
            last_turn = sig_state.get("turn", -1)
            
            time_elapsed = current_time - last_time
            turn_elapsed = turn_index - last_turn
            
            # We ONLY apply cooldown checks if the signal was previously triggered (last_time > 0)
            if last_time > 0.0:
                if time_elapsed < self.cooldown_seconds or turn_elapsed < self.cooldown_turns:
                    continue
                
            # Signal is valid and eligible
            valid_signals.append({
                "signal": name,
                "confidence": conf,
                "nudge": nudge_text
            })
            
        if not valid_signals:
            return {
                "signal": "None",
                "confidence": 0.0,
                "nudge": "No active nudges at this time.",
                "latency": llm_latency,
                "timestamp": time.strftime("%H:%M:%S")
            }
            
        # D. Topic Grouping & Priority sorting
        valid_signals.sort(key=lambda x: self.signal_priorities.get(x["signal"], 99))
        chosen = valid_signals[0]
        
        # Update state for the chosen signal to trigger cooldown next time
        state[chosen["signal"]] = {
            "timestamp": current_time,
            "turn": turn_index,
            "text": chosen["nudge"]
        }
        
        return {
            "signal": chosen["signal"],
            "confidence": chosen["confidence"],
            "nudge": chosen["nudge"],
            "latency": llm_latency,
            "timestamp": time.strftime("%H:%M:%S")
        }
