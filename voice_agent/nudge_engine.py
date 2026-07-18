# voice_agent/nudge_engine.py

import os
import re
import time
import json
import openai

class NudgeEngine:
    """
    Analyzes conversation transcripts in real-time to detect consumer signals,
    applying confidence thresholds, duplicate suppression, cooldown rules,
    expiry timers, priority sorting, and topic grouping.
    Provides an LLM-based analyzer and an offline rule-based fallback.
    """
    def __init__(self, confidence_threshold: float = 0.65, cooldown_seconds: float = 10.0, cooldown_turns: int = 2, max_active_nudges: int = 5):
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.cooldown_turns = cooldown_turns
        self.max_active_nudges = max_active_nudges
        
        # State tracking: session_id -> { 
        #   "active_nudges": [nudge_dict, ...], 
        #   "cooldowns": { signal_name: timestamp },
        #   "last_turns": { signal_name: turn_index },
        #   "sentiment": str,
        #   "language": str,
        #   "intent": str
        # }
        self.session_states = {}

        # Signal priority hierarchy (lower number = higher priority)
        self.signal_priorities = {
            "compliance_issue": 1,
            "risk_statement": 2,
            "escalation_requirement": 3,
            "customer_frustration": 4,
            "payment_difficulty": 5,
            "buying_signal": 6,
            "callback_request": 7,
            "intent_change": 8,
            "question_repetition": 9,
            "missed_cross_sell": 10,
            "language_change": 11
        }

        # Local rule-based dictionary for offline fallback
        self.fallback_rules = [
            {
                "signal": "compliance_issue",
                "patterns": [
                    r"\b(record|taping|recording|monitored|privacy|hukum|syarat|ketentuan|regulasi|compliance)\b",
                    r"\b(not disclose|lawyer|rekanan|resmi|not allowed|illegal)\b"
                ],
                "nudge": "Compliance note: State the call recording disclosure and read policy terms clearly.",
                "priority": "Critical",
                "reason": "Customer mentioned legal, privacy, or recording concerns.",
                "confidence": 0.85
            },
            {
                "signal": "risk_statement",
                "patterns": [
                    r"\b(day one|from day 1|immediately covered|pre-existing disease immediate|no waiting period|tidak ada masa tunggu)\b"
                ],
                "nudge": "Risk Alert: Clarify standard 48-month waiting period for pre-existing conditions. Do not promise day-one cover.",
                "priority": "Critical",
                "reason": "Agent or customer suggested immediate coverage of pre-existing conditions.",
                "confidence": 0.90
            },
            {
                "signal": "escalation_requirement",
                "patterns": [
                    r"\b(supervisor|manager|representative|human|speak to someone|connect me|tao|orang|customer service|staf|petugas|cs)\b"
                ],
                "nudge": "Escalation requested: Prepare to transfer the call to a human supervisor.",
                "priority": "Critical",
                "reason": "Customer explicitly requested transfer to a human or manager.",
                "confidence": 0.90
            },
            {
                "signal": "customer_frustration",
                "patterns": [
                    r"\b(mahal|expensive|waste of money|sayang pera|terrible|bad service|angry|annoyed|frustrated|unfair|denda|lambat|sucks|hate|tidak adil|kecewa|marah)\b",
                    r"\b(so expensive|too high|costly|lousy|slow|stupid|worst|waiting too long)\b"
                ],
                "nudge": "Recommend empathy statement",
                "priority": "High",
                "reason": "Customer expressed annoyance, anger, or cost objections.",
                "confidence": 0.85
            },
            {
                "signal": "payment_difficulty",
                "patterns": [
                    r"\b(cannot pay|no money|insufficient|next week|later|installment|cicilan|tidak ada uang|bokek|belum gajian|minta tempo|nyicil|tunda|late fee)\b",
                    r"\b(susah bayar|telat bayar|tidak sanggup)\b"
                ],
                "nudge": "Suggest callback, payment assistance program, or installment schedule.",
                "priority": "High",
                "reason": "Customer mentioned difficulty making payments or requested installment schedules.",
                "confidence": 0.85
            },
            {
                "signal": "buying_signal",
                "patterns": [
                    r"\b(want to buy|sign up|interested|upgrade|avail|purchase|kumuha|beli|saya mau|tertarik|ingin daftar|bind coverage)\b",
                    r"\b(how much is the premium|how to pay|send link|bisa bantu daftar|mau beli)\b"
                ],
                "nudge": "Present payment options or explain how to bind coverage",
                "priority": "Medium",
                "reason": "Customer asked how to purchase, pay, or subscribe.",
                "confidence": 0.90
            },
            {
                "signal": "intent_change",
                "patterns": [
                    r"\b(change my mind|actually|instead|wait|pala|ternyata|eh|sebentar|tunggu)\b"
                ],
                "nudge": "Intent shift detected: pivot conversation to address the new customer request.",
                "priority": "Medium",
                "reason": "Customer redirected conversation topic suddenly.",
                "confidence": 0.70
            },
            {
                "signal": "callback_request",
                "patterns": [
                    r"\b(call me back|later|busy|tomorrow|tawagan|telepon nanti|telepon kembali|hubungi nanti|sibuk|sedang rapat)\b"
                ],
                "nudge": "Acknowledge immediately, check preferred time, and offer callback.",
                "priority": "Medium",
                "reason": "Customer requested a callback due to schedule constraints.",
                "confidence": 0.80
            },
            {
                "signal": "question_repetition",
                "patterns": [
                    r"\b(repeat|what did you say|say again|maternity again|cataract again|pakiulit|apa tadi)\b"
                ],
                "nudge": "Clarify standard coverage terms simply. Use clear language.",
                "priority": "Medium",
                "reason": "Customer asked to repeat policy terms or coverage details.",
                "confidence": 0.75
            },
            {
                "signal": "missed_cross_sell",
                "patterns": [
                    r"\b(another car|second vehicle|family member|wife|child|motorcycle|kotse|sasakyan|motor|istri|anak|suami|mobil kedua|kendaraan lain|children|spouse)\b"
                ],
                "nudge": "Pitch Multi-Vehicle or Family Float discount terms.",
                "priority": "Low",
                "reason": "Customer mentioned secondary assets (second car) or family members.",
                "confidence": 0.75
            },
            {
                "signal": "language_change",
                "patterns": [
                    r"\b(tagalog|taglish|filipino|bahasa|indonesia|ingat po|po|opo|masyado|kasi|po kayo|selamat siang)\b"
                ],
                "nudge": "Suggest adjusting localized model (Taglish/Bahasa) for smoother interaction.",
                "priority": "Low",
                "reason": "Customer introduced localized words or switched languages.",
                "confidence": 0.70
            }
        ]

    def _get_or_create_state(self, session_id: str) -> dict:
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                "active_nudges": [],
                "cooldowns": {},
                "last_turns": {},
                "sentiment": "Neutral",
                "language": "English",
                "intent": "Inquiry"
            }
        return self.session_states[session_id]

    def run_llm_detection(self, transcript: str) -> dict:
        """
        Sends transcript to OpenAI for multi-faceted signal detection, intent classification,
        sentiment analysis, and language detection.
        """
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key.startswith("your_"):
            return {}
            
        system_prompt = (
            "You are a real-time conversational analysis engine processing customer transcripts. "
            "Your task is to analyze the transcript and return a JSON object with: "
            "1. sentiment: 'Positive', 'Neutral', 'Negative', 'Frustrated'.\n"
            "2. language: 'English', 'Taglish', 'Filipino', 'Bahasa Indonesia', 'Mixed'.\n"
            "3. intent: 'Inquiry', 'Objection', 'Complaint', 'Payment Assistance', 'Escalation', 'Callback'.\n"
            "4. detected_signals: array of objects containing:\n"
            "   - signal: e.g. buying_signal, customer_frustration, compliance_issue, payment_difficulty, "
            "             callback_request, intent_change, missed_cross_sell, risk_statement, question_repetition, "
            "             escalation_requirement, language_change.\n"
            "   - confidence: float (0.0 to 1.0)\n"
            "   - priority: 'Low', 'Medium', 'High', 'Critical'\n"
            "   - reason: concise explanation of trigger context\n"
            "   - recommendation: short actionable nudge recommendation text\n\n"
            "Format your output strictly as a JSON object matching this schema:\n"
            "{\n"
            "  \"sentiment\": \"Neutral\",\n"
            "  \"language\": \"English\",\n"
            "  \"intent\": \"Inquiry\",\n"
            "  \"detected_signals\": [\n"
            "    {\n"
            "      \"signal\": \"buying_signal\",\n"
            "      \"confidence\": 0.85,\n"
            "      \"priority\": \"Medium\",\n"
            "      \"reason\": \"Customer asked about premium payments.\",\n"
            "      \"recommendation\": \"Present payment options or explain how to bind coverage.\"\n"
            "    }\n"
            "  ]\n"
            "}"
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
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[NudgeEngine LLM Error]: {e}. Falling back to rule-based engine.")
            return {}

    def run_fallback_detection(self, transcript: str) -> dict:
        """
        Regex keyword pattern match fallback when LLM is unavailable.
        """
        detected_signals = []
        text_lower = transcript.lower()
        
        # Core heuristics for metadata
        sentiment = "Neutral"
        language = "English"
        intent = "Inquiry"
        
        # Sentiment heuristic
        if any(w in text_lower for w in ["mahal", "terrible", "bad service", "angry", "annoyed", "frustrated", "worst", "kecewa", "marah"]):
            sentiment = "Frustrated"
            intent = "Complaint"
        
        # Language heuristic
        if any(w in text_lower for w in ["po", "opo", "naku", "salamat", "paalam"]):
            language = "Taglish"
        elif any(w in text_lower for w in ["cicilan", "denda", "jatuh tempo", "angsuran", "pembiayaan"]):
            language = "Bahasa Indonesia"
            
        # Intent heuristic
        if any(w in text_lower for w in ["buy", "interested", "upgrade", "beli", "daftar"]):
            intent = "Objection" if "mahal" in text_lower else "Inquiry"
        if any(w in text_lower for w in ["supervisor", "manager", "representative", "human", "tao"]):
            intent = "Escalation"
            
        for rule in self.fallback_rules:
            matched = False
            for pat in rule["patterns"]:
                if re.search(pat, text_lower):
                    matched = True
                    break
            
            if matched:
                detected_signals.append({
                    "signal": rule["signal"],
                    "confidence": rule["confidence"],
                    "priority": rule["priority"],
                    "reason": rule["reason"],
                    "recommendation": rule["nudge"]
                })
        
        # Simple Question Repetition check (simulated repetition metric)
        words = text_lower.split()
        for w in set(words):
            if len(w) > 4 and words.count(w) >= 3 and w not in ["about", "their", "there", "would", "could", "should", "dengan", "untuk", "dalam"]:
                detected_signals.append({
                    "signal": "question_repetition",
                    "confidence": 0.75,
                    "priority": "Medium",
                    "reason": f"Customer repeated the term '{w}' multiple times.",
                    "recommendation": "Clarify standard terms and ask customer if they need supervisor assistance."
                })
                break

        return {
            "sentiment": sentiment,
            "language": language,
            "intent": intent,
            "detected_signals": detected_signals
        }

    def process_transcript(self, transcript: str, session_id: str, turn_index: int) -> dict:
        """
        Analyzes the transcript, runs signal detection, filters outcomes, and groups them.
        Applies expiry pruning, duplicate checks, cooldowns, and queue priorities.
        """
        start_time = time.time()
        
        # 1. Run detection (LLM first, fallback to regex)
        result = self.run_llm_detection(transcript)
        
        # Format normalization for list type inputs (mocks)
        if isinstance(result, list):
            result = {"detected_signals": result}
        elif not isinstance(result, dict) or "detected_signals" not in result:
            result = self.run_fallback_detection(transcript)
            
        llm_latency = time.time() - start_time
        
        state = self._get_or_create_state(session_id)
        current_time = time.time()
        
        # Update session metadata
        state["sentiment"] = result.get("sentiment", "Neutral")
        state["language"] = result.get("language", "English")
        state["intent"] = result.get("intent", "Inquiry")
        
        # Step A: Prune expired nudges from active queue
        state["active_nudges"] = [n for n in state["active_nudges"] if current_time <= n["expires_at"]]
        
        new_candidates = []
        valid_signals = []
        
        # Process newly detected signals
        for sig in result.get("detected_signals", []):
            name = sig["signal"]
            conf = sig["confidence"]
            recommendation = sig.get("recommendation", sig.get("nudge", ""))
            
            # Map name to priority/severity class
            if name in ["compliance_issue", "risk_statement", "escalation_requirement"]:
                priority = "Critical"
            elif name in ["customer_frustration", "payment_difficulty"]:
                priority = "High"
            elif name in ["buying_signal", "callback_request", "intent_change", "question_repetition"]:
                priority = "Medium"
            else:
                priority = "Low"
                
            # Allow priority override if explicitly provided
            priority = sig.get("priority", priority)
            reason = sig.get("reason", "Detected pattern.")
            
            # 1. Confidence Threshold filter
            if conf < self.confidence_threshold:
                continue
                
            # 2. Cooldown filter
            last_cooldown_time = state["cooldowns"].get(name, 0.0)
            last_turn_idx = state["last_turns"].get(name, -1)
            
            # Only enforce cooldown if previously triggered
            if last_cooldown_time > 0.0:
                time_elapsed = current_time - last_cooldown_time
                turn_elapsed = turn_index - last_turn_idx
                if time_elapsed < self.cooldown_seconds or turn_elapsed < self.cooldown_turns:
                    continue
            
            # 3. Duplicate filter against existing active nudges
            is_dup = False
            for active_n in state["active_nudges"]:
                if active_n["type"] == name and active_n["recommendation"] == recommendation:
                    is_dup = True
                    break
            if is_dup:
                continue
                
            # Prepare new nudge dictionary
            expires_at = current_time + 15.0 # Expiry timer: 15 seconds
            timestamp = time.strftime("%H:%M:%S")
            
            new_n = {
                "type": name,
                "confidence": conf,
                "timestamp": timestamp,
                "priority": priority,
                "expires_at": expires_at,
                "reason": reason,
                "recommendation": recommendation,
                "severity": priority
            }
            new_candidates.append(new_n)
            
            # Save for single-turn return values
            valid_signals.append({
                "signal": name,
                "confidence": conf,
                "recommendation": recommendation,
                "priority": priority,
                "reason": reason
            })
            
            # Update cooldown records
            state["cooldowns"][name] = current_time
            state["last_turns"][name] = turn_index

        # Step B: Latest Nudge Replacement & Incremental Additions
        for candidate in new_candidates:
            replaced = False
            for idx, active_n in enumerate(state["active_nudges"]):
                if active_n["type"] == candidate["type"]:
                    state["active_nudges"][idx] = candidate
                    replaced = True
                    break
            if not replaced:
                state["active_nudges"].append(candidate)

        # Step C: Expiry Pruning again (to ensure freshness)
        state["active_nudges"] = [n for n in state["active_nudges"] if current_time <= n["expires_at"]]

        # Step D: Priority Topic Grouping (based on signal_priorities list)
        state["active_nudges"].sort(key=lambda x: (self.signal_priorities.get(x["type"], 99), -x["expires_at"]))

        # Step E: Maximum active nudges check (Default 5)
        if len(state["active_nudges"]) > self.max_active_nudges:
            state["active_nudges"] = state["active_nudges"][:self.max_active_nudges]

        # Step F: Extract Latest Nudge & Format Backwards-compatible outputs
        if valid_signals:
            # Sort valid signals triggered this turn by original priority ordering
            valid_signals.sort(key=lambda x: self.signal_priorities.get(x["signal"], 99))
            newly_triggered = valid_signals[0]
            out_signal = newly_triggered["signal"]
            out_confidence = newly_triggered["confidence"]
            out_nudge = newly_triggered["recommendation"]
            
            latest_nudge = {
                "type": out_signal,
                "confidence": out_confidence,
                "recommendation": out_nudge,
                "priority": newly_triggered["priority"],
                "reason": newly_triggered["reason"],
                "timestamp": time.strftime("%H:%M:%S"),
                "expires_at": current_time + 15.0
            }
        else:
            out_signal = "None"
            out_confidence = 0.0
            out_nudge = "No active nudges at this time."
            if state["active_nudges"]:
                latest_nudge = state["active_nudges"][0]
            else:
                latest_nudge = {
                    "type": "None",
                    "confidence": 0.0,
                    "recommendation": "No active nudges at this time.",
                    "priority": "Low",
                    "reason": "None",
                    "timestamp": "--:--:--",
                    "expires_at": 0.0
                }

        return {
            "signal": out_signal,           # Backwards compatible key
            "confidence": out_confidence,   # Backwards compatible key
            "nudge": out_nudge,             # Backwards compatible key
            "active_nudges": state["active_nudges"],
            "latest_nudge": latest_nudge,
            "sentiment": state["sentiment"],
            "language": state["language"],
            "intent": state["intent"],
            "latency": llm_latency,
            "timestamp": time.strftime("%H:%M:%S")
        }
