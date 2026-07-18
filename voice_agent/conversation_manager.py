import time
from voice_agent.localization import LOCALIZATION_CONFIGS

class ConversationManager:
    """
    Orchestrates dialogue management, manages conversation history sessions,
    filters intent objections/escalations/farewells using localization layers,
    and routes regular questions to the unified backend RAG service helper.
    """
    def __init__(self, vector_store, query_resolver):
        self.store = vector_store
        self.query_resolver = query_resolver
        self.sessions = {}  # In-memory dictionary: session_id -> session_state

    def get_or_create_session(self, session_id: str, bot_type: str = "default") -> dict:
        """
        Retrieves or initializes session storage for conversation tracking.
        """
        if not session_id:
            session_id = f"session_{int(time.time())}"
            
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "session_id": session_id,
                "history": [],
                "active": True,
                "last_intent": "greeting",
                "timestamp": time.time(),
                "bot_type": bot_type
            }
        else:
            # Update bot type if provided explicitly
            if bot_type and bot_type != "default":
                self.sessions[session_id]["bot_type"] = bot_type
                
        return self.sessions[session_id]

    def detect_intent(self, text: str, bot_type: str = "default") -> str:
        """
        Classifies incoming transcripts into normal_query, objection, escalate, or farewell
        based on localized trigger keywords.
        """
        text_lower = text.lower().strip()
        config = LOCALIZATION_CONFIGS.get(bot_type, LOCALIZATION_CONFIGS["default"])
        
        if any(kw in text_lower for kw in config.get("farewell_keywords", [])):
            return "farewell"
            
        if any(kw in text_lower for kw in config.get("escalation_keywords", [])):
            return "escalate"
            
        if any(kw in text_lower for kw in config.get("objection_keywords", [])):
            return "objection"
            
        return "normal_query"

    def process_message(self, text: str, session_id: str = None, bot_type: str = "default") -> dict:
        """
        Processes statement within a tracking session context. Objections/escalations/farewells 
        return standard responses; normal queries consult the history-aware RAG resolver.
        """
        session = self.get_or_create_session(session_id, bot_type)
        current_bot = session.get("bot_type", bot_type)
        config = LOCALIZATION_CONFIGS.get(current_bot, LOCALIZATION_CONFIGS["default"])
        
        # Check active session constraints
        if not session["active"]:
            return {
                "answer": "This voice session is currently inactive. Please start a new conversation.",
                "confidence": 1.0,
                "source": "Session Manager",
                "page": "N/A",
                "url": "",
                "intent": "session_inactive",
                "session_id": session["session_id"],
                "active": False
            }

        intent = self.detect_intent(text, current_bot)
        
        # 1. Farewell/Exit Trigger
        if intent == "farewell":
            session["active"] = False
            session["last_intent"] = "farewell"
            answer = config.get("farewell_response", "Goodbye!")
            session["history"].append({"role": "user", "text": text})
            session["history"].append({"role": "assistant", "text": answer})
            return {
                "answer": answer,
                "confidence": 1.0,
                "source": "Session Manager",
                "page": "N/A",
                "url": "",
                "intent": "farewell",
                "session_id": session["session_id"],
                "active": False
            }

        # 2. Escalation Trigger
        if intent == "escalate":
            session["active"] = False  # End agent session to bridge human line
            session["last_intent"] = "escalate"
            answer = config.get("escalation_response", "I will connect you with support.")
            session["history"].append({"role": "user", "text": text})
            session["history"].append({"role": "assistant", "text": answer})
            return {
                "answer": answer,
                "confidence": 1.0,
                "source": "Escalation Router",
                "page": "N/A",
                "url": "",
                "intent": "escalate",
                "session_id": session["session_id"],
                "active": False
            }
            
        # 3. Value/Price Objection Trigger
        if intent == "objection":
            session["last_intent"] = "objection"
            answer = config.get("objection_response", "I understand your concern. Let's discuss options.")
            session["history"].append({"role": "user", "text": text})
            session["history"].append({"role": "assistant", "text": answer})
            return {
                "answer": answer,
                "confidence": 1.0,
                "source": "Intelligent Objection Handler",
                "page": "N/A",
                "url": "",
                "intent": "objection",
                "session_id": session["session_id"],
                "active": True
            }

        # 4. Normal RAG Inquiries (Context & History Aware)
        session["last_intent"] = "normal_query"
        
        # Pass conversation history list and bot type to allow query rephrasing before indexing
        res = self.query_resolver(text, self.store, session["history"], bot_type=current_bot)
        
        answer = res["answer"]
        session["history"].append({"role": "user", "text": text})
        session["history"].append({"role": "assistant", "text": answer})
        
        return {
            "answer": answer,
            "confidence": res.get("confidence", 0.0),
            "source": str(res.get("source", "Unknown")),
            "page": str(res.get("page", "1")),
            "url": str(res.get("url", "")),
            "intent": "normal_query",
            "session_id": session["session_id"],
            "active": True,
            "retrieved_chunks": res.get("retrieved_chunks", [])
        }
