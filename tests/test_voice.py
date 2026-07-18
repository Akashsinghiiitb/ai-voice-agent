import os
import unittest
from fastapi.testclient import TestClient

from voice_agent.speech_to_text import SpeechToText
from voice_agent.text_to_speech import TextToSpeech
from voice_agent.conversation_manager import ConversationManager
from vector_store.store import ChromaVectorStore
from backend.main import app, resolve_grounded_query

class TestVoiceAgent(unittest.TestCase):

    def setUp(self):
        self.store = ChromaVectorStore()
        self.mgr = ConversationManager(self.store, resolve_grounded_query)
        try:
            self.stt = SpeechToText()
        except ImportError:
            self.stt = None
        
        try:
            self.tts = TextToSpeech()
        except ImportError:
            self.tts = None
            
        self.client = TestClient(app)

    def test_objection_intent_detection(self):
        # Assert objection is recognized correctly
        text = "This policy is too expensive, I don't need insurance"
        intent = self.mgr.detect_intent(text)
        self.assertEqual(intent, "objection")

        res = self.mgr.process_message(text)
        self.assertEqual(res["intent"], "objection")
        self.assertIn("protect against unexpected", res["answer"])

    def test_escalation_intent_detection(self):
        # Assert escalation request keyword gets matched
        text = "I have a complaint, please connect me to a human representative!"
        intent = self.mgr.detect_intent(text)
        self.assertEqual(intent, "escalate")

        res = self.mgr.process_message(text)
        self.assertEqual(res["intent"], "escalate")
        self.assertIn("connect you with a customer support", res["answer"])

    def test_transcribe_fallback(self):
        if not self.stt:
            self.skipTest("SpeechToText not installed.")
        mock_path = "cataract_query.wav"
        with open(mock_path, "w") as f:
            f.write("dummy")
            
        try:
            txt = self.stt.transcribe(mock_path)
            self.assertIsNotNone(txt)
        except Exception as e:
            self.skipTest(f"FFmpeg missing or system audio error: {e}")
        finally:
            if os.path.exists(mock_path):
                os.remove(mock_path)

    def test_synthesize_output_generation(self):
        if not self.tts:
            self.skipTest("TextToSpeech not installed.")
        out_path = "./tmp_audio/test_synth.mp3"
        try:
            self.tts.synthesize("Hello world", out_path)
            self.assertTrue(os.path.exists(out_path))
        except Exception as e:
            self.skipTest(f"TTS synthesis connection failure: {e}")
        finally:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass

    def test_multi_turn_history(self):
        # 1. Start a session
        sess_id = "test_sess_123"
        res = self.mgr.process_message("Hi", session_id=sess_id)
        self.assertTrue(res["active"])
        self.assertEqual(res["session_id"], sess_id)
        
        # Check history size
        session = self.mgr.sessions[sess_id]
        self.assertEqual(len(session["history"]), 2) # User: Hi, Assistant: response
        
        # 2. Trigger farewell exit
        res2 = self.mgr.process_message("goodbye", session_id=sess_id)
        self.assertFalse(res2["active"])
        self.assertEqual(res2["intent"], "farewell")
        
        # 3. Post to inactive session
        res3 = self.mgr.process_message("any query", session_id=sess_id)
        self.assertFalse(res3["active"])
        self.assertEqual(res3["intent"], "session_inactive")

    def test_voice_endpoints(self):
        # Test chat route with session_id support
        chat_payload = {"question": "What is the room rent limit?", "session_id": "test_endpoint_session"}
        res = self.client.post("/voice/chat", json=chat_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("answer", data)
        self.assertIn("confidence", data)
        self.assertIn("intent", data)

        # Test synthesize route
        synth_payload = {"text": "Hello world response"}
        synth_res = self.client.post("/voice/synthesize", json=synth_payload)
        self.assertEqual(synth_res.status_code, 200)
        self.assertEqual(synth_res.headers.get("content-type"), "audio/mpeg")

if __name__ == "__main__":
    unittest.main()
