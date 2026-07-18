import unittest
from voice_agent.localization import LOCALIZATION_CONFIGS
from voice_agent.conversation_manager import ConversationManager
from vector_store.store import ChromaVectorStore
from backend.main import resolve_grounded_query

class TestLocalization(unittest.TestCase):

    def setUp(self):
        self.store = ChromaVectorStore()
        self.mgr = ConversationManager(self.store, resolve_grounded_query)

    def test_configs_exist(self):
        self.assertIn("default", LOCALIZATION_CONFIGS)
        self.assertIn("philippines", LOCALIZATION_CONFIGS)
        self.assertIn("indonesia", LOCALIZATION_CONFIGS)

    def test_philippines_locales(self):
        ph_config = LOCALIZATION_CONFIGS["philippines"]
        self.assertEqual(ph_config["tts_lang"], "tl")
        self.assertIn("beneficiary", ph_config["system_prompt"])
        self.assertIn("rider", ph_config["system_prompt"])
        self.assertIn("lapse", ph_config["system_prompt"])

    def test_indonesia_locales(self):
        id_config = LOCALIZATION_CONFIGS["indonesia"]
        self.assertEqual(id_config["tts_lang"], "id")
        self.assertIn("cicilan", id_config["system_prompt"])
        self.assertIn("denda", id_config["system_prompt"])

    def test_philippines_intent_detection(self):
        # Test objection keyword in Taglish
        text_obj = "Naku, sobrang mahal naman ng premium na yan!"
        intent = self.mgr.detect_intent(text_obj, bot_type="philippines")
        self.assertEqual(intent, "objection")

        # Test escalation keyword in Tagalog
        text_esc = "Gusto ko makausap ang operator o tao."
        intent = self.mgr.detect_intent(text_esc, bot_type="philippines")
        self.assertEqual(intent, "escalate")

        # Test farewell keyword in Tagalog
        text_bye = "Maraming salamat po, paalam!"
        intent = self.mgr.detect_intent(text_bye, bot_type="philippines")
        self.assertEqual(intent, "farewell")

    def test_indonesia_intent_detection(self):
        # Test objection keyword in Bahasa
        text_obj = "Waduh angsuran motor ini terlalu mahal bagi saya."
        intent = self.mgr.detect_intent(text_obj, bot_type="indonesia")
        self.assertEqual(intent, "objection")

        # Test escalation keyword in Bahasa
        text_esc = "Tolong hubungkan ke customer service sekarang."
        intent = self.mgr.detect_intent(text_esc, bot_type="indonesia")
        self.assertEqual(intent, "escalate")

        # Test farewell keyword in Bahasa
        text_bye = "Baik, terima kasih banyak. Selamat tinggal."
        intent = self.mgr.detect_intent(text_bye, bot_type="indonesia")
        self.assertEqual(intent, "farewell")

if __name__ == "__main__":
    unittest.main()
