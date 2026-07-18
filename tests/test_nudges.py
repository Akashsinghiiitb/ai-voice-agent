import unittest
import time
from voice_agent.nudge_engine import NudgeEngine

class TestNudgeEngine(unittest.TestCase):

    def setUp(self):
        # Instantiate nudge engine with small cooldowns for easy testing
        self.engine = NudgeEngine(
            confidence_threshold=0.65,
            cooldown_seconds=1.0,
            cooldown_turns=2
        )

    def test_fallback_signal_detection(self):
        # Frustration detection
        res = self.engine.process_transcript("This is terrible service, I am so angry and frustrated!", "session_1", 0)
        self.assertEqual(res["signal"], "customer_frustration")
        self.assertIn("empathy statement", res["nudge"].lower())

        # Payment difficulty detection
        res_pay = self.engine.process_transcript("I cannot pay my cicilan right now, I don't have enough money.", "session_2", 0)
        self.assertEqual(res_pay["signal"], "payment_difficulty")
        self.assertIn("installment", res_pay["nudge"].lower())

        # Buying signal
        res_buy = self.engine.process_transcript("I want to buy auto insurance and upgrade my policy.", "session_3", 0)
        self.assertEqual(res_buy["signal"], "buying_signal")
        self.assertIn("present payment options", res_buy["nudge"].lower())

    def test_confidence_threshold(self):
        # Low confidence signal in custom rules (mock test)
        # We manually inject a detection list to verify threshold filtering
        self.engine.run_llm_detection = lambda t: [
            {"signal": "buying_signal", "confidence": 0.40, "nudge": "Pitch upgrade"}
        ]
        res = self.engine.process_transcript("low confidence query", "session_4", 0)
        self.assertEqual(res["signal"], "None")
        self.assertEqual(res["confidence"], 0.0)

    def test_duplicate_suppression(self):
        # First trigger
        res1 = self.engine.process_transcript("This is terrible, I am angry", "session_5", 0)
        self.assertEqual(res1["signal"], "customer_frustration")

        # Second trigger with identical query/nudge right after
        # It should be suppressed (return "None")
        res2 = self.engine.process_transcript("This is terrible, I am angry", "session_5", 1)
        self.assertEqual(res2["signal"], "None")

    def test_cooldown_by_turns(self):
        # Disable time cooldown for this test
        self.engine.cooldown_seconds = 0.0
        # First trigger
        res1 = self.engine.process_transcript("I cannot pay today, no money", "session_6", 0)
        self.assertEqual(res1["signal"], "payment_difficulty")

        # Second trigger on turn 1 (within cooldown_turns = 2) -> suppressed
        res2 = self.engine.process_transcript("I cannot pay today, no money", "session_6", 1)
        self.assertEqual(res2["signal"], "None")

        # Third trigger on turn 2 (cooldown turns elapsed: 2 - 0 = 2) -> triggers!
        # Make sure the nudge text is slightly different to bypass duplicate suppression
        self.engine.run_llm_detection = lambda t: [
            {"signal": "payment_difficulty", "confidence": 0.90, "nudge": "Offer a payment waiver program"}
        ]
        res3 = self.engine.process_transcript("No money left to pay", "session_6", 2)
        self.assertEqual(res3["signal"], "payment_difficulty")
        self.assertEqual(res3["nudge"], "Offer a payment waiver program")

    def test_cooldown_by_time(self):
        # First trigger
        res1 = self.engine.process_transcript("Call me back tomorrow", "session_7", 0)
        self.assertEqual(res1["signal"], "callback_request")

        # Try triggering again immediately with a different nudge text
        # (It should be suppressed by time cooldown of 1.0 second)
        self.engine.run_llm_detection = lambda t: [
            {"signal": "callback_request", "confidence": 0.85, "nudge": "Schedule callback time"}
        ]
        res2 = self.engine.process_transcript("Call me back later", "session_7", 2) # turn 2 clears turn cooldown
        self.assertEqual(res2["signal"], "None") # suppressed by time

        # Wait 1.1 seconds for time cooldown to expire
        time.sleep(1.1)
        res3 = self.engine.process_transcript("Call me back later", "session_7", 3)
        self.assertEqual(res3["signal"], "callback_request")

    def test_priority_grouping(self):
        # Triggering both Customer Frustration (Priority 2) and Buying Signal (Priority 5)
        # Customer Frustration should be selected because it is higher priority (lower priority index)
        self.engine.run_llm_detection = lambda t: [
            {"signal": "buying_signal", "confidence": 0.90, "nudge": "Present payment options"},
            {"signal": "customer_frustration", "confidence": 0.90, "nudge": "Recommend empathy statement"}
        ]
        res = self.engine.process_transcript("angry but interested", "session_8", 0)
        self.assertEqual(res["signal"], "customer_frustration")
        self.assertEqual(res["nudge"], "Recommend empathy statement")

    def test_compliance_and_risk_signals(self):
        # Test compliance gap check (forgot recording warning)
        res_comp = self.engine.process_transcript("Are you recording this call? Legal privacy.", "session_comp", 0)
        self.assertEqual(res_comp["signal"], "compliance_issue")
        self.assertIn("recording disclosure", res_comp["nudge"].lower())

        # Test risk statement (unsupported day-one cover promise)
        res_risk = self.engine.process_transcript("Yes pre-existing disease immediate cover from day 1.", "session_risk", 0)
        self.assertEqual(res_risk["signal"], "risk_statement")
        self.assertIn("standard 48-month waiting period", res_risk["nudge"].lower())

        # Test escalation request
        res_esc = self.engine.process_transcript("Let me speak to your supervisor or manager right now.", "session_esc", 0)
        self.assertEqual(res_esc["signal"], "escalation_requirement")
        self.assertIn("transfer the call to a human supervisor", res_esc["nudge"].lower())

    def test_queue_limitations(self):
        # Trigger 6 distinct signals to exceed the max_active_nudges limit (default 5)
        self.engine.max_active_nudges = 5
        self.engine.run_llm_detection = lambda t: [
            {"signal": "buying_signal", "confidence": 0.90, "nudge": "A"},
            {"signal": "customer_frustration", "confidence": 0.90, "nudge": "B"},
            {"signal": "compliance_issue", "confidence": 0.90, "nudge": "C"},
            {"signal": "payment_difficulty", "confidence": 0.90, "nudge": "D"},
            {"signal": "callback_request", "confidence": 0.90, "nudge": "E"},
            {"signal": "intent_change", "confidence": 0.90, "nudge": "F"}
        ]
        res = self.engine.process_transcript("test active queue limit exceed", "session_limit", 0)
        # Verify queue is capped at 5
        self.assertEqual(len(res["active_nudges"]), 5)

    def test_expiry_pruning(self):
        self.engine.run_llm_detection = lambda t: [
            {"signal": "buying_signal", "confidence": 0.90, "nudge": "Purchase insurance"}
        ]
        res1 = self.engine.process_transcript("buy now", "session_expiry", 0)
        self.assertEqual(len(res1["active_nudges"]), 1)

        # Manually alter the active nudge's expires_at timestamp to be in the past
        self.engine.session_states["session_expiry"]["active_nudges"][0]["expires_at"] = time.time() - 1.0

        # Triggering a blank check - the expired nudge should be pruned
        self.engine.run_llm_detection = lambda t: []
        res2 = self.engine.process_transcript("blank text", "session_expiry", 1)
        self.assertEqual(len(res2["active_nudges"]), 0)

if __name__ == "__main__":
    unittest.main()

