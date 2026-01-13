import re
import unittest

from app.models import SessionContext, SessionMeta
from app.services import build_encounter_packet
from app.soap.schema import SoapStructured
from app.soap.renderer import render_soap


class TestSoapPipeline(unittest.TestCase):
    def _make_context(self) -> SessionContext:
        return SessionContext(session_meta=SessionMeta(session_id="sid-1"))

    def test_packet_transcript_only(self):
        context = self._make_context()
        context.transcript.raw_text = "Patient reports mild sore throat."
        packet = build_encounter_packet(context)
        self.assertIn("sore throat", packet["transcript"]["text"])
        self.assertIsNone(packet["emr_context"]["meds"])
        self.assertIsNone(packet["emr_context"]["allergies"])
        self.assertIsNone(packet["objective_data"]["labs"])

    def test_packet_emr_meds_allergies(self):
        context = self._make_context()
        context.clinical_background.emr_dump = (
            "Medications\n"
            "Metformin 500 mg\n"
            "Amlodipine 10 mg\n"
            "Allergies/Intolerances\n"
            "Penicillin rash\n"
        )
        packet = build_encounter_packet(context)
        self.assertIn("Metformin", packet["emr_context"]["meds"] or "")
        self.assertIn("Penicillin", packet["emr_context"]["allergies"] or "")

    def test_packet_attachments_to_objective(self):
        context = self._make_context()
        packet = build_encounter_packet(context, attachments_text="Hb 120 g/L; WBC 9.2")
        labs = packet["objective_data"]["labs"] or ""
        self.assertIn("Hb 120", labs)

    def test_render_formatting_rules(self):
        payload = SoapStructured.parse_obj({
            "issues": [{"number": 1, "title": "Sore throat"}],
            "subjective": [{"issue_number": 1, "lines": ["Patient reports sore throat."]}],
            "safety_red_flags": [],
            "social_hx": [],
            "objective": [],
            "assessment": [{"issue_number": 1, "lines": ["Likely viral pharyngitis."]}],
            "plan": [{"issue_number": 1, "lines": ["Supportive care."]}],
        })
        out = render_soap(payload)
        self.assertIn("**Issues:**", out)
        self.assertIn("**Safety / Red Flags:**\nnone", out)
        self.assertNotRegex(out, r"^[-•]\s", msg="No bullets allowed")
        # Exactly one blank line between sections
        self.assertNotIn("\n\n\n", out)

    def test_render_acceptance_example(self):
        payload = SoapStructured.parse_obj({
            "issues": [{"number": 1, "title": "Cough and chest symptoms"}],
            "subjective": [{"issue_number": 1, "lines": [
                "Cough for 2 months with intermittent shortness of breath.",
            ]}],
            "safety_red_flags": ["Pregnancy check discussed before imaging."],
            "social_hx": [
                "Works with occupational fume exposure; denies smoking or vaping today.",
            ],
            "objective": ["Lungs sound a little wet on exam."],
            "assessment": [{"issue_number": 1, "lines": [
                "Chronic cough with wet lung sounds discussed today.",
            ]}],
            "plan": [{"issue_number": 1, "lines": [
                "CXR ordered.",
                "Phone follow-up tomorrow between 2–3 PM.",
            ]}],
        })
        out = render_soap(payload)
        for phrase in [
            "lungs sound a little wet",
            "Pregnancy check discussed before imaging",
            "CXR ordered",
            "Phone follow-up tomorrow",
            "fume exposure",
            "denies smoking or vaping",
        ]:
            self.assertIn(phrase, out)
        self.assertTrue(re.search(r"\\*\\*Plan:\\*\\*", out))


if __name__ == "__main__":
    unittest.main()
