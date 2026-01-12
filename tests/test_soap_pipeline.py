import unittest

from app.models import SessionContext, SessionMeta
from app.services import (
    build_encounter_packet,
    normalize_soap_output,
    should_include_procedure_section,
    SOAP_STRUCTURE_SYSTEM,
    SOAP_SYNTHESIS_SYSTEM,
    SOAP_SCRUB_SYSTEM,
)


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

    def test_procedure_section_toggle(self):
        self.assertFalse(should_include_procedure_section({"procedures": {"performed": False}}))
        self.assertTrue(should_include_procedure_section({"procedures": {"performed": True}}))

    def test_social_hx_guardrails_in_prompts(self):
        self.assertIn("Social history must come ONLY from today's transcript", SOAP_STRUCTURE_SYSTEM)
        self.assertIn("Social Hx must be derived only from today", SOAP_SYNTHESIS_SYSTEM)
        self.assertIn("Social Hx is transcript-only", SOAP_SCRUB_SYSTEM)

    def test_format_normalization(self):
        raw = """
Issues:
- 1. Sore throat

Subjective:
- Patient reports sore throat.

Safety / Red Flags:

Plan:
- Supportive care.
""".strip()
        normalized = normalize_soap_output(raw, include_procedure=False)
        self.assertIn("**Issues:**", normalized)
        self.assertIn("Sore throat", normalized)
        self.assertNotIn("-", normalized)
        self.assertIn("**Safety / Red Flags:**\nnone", normalized)


if __name__ == "__main__":
    unittest.main()
