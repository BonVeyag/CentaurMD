from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


def _now_utc() -> datetime:
    # Use timezone-aware UTC to avoid subtle comparisons/serialization issues.
    return datetime.now(timezone.utc)


# =========================
# Shared strict base model (Pydantic v2)
# =========================

class StrictBaseModel(BaseModel):
    """
    Strict, assignment-validating base model (Pydantic v2).
    - extra fields are forbidden (schema discipline)
    - assignment is validated (catches subtle runtime drift)
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# =========================
# Session metadata
# =========================

class SessionMeta(StrictBaseModel):
    session_id: str
    clinician_id: Optional[str] = None

    created_at: datetime = Field(default_factory=_now_utc)
    last_updated_at: datetime = Field(default_factory=_now_utc)

    context_version: Literal["v1"] = "v1"
    locale: str = "en-CA"
    jurisdiction: str = "Alberta"


# =========================
# Patient anchor (canonical lightweight identifiers)
# =========================

SexLiteral = Literal["male", "female", "other", "unknown"]


class PatientAnchor(StrictBaseModel):
    """
    Canonical patient identifiers that are safe to inject broadly into prompts.
    This is the preferred location for demographics extraction / display.
    """
    patient_ref: Optional[str] = None  # optional internal reference

    name: str = ""
    dob: str = ""   # leave as string (EMR formats vary)
    phn: str = ""   # leave as string; may include leading zeros if ever present

    age: Optional[int] = None
    sex: Optional[SexLiteral] = None


# =========================
# Patient demographics (optional / future-facing)
# =========================

class PatientDemographics(StrictBaseModel):
    """
    Optional structured demographics store.
    If you are not actively using it, keep it, but treat PatientAnchor as canonical.
    """
    name: str = ""
    dob: str = ""
    phn: str = ""

    age: Optional[int] = None
    sex: Optional[SexLiteral] = None


# =========================
# Clinical background
# =========================

class ClinicalBackground(StrictBaseModel):
    emr_dump: Optional[str] = None
    netcare_dump: Optional[str] = None
    consult_notes: List[str] = Field(default_factory=list)


# =========================
# Transcript
# =========================

class TranscriptSegment(StrictBaseModel):
    speaker: Literal["clinician", "patient", "other"]
    text: str
    timestamp_start: datetime
    timestamp_end: datetime


class Transcript(StrictBaseModel):
    mode: Literal["live_audio", "imported"] = "live_audio"
    segments: List[TranscriptSegment] = Field(default_factory=list)
    raw_text: str = ""


# =========================
# Ambient capture (session-scoped)
# =========================

class AmbientSegment(StrictBaseModel):
    segment_id: str
    start_ts: datetime
    end_ts: datetime
    text: str
    language: Optional[str] = None
    text_en: Optional[str] = None


class AmbientEncounter(StrictBaseModel):
    encounter_id: str
    session_id: str
    created_at: datetime = Field(default_factory=_now_utc)
    consent_confirmed: bool = False
    segments: List[AmbientSegment] = Field(default_factory=list)
    transcript_assembled: str = ""


class AmbientState(StrictBaseModel):
    active: Optional[AmbientEncounter] = None


# =========================
# Clinician inputs
# =========================

class ClinicianInputs(StrictBaseModel):
    free_text_notes: List[str] = Field(default_factory=list)
    pasted_text: List[str] = Field(default_factory=list)


# =========================
# Interaction state
# =========================

class ChatMessage(StrictBaseModel):
    role: Literal["clinician", "assistant"]
    message: str
    timestamp: datetime = Field(default_factory=_now_utc)


class TaskInvocation(StrictBaseModel):
    task_type: Literal[
        "MAKE_SOAP",
        "WRITE_REFERRAL",
        "RUN_DIFFERENTIAL",
    ]
    invoked_at: datetime = Field(default_factory=_now_utc)
    input_snapshot_hash: str
    output_ref: str


class InteractionState(StrictBaseModel):
    chat_history: List[ChatMessage] = Field(default_factory=list)
    task_history: List[TaskInvocation] = Field(default_factory=list)


# =========================
# Derived outputs
# =========================

class SoapNoteOutput(StrictBaseModel):
    """
    SOAP output stored in-session.
    - text: the canonical chart-ready text
    - structured: optional structured representation for future UI uses
    """
    text: str
    structured: Optional[Dict[str, Any]] = None
    generated_at: datetime = Field(default_factory=_now_utc)


class ReferralLetterOutput(StrictBaseModel):
    id: str
    text: str
    generated_at: datetime = Field(default_factory=_now_utc)


class DerivedOutputs(StrictBaseModel):
    """
    Cached outputs produced during a session to avoid re-computation and for UX continuity.
    """

    # Core clinical artifacts
    soap_note: Optional[SoapNoteOutput] = None
    referrals: List[ReferralLetterOutput] = Field(default_factory=list)

    # Coach output only
    differential: Optional[str] = None

    preventive_flags: List[Dict[str, Any]] = Field(default_factory=list)


# =========================
# Billing (daily state models) — Structured entries (matches service.py)
# =========================

BillingModelLiteral = Literal["FFS", "PCPCM"]

BillingIcd9Source = Literal["user_selected", "ai_suggested"]


class BillingIcd9Code(StrictBaseModel):
    code: str
    label: str
    source: BillingIcd9Source = "user_selected"
    confidence: Optional[float] = None


class BillingSession(StrictBaseModel):
    icd9_codes: List[BillingIcd9Code] = Field(default_factory=list)


class DailyBillingEntry(StrictBaseModel):
    """
    One 3-line billing block as shown/edited in the UI.
    """
    entry_id: str
    created_at_utc: str

    line1_patient: str
    line2_icd9: str
    line3_billing: str

    meta: Dict[str, Any] = Field(default_factory=dict)


class BillingDayState(StrictBaseModel):
    """
    Server-level daily billing state keyed by Edmonton-local date_key ("YYYY-MM-DD").
    Mirrors service.py BillingDayState.to_dict().
    """
    date_key: str
    physician: str = ""
    billing_model: BillingModelLiteral = "FFS"

    entries: List[DailyBillingEntry] = Field(default_factory=list)
    updated_at_utc: str = ""  # isoformat string

    total_patient_count: int = 0


class BillingGetResponse(StrictBaseModel):
    date_key: str
    physician: str
    billing_model: BillingModelLiteral
    total_patient_count: int
    entries: List[DailyBillingEntry] = Field(default_factory=list)
    updated_at_utc: Optional[str] = None
    is_empty: bool = False


class BillingSetHeaderPayload(StrictBaseModel):
    physician: Optional[str] = None
    billing_model: Optional[BillingModelLiteral] = None


class BillingSavePayload(StrictBaseModel):
    """
    Frontend sends the current editable display.
    """
    physician: Optional[str] = None
    billing_model: Optional[BillingModelLiteral] = None
    entries: List[DailyBillingEntry] = Field(default_factory=list)


class BillingBillPayload(StrictBaseModel):
    """
    Frontend clicks Bill:
    - server generates a new entry from the current session context
    - server appends it to the daily state
    """
    physician: Optional[str] = None
    billing_model: Optional[BillingModelLiteral] = None


class BillingBillResponse(BillingDayState):
    """
    Same as BillingDayState but includes new_entry_id from service.py.
    """
    new_entry_id: Optional[str] = None


class BillingPrintEntry(StrictBaseModel):
    entry_id: str
    line1_patient: str
    line2_icd9: str  # stripped to codes only
    line3_billing: str  # stripped to codes only


class BillingPrintResponse(StrictBaseModel):
    """
    Returned by Print:
    - server strips descriptions for ICD-9 and billing codes
    - then clears server-side state for that day
    """
    date_key: str
    physician: str
    billing_model: BillingModelLiteral
    total_patient_count: int
    entries: List[BillingPrintEntry] = Field(default_factory=list)
    generated_at_utc: Optional[str] = None
    cleared: bool = True


# =========================
# Attachments
# =========================

class AttachmentMeta(StrictBaseModel):
    """
    Attachment metadata stored in-session. Raw bytes live elsewhere (in-memory blob store for now).
    kind: "pdf" or "image"
    """
    id: str
    kind: Literal["pdf", "image"]
    filename: str
    mime: str
    size_bytes: int
    uploaded_at: datetime = Field(default_factory=_now_utc)


# =========================
# Session Context (ROOT)
# =========================

class SessionContext(StrictBaseModel):
    session_meta: SessionMeta

    # Canonical lightweight identifiers for prompt injection + UI headers
    patient_anchor: PatientAnchor = Field(default_factory=PatientAnchor)

    # Optional structured demographics store (not required for current flow)
    patient: PatientDemographics = Field(default_factory=PatientDemographics)

    clinical_background: ClinicalBackground = Field(default_factory=ClinicalBackground)
    transcript: Transcript = Field(default_factory=Transcript)
    clinician_inputs: ClinicianInputs = Field(default_factory=ClinicianInputs)
    interaction_state: InteractionState = Field(default_factory=InteractionState)
    derived_outputs: DerivedOutputs = Field(default_factory=DerivedOutputs)

    # Session-scoped attachments (metadata only)
    attachments: List[AttachmentMeta] = Field(default_factory=list)

    # Session-scoped billing helpers (ICD-9 chips, suggestions)
    billing: BillingSession = Field(default_factory=BillingSession)

    # Ambient capture state (session scoped, no raw audio persistence)
    ambient: AmbientState = Field(default_factory=AmbientState)
