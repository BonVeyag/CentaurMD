from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from pydantic import Field

from app.models import StrictBaseModel, SessionContext
from app.services import client, extract_demographics_from_text, _clip_text

logger = logging.getLogger("centaurweb.referral")

REFERRAL_MODEL = os.getenv("REFERRAL_MODEL", "gpt-5.2")


def _float_env(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


REFERRAL_TEMPERATURE = _float_env("REFERRAL_TEMPERATURE", 0.3)
REFERRAL_TOP_P = _float_env("REFERRAL_TOP_P", 0.9)

EDMONTON_TZ = ZoneInfo("America/Edmonton")


class ReferralMeta(StrictBaseModel):
    generated_at: str
    timezone: str = "America/Edmonton"
    version: str = "referral.v1"


class PatientInfo(StrictBaseModel):
    full_name: str = ""
    dob: str = ""
    phn: str = ""
    phone: str = ""
    address: str = ""
    language: str = "English"
    interpreter_needed: str = "No"
    guardian_or_sdm: str = ""


class ReferrerInfo(StrictBaseModel):
    name: str = "Dr. Rajat Thapa"
    credentials: str = "MD PhD CCFP"
    cpsa: str = "032698"
    clinic_name: str = ""
    clinic_address: str = ""
    phone: str = ""
    fax: str = ""
    fax_or_econsult_inbox: str = ""
    signature_block: str = ""


class ReferralInfo(StrictBaseModel):
    specialty: str = ""
    specialty_name: str = ""
    subspecialty_or_clinic: str = ""
    reason_short: str = ""
    consult_question: str = ""
    urgency_label: str = "Routine"
    target_timeframe: str = ""
    urgency_rationale: str = ""
    patient_aware_yes_no: str = "Unclear"


class ClinicalBlock(StrictBaseModel):
    summary_symptoms: str = ""
    key_positives: str = ""
    key_negatives_and_redflags: str = ""
    pertinent_exam: str = ""


class ObjectiveLab(StrictBaseModel):
    name: str = ""
    value: str = ""
    units: str = ""
    date: str = ""
    flag: str = ""


class ObjectiveImaging(StrictBaseModel):
    test: str = ""
    result_summary: str = ""
    date: str = ""
    access: str = ""


class ObjectivePathology(StrictBaseModel):
    specimen: str = ""
    diagnosis: str = ""
    date: str = ""
    access: str = ""


class ObjectiveBlock(StrictBaseModel):
    labs: List[ObjectiveLab] = Field(default_factory=list)
    imaging: List[ObjectiveImaging] = Field(default_factory=list)
    pathology: List[ObjectivePathology] = Field(default_factory=list)
    results_location: str = ""
    labs_block: str = ""
    imaging_block: str = ""
    pathology_block: str = ""


class ManagementItem(StrictBaseModel):
    intervention: str = ""
    dose: str = ""
    duration: str = ""
    response: str = ""


class PendingItem(StrictBaseModel):
    item: str = ""
    eta: str = ""
    notes: str = ""


class ManagementBlock(StrictBaseModel):
    tried: List[ManagementItem] = Field(default_factory=list)
    pending: List[PendingItem] = Field(default_factory=list)
    tried_block: str = ""
    pending_block: str = ""


class AssessmentBlock(StrictBaseModel):
    working_dx_and_ddx: str = ""


class BackgroundBlock(StrictBaseModel):
    pmHx_relevant: str = ""
    psHx_relevant: str = ""
    meds_relevant: str = ""
    allergies: str = ""


class LogisticsBlock(StrictBaseModel):
    high_risk_context: str = ""
    barriers: str = ""
    patient_goals: str = ""


class AttachmentItem(StrictBaseModel):
    label: str = ""
    source: str = ""
    date: str = ""


class AttachmentsBlock(StrictBaseModel):
    items: List[AttachmentItem] = Field(default_factory=list)
    list_block: str = ""


class SafetyBlock(StrictBaseModel):
    advice_line: str = "If symptoms worsen (e.g., red flags listed above), patient advised to seek urgent care / ED as appropriate."


class QualityBlock(StrictBaseModel):
    missing_critical: List[str] = Field(default_factory=list)
    missing_recommended: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    confidence: str = "Medium"


class ReferralDraft(StrictBaseModel):
    meta: ReferralMeta
    patient: PatientInfo
    referrer: ReferrerInfo
    referral: ReferralInfo
    clinical: ClinicalBlock
    objective: ObjectiveBlock
    management: ManagementBlock
    assessment: AssessmentBlock
    background: BackgroundBlock
    logistics: LogisticsBlock
    attachments: AttachmentsBlock
    safety: SafetyBlock
    quality: QualityBlock


KNOWN_HEADERS = {
    "family hx",
    "family history",
    "vaccines",
    "health profile",
    "social history",
    "allergies",
    "allergies/intolerances",
    "medications",
    "notices",
    "clinical notes",
    "consults",
    "vitals",
    "invoices",
    "patient lists",
    "reminders",
    "clinic policies",
    "screening",
    "labs",
    "lab",
    "laboratory",
    "investigations",
    "imaging",
    "other documents/forms",
    "letters",
    "calculated results",
    "photos",
    "patient notes",
    "assessment",
    "plan",
    "subjective",
    "objective",
    "soap",
}


def _norm_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


def _is_header(line: str) -> bool:
    raw = _norm_line(line).lower().rstrip(":")
    if not raw:
        return False
    if raw in KNOWN_HEADERS:
        return True
    for h in KNOWN_HEADERS:
        if raw.startswith(h + " "):
            return True
    return False


def _extract_section_block(text: str, headers: List[str], max_lines: int = 10) -> str:
    if not text:
        return ""
    header_set = {h.lower() for h in headers}
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        norm = _norm_line(line).lower().rstrip(":")
        if norm in header_set:
            start_idx = i + 1
            break
    if start_idx is None:
        return ""
    collected: List[str] = []
    for j in range(start_idx, len(lines)):
        line = _norm_line(lines[j])
        if not line:
            continue
        if _is_header(line):
            break
        collected.append(line)
        if len(collected) >= max_lines:
            break
    return "\n".join(collected).strip()


def _extract_section_block_last(text: str, headers: List[str], max_lines: int = 10) -> str:
    if not text:
        return ""
    header_set = {h.lower() for h in headers}
    lines = text.splitlines()
    matches: List[int] = []
    for i, line in enumerate(lines):
        norm = _norm_line(line).lower().rstrip(":")
        if norm in header_set:
            matches.append(i + 1)
    if not matches:
        return ""
    start_idx = matches[-1]
    collected: List[str] = []
    for j in range(start_idx, len(lines)):
        line = _norm_line(lines[j])
        if not line:
            continue
        if _is_header(line):
            break
        collected.append(line)
        if len(collected) >= max_lines:
            break
    return "\n".join(collected).strip()


PHONE_RE = re.compile(r"(?:\+?1[\s.-]*)?\(?\d{3}\)?[\s.-]*\d{3}[\s.-]*\d{4}")
POSTAL_RE = re.compile(r"[A-Za-z]\d[A-Za-z][\s-]?\d[A-Za-z]\d")


def _extract_phone(text: str) -> str:
    if not text:
        return ""
    m = PHONE_RE.search(text)
    return m.group(0).strip() if m else ""


def _extract_address(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if POSTAL_RE.search(line):
            if i > 0 and len(lines[i - 1]) < 64 and not POSTAL_RE.search(lines[i - 1]):
                return f"{lines[i - 1]} {line}".strip()
            return line
    return ""


def _infer_patient_aware(transcript: str) -> str:
    t = (transcript or "").lower()
    if not t:
        return "Unclear"
    if re.search(r"\b(patient|pt) (aware|understands|agreed)\b", t):
        return "Yes"
    if "discussed referral" in t or "referred" in t:
        return "Yes"
    if re.search(r"\b(patient|pt) (not aware|declined|refused)\b", t):
        return "No"
    return "Unclear"


def _summarize_from_transcript_and_emr(
    transcript: str,
    emr_text: str,
    netcare_text: str,
    specialty_override: str = "",
) -> Dict[str, str]:
    t = (transcript or "").strip()
    bg = (emr_text or "").strip()
    nc = (netcare_text or "").strip()
    if not (t or bg or nc):
        return {}
    clip_t = _clip_text(t, max_chars=7000)
    clip_bg = _clip_text(bg, max_chars=5000)
    clip_nc = _clip_text(nc, max_chars=4000)
    focus_parts: List[str] = []
    def add_focus(label: str, block: str) -> None:
        if not block:
            return
        focus_parts.append(f"[{label}]\n{block}".strip())

    add_focus("PATIENT NOTES", _extract_section_block_last(bg, ["Patient Notes"], max_lines=120))
    add_focus("CLINICAL NOTES", _extract_section_block_last(bg, ["Clinical Notes"], max_lines=60))
    add_focus("ACTIVE CONDITIONS", _extract_section_block_last(bg, ["Active & Past Medical Conditions", "Active Medical Conditions"], max_lines=80))
    add_focus("HEALTH PROFILE", _extract_section_block_last(bg, ["Health Profile", "Problem List", "Diagnoses"], max_lines=25))
    add_focus("SOCIAL HISTORY", _extract_section_block_last(bg, ["Social History", "Social Hx"], max_lines=30))
    add_focus("FAMILY HISTORY", _extract_section_block_last(bg, ["Family Hx", "Family History"], max_lines=20))
    add_focus("OBJECTIVE DATA", _extract_section_block_last(bg, ["Objective Data", "Objective"], max_lines=40))
    add_focus("INVESTIGATIONS", _extract_section_block_last(bg, ["Investigations", "Imaging", "Labs"], max_lines=30))
    add_focus("ASSESSMENT", _extract_section_block_last(bg, ["Assessment"], max_lines=30))
    add_focus("PLAN", _extract_section_block_last(bg, ["Plan"], max_lines=40))
    focus_text = "\n\n".join(focus_parts).strip() or "[none]"
    specialty_hint = specialty_override.strip()
    prompt = f"""
Return strict JSON only with keys:
specialty_name, subspecialty_or_clinic, reason_short, consult_question,
summary_symptoms, key_positives, key_negatives_and_redflags, pertinent_exam,
treatments_tried, pending_items, working_dx_and_ddx, patient_goals, target_timeframe,
objective_labs, objective_imaging, objective_pathology, safety_advice.

Rules:
- Use ONLY the transcript and EMR data below.
- You MAY infer specialty_name, reason_short, and consult_question from the clinical context.
- If a specialty override is provided, set specialty_name to that value and tailor reason_short/consult_question to that specialty.
- For all other fields, include only what is supported; if uncertain, return empty string.
- Do NOT invent PMHx, meds, labs, imaging, or diagnoses not stated.
- subspecialty_or_clinic: only if explicitly stated in the EMR or transcript.
- target_timeframe: only if explicitly stated; otherwise empty.
- clinical summary fields: prefer explicitly documented symptoms/findings from transcript/EMR notes.
- objective_labs: list lab name + value + units + date if available; if only test name/date, state "value not listed in EMR".
- objective_imaging: list test + date + brief result summary if present; if report not present, state "report not available in EMR".
- objective_pathology: only if pathology present; otherwise empty string.
- safety_advice: issue-specific red flags and what to do (urgent care/ED) based on the presenting concern.

SPECIALTY OVERRIDE:
{specialty_hint or "[none]"}

TRANSCRIPT:
{clip_t or "[none]"}

EMR:
{clip_bg or "[none]"}

NETCARE:
{clip_nc or "[none]"}

EMR FOCUS:
{focus_text}
""".strip()
    try:
        resp = client.chat.completions.create(
            model=REFERRAL_MODEL,
            messages=[
                {"role": "system", "content": "Return strict JSON only. Use transcript/EMR only."},
                {"role": "user", "content": prompt},
            ],
            temperature=REFERRAL_TEMPERATURE,
            top_p=REFERRAL_TOP_P,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        if isinstance(data, dict):
            return {k: str(v or "").strip() for k, v in data.items()}
    except Exception as exc:
        logger.warning("Referral summary failed: %s", exc)
    return {}


def _suggest_urgency(text: str) -> Tuple[str, str]:
    t = (text or "").lower()
    if not t:
        return ("Routine", "No red flags identified; routine triage.")
    urgent_terms = [
        "syncope",
        "faint",
        "chest pain",
        "shortness of breath",
        "dyspnea",
        "hemoptysis",
        "gi bleed",
        "melena",
        "hematemesis",
        "neuro deficit",
        "weakness",
        "paralysis",
        "severe",
        "uncontrolled",
        "suicidal",
        "self-harm",
        "acute vision loss",
        "seizure",
    ]
    urgent_hits = [term for term in urgent_terms if term in t]
    if urgent_hits:
        unique = ", ".join(sorted(set(urgent_hits)))
        return ("Urgent", f"Red flags noted: {unique}.")
    semi_terms = [
        "worsening",
        "progressive",
        "unable to work",
        "unable to walk",
        "functional decline",
        "significant pain",
    ]
    semi_hits = [term for term in semi_terms if term in t]
    if semi_hits:
        unique = ", ".join(sorted(set(semi_hits)))
        return ("Semi-urgent", f"Symptoms worsening or functionally limiting: {unique}.")
    return ("Routine", "No red flags identified; routine triage.")


def _collect_high_risk_context(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return ""
    tags = []
    if "anticoag" in t or "warfarin" in t or "rivaroxaban" in t or "apixaban" in t:
        tags.append("Anticoagulation")
    if "ckd" in t or "dialysis" in t:
        tags.append("CKD/dialysis")
    if "pregnan" in t:
        tags.append("Pregnancy")
    if "immunosupp" in t or "transplant" in t or "chemo" in t:
        tags.append("Immunosuppression")
    return "; ".join(tags)


def _append_missing(value: str, label: str) -> str:
    v = (value or "").strip()
    tag = f"[MISSING: {label}]"
    if not v:
        return tag
    if tag in v:
        return v
    return f"{v}\n{tag}"


def _detect_minimum_dataset(draft: ReferralDraft, warnings: List[str]) -> None:
    reason = (draft.referral.reason_short or "").lower()
    text = " ".join([
        draft.clinical.summary_symptoms,
        draft.clinical.key_positives,
        draft.clinical.key_negatives_and_redflags,
    ]).lower()
    labs = (draft.objective.labs_block or "").lower()
    if any(k in reason for k in ["iron deficiency", "low ferritin", "microcytic", "anemia"]):
        if not re.search(r"\b(hb|hemoglobin|mcv)\b", labs):
            warnings.append("IDA trigger: missing CBC indices (Hb/MCV).")
            draft.objective.labs_block = _append_missing(draft.objective.labs_block, "CBC indices (Hb/MCV)")
        if "ferritin" not in labs:
            warnings.append("IDA trigger: missing ferritin.")
            draft.objective.labs_block = _append_missing(draft.objective.labs_block, "ferritin")
        if "bleed" not in text and "melena" not in text and "hematochezia" not in text:
            warnings.append("IDA trigger: bleeding history not documented.")
            draft.clinical.key_negatives_and_redflags = _append_missing(
                draft.clinical.key_negatives_and_redflags, "bleeding history"
            )

    if "diarrhea" in reason or "loose stool" in reason:
        if not re.search(r"\b(week|weeks|month|months|year|years)\b", text):
            warnings.append("Chronic diarrhea trigger: duration not documented.")
            draft.clinical.summary_symptoms = _append_missing(draft.clinical.summary_symptoms, "diarrhea duration")
        if "weight loss" not in text:
            warnings.append("Chronic diarrhea trigger: weight loss status missing.")
            draft.clinical.key_negatives_and_redflags = _append_missing(
                draft.clinical.key_negatives_and_redflags, "weight loss yes/no"
            )
        if "blood" not in text and "melena" not in text and "hematochezia" not in text:
            warnings.append("Chronic diarrhea trigger: blood in stool status missing.")
            draft.clinical.key_negatives_and_redflags = _append_missing(
                draft.clinical.key_negatives_and_redflags, "blood in stool yes/no"
            )

    if "dysphagia" in reason or "swallow" in reason or "food stuck" in reason:
        if "solid" not in text and "liquid" not in text:
            warnings.append("Dysphagia trigger: solids vs liquids not documented.")
            draft.clinical.summary_symptoms = _append_missing(
                draft.clinical.summary_symptoms, "solids vs liquids"
            )
        if "weight loss" not in text:
            warnings.append("Dysphagia trigger: weight loss status missing.")
            draft.clinical.key_negatives_and_redflags = _append_missing(
                draft.clinical.key_negatives_and_redflags, "weight loss yes/no"
            )
        if "odynophagia" not in text and "painful swallow" not in text:
            warnings.append("Dysphagia trigger: odynophagia status missing.")
            draft.clinical.key_negatives_and_redflags = _append_missing(
                draft.clinical.key_negatives_and_redflags, "odynophagia yes/no"
            )


def build_referral_draft(context: SessionContext, payload: Any, referrer_overrides: Optional[Dict[str, str]] = None) -> ReferralDraft:
    emr_text = (getattr(context.clinical_background, "emr_dump", None) or "").strip()
    netcare_text = (getattr(context.clinical_background, "netcare_dump", None) or "").strip()
    transcript = (getattr(context.transcript, "raw_text", None) or "").strip()

    name, phn, dob, _, _ = extract_demographics_from_text(emr_text or netcare_text)
    if not name:
        name = getattr(context.patient_anchor, "name", "") or ""
    if not phn:
        phn = getattr(context.patient_anchor, "phn", "") or ""
    if not dob:
        dob = getattr(context.patient_anchor, "dob", "") or ""

    phone = _extract_phone(emr_text)
    address = _extract_address(emr_text)

    specialty_override = (getattr(payload, "specialty", "") or "").strip()
    summary = _summarize_from_transcript_and_emr(transcript, emr_text, netcare_text, specialty_override)
    reason_short = (getattr(payload, "reason_short", "") or "").strip() or summary.get("reason_short", "")
    consult_question = (getattr(payload, "consult_question", "") or "").strip() or summary.get("consult_question", "")

    specialty_name = specialty_override or summary.get("specialty_name", "")
    subspecialty = (getattr(payload, "subspecialty_or_clinic", "") or "").strip() or summary.get("subspecialty_or_clinic", "")
    urgency_override = (getattr(payload, "urgency_override", None) or "").strip()
    include_objective = True

    urgency_label, urgency_rationale = _suggest_urgency(f"{transcript}\n{emr_text}\n{netcare_text}")
    if urgency_override in {"Routine", "Semi-urgent", "Urgent"}:
        urgency_label = urgency_override
        urgency_rationale = f"Clinician override selected. {urgency_rationale}"

    patient_aware = _infer_patient_aware(transcript)

    patient = PatientInfo(
        full_name=name or "",
        dob=dob or "",
        phn=phn or "",
        phone=phone or "",
        address=address or "",
    )

    overrides = referrer_overrides or {}
    referrer = ReferrerInfo(
        name=overrides.get("signature_name") or os.getenv("REFERRER_NAME", "Dr. Rajat Thapa"),
        credentials=os.getenv("REFERRER_CREDENTIALS", "MD PhD CCFP"),
        cpsa=os.getenv("REFERRER_CPSA", "032698"),
        clinic_name=overrides.get("clinic_name") or os.getenv("REFERRER_CLINIC_NAME", ""),
        clinic_address=overrides.get("clinic_address") or os.getenv("REFERRER_CLINIC_ADDRESS", ""),
        phone=overrides.get("clinic_phone") or os.getenv("REFERRER_PHONE", ""),
        fax=overrides.get("clinic_fax") or os.getenv("REFERRER_FAX", ""),
        fax_or_econsult_inbox=os.getenv("REFERRER_RETURN", ""),
        signature_block=overrides.get("signature_name") or os.getenv("REFERRER_SIGNATURE", ""),
    )

    referral = ReferralInfo(
        specialty=specialty_name,
        specialty_name=specialty_name,
        subspecialty_or_clinic=subspecialty,
        reason_short=reason_short,
        consult_question=consult_question,
        urgency_label=urgency_label,
        target_timeframe=summary.get("target_timeframe", ""),
        urgency_rationale=urgency_rationale,
        patient_aware_yes_no=patient_aware,
    )

    clinical = ClinicalBlock(
        summary_symptoms=summary.get("summary_symptoms", ""),
        key_positives=summary.get("key_positives", ""),
        key_negatives_and_redflags=summary.get("key_negatives_and_redflags", ""),
        pertinent_exam=summary.get("pertinent_exam", ""),
    )

    management = ManagementBlock(
        tried_block=summary.get("treatments_tried", ""),
        pending_block=summary.get("pending_items", ""),
    )

    assessment = AssessmentBlock(
        working_dx_and_ddx=summary.get("working_dx_and_ddx", ""),
    )

    background = BackgroundBlock(
        pmHx_relevant=_extract_section_block(emr_text, ["Health Profile", "PMHx", "Past Medical History"], max_lines=12),
        psHx_relevant=_extract_section_block(emr_text, ["PSHx", "Past Surgical History", "Surgical History"], max_lines=8),
        meds_relevant=_extract_section_block(emr_text, ["Medications"], max_lines=14),
        allergies=_extract_section_block(emr_text, ["Allergies", "Allergies/Intolerances"], max_lines=6),
    )

    labs_block = _extract_section_block(emr_text, ["Labs", "Laboratory", "Calculated Results"], max_lines=10)
    imaging_block = _extract_section_block(emr_text, ["Investigations", "Imaging"], max_lines=10)
    pathology_block = _extract_section_block(emr_text, ["Pathology", "Biopsy"], max_lines=6)

    results_location = ""
    if include_objective:
        has_attach = bool(getattr(context, "attachments", None))
        results_location = "Netcare + Attached" if has_attach else "Netcare"
    else:
        labs_block = "Not included (per clinician choice)."
        imaging_block = "Not included (per clinician choice)."
        pathology_block = "Not included (per clinician choice)."
        results_location = "Netcare"

    if include_objective and not (labs_block or imaging_block or pathology_block):
        labs_block = ""

    objective = ObjectiveBlock(
        results_location=results_location,
        labs_block=labs_block,
        imaging_block=imaging_block,
        pathology_block=pathology_block,
    )

    attachments_items: List[AttachmentItem] = []
    list_lines: List[str] = []
    for item in (getattr(context, "attachments", None) or []):
        label = item.filename
        date_str = item.uploaded_at.date().isoformat() if getattr(item, "uploaded_at", None) else ""
        attachments_items.append(AttachmentItem(label=label, source="Uploaded", date=date_str))
        list_lines.append(f"{label} (Uploaded{', ' + date_str if date_str else ''})")

    if include_objective and not attachments_items:
        if labs_block and labs_block != "No investigations to date.":
            list_lines.append("Labs: Available in Netcare")
        if imaging_block:
            list_lines.append("Imaging/Procedures: Available in Netcare")
        if pathology_block:
            list_lines.append("Pathology: Available in Netcare")

    attachments = AttachmentsBlock(
        items=attachments_items,
        list_block="\n".join(list_lines).strip() or "None.",
    )

    logistics = LogisticsBlock(
        high_risk_context=_collect_high_risk_context(f"{emr_text}\n{netcare_text}"),
        barriers="",
        patient_goals=summary.get("patient_goals", ""),
    )

    meta = ReferralMeta(
        generated_at=datetime.now(EDMONTON_TZ).isoformat(timespec="seconds"),
    )

    quality = QualityBlock()
    draft = ReferralDraft(
        meta=meta,
        patient=patient,
        referrer=referrer,
        referral=referral,
        clinical=clinical,
        objective=objective,
        management=management,
        assessment=assessment,
        background=background,
        logistics=logistics,
        attachments=attachments,
        safety=SafetyBlock(),
        quality=quality,
    )

    missing_critical = []
    if not draft.referral.specialty_name:
        missing_critical.append("referral.specialty_name")
    if not draft.referral.reason_short:
        missing_critical.append("referral.reason_short")
    if not draft.referral.consult_question:
        missing_critical.append("referral.consult_question")
    if not draft.referral.urgency_rationale:
        missing_critical.append("referral.urgency_rationale")
    if not draft.objective.results_location:
        missing_critical.append("objective.results_location")

    missing_recommended = []
    if not draft.patient.phone:
        missing_recommended.append("patient.phone")
    if not draft.background.meds_relevant:
        missing_recommended.append("background.meds_relevant")
    if not draft.background.allergies:
        missing_recommended.append("background.allergies")
    if not (draft.objective.labs_block or draft.objective.imaging_block or draft.objective.pathology_block):
        missing_recommended.append("objective.data_or_no_investigations_statement")

    warnings: List[str] = []
    if not include_objective:
        warnings.append("Objective data excluded by clinician selection.")

    _detect_minimum_dataset(draft, warnings)

    quality.missing_critical = missing_critical
    quality.missing_recommended = missing_recommended
    quality.warnings = warnings
    if missing_critical:
        quality.confidence = "Low"
    elif missing_recommended or warnings:
        quality.confidence = "Medium"
    else:
        quality.confidence = "High"

    return draft


def _display(value: str, label: str) -> str:
    v = (value or "").strip()
    if v:
        return v
    return f"[MISSING: {label}]"


def render_referral_letter(draft: ReferralDraft) -> str:
    r = draft.referral
    p = draft.patient
    ref = draft.referrer
    c = draft.clinical
    o = draft.objective
    m = draft.management
    a = draft.assessment
    b = draft.background
    l = draft.logistics
    ref_line = _display(r.specialty_name, "specialty")
    if (r.subspecialty_or_clinic or "").strip():
        ref_line = f"{ref_line} ({r.subspecialty_or_clinic})"

    lines = [
        f"REFERRAL TO: {ref_line}",
        f"DATE: {_display(draft.meta.generated_at.split('T')[0], 'date')}",
        "",
        f"PATIENT: {_display(p.full_name, 'patient name')} | DOB: {_display(p.dob, 'DOB')} | PHN: {_display(p.phn, 'PHN')}",
        f"CONTACT: {_display(p.phone, 'phone')} | ADDRESS: {_display(p.address, 'address')}",
        f"LANGUAGE / INTERPRETER: {_display(p.language, 'language')} / {_display(p.interpreter_needed, 'interpreter needed')}",
    ]

    if (p.guardian_or_sdm or "").strip():
        lines.append(f"GUARDIAN / SDM: {p.guardian_or_sdm}")

    lines.extend(
        [
            "",
            f"REFERRING CLINICIAN: {_display(ref.name, 'referrer name')} ({_display(ref.credentials, 'credentials')}) | CPSA: {_display(ref.cpsa, 'CPSA')}",
            f"CLINIC: {_display(ref.clinic_name, 'clinic name')} | ADDRESS: {_display(ref.clinic_address, 'clinic address')} | PHONE: {_display(ref.phone, 'clinic phone')} | FAX: {_display(ref.fax, 'clinic fax')}",
            f"RETURN REPORT TO: {_display(ref.fax_or_econsult_inbox, 'return inbox')}",
            "",
            "1) REFERRAL INTENT",
            f"Reason for referral: {_display(r.reason_short, 'reason for referral')}",
            f"Specific request / question: {_display(r.consult_question, 'consult question')}",
            f"Urgency: {_display(r.urgency_label, 'urgency')} — {_display(r.urgency_rationale, 'urgency rationale')}",
            f"Patient aware of referral reason: {_display(r.patient_aware_yes_no, 'patient aware')}",
            "",
            "2) CLINICAL SUMMARY",
            f"Presenting symptoms: {_display(c.summary_symptoms, 'presenting symptoms')}",
            f"Key positives: {_display(c.key_positives, 'key positives')}",
            f"Key negatives / red flags: {_display(c.key_negatives_and_redflags, 'key negatives/red flags')}",
            f"Pertinent exam: {_display(c.pertinent_exam, 'pertinent exam')}",
            "",
            "3) OBJECTIVE DATA",
            "Pertinent labs:",
            _display(o.labs_block, "labs"),
            "",
            "Pertinent imaging/procedures:",
            _display(o.imaging_block, "imaging/procedures"),
            "",
            "Pathology:",
            _display(o.pathology_block, "pathology"),
            "",
            f"Availability of results: {_display(o.results_location, 'results location')}",
            "",
            "4) MANAGEMENT TO DATE",
            f"Treatments tried: {_display(m.tried_block, 'treatments tried')}",
            f"Pending investigations/referrals: {_display(m.pending_block, 'pending investigations/referrals')}",
            f"Working diagnosis / differential: {_display(a.working_dx_and_ddx, 'working diagnosis/differential')}",
            "",
            "5) RELEVANT BACKGROUND",
            f"PMHx relevant: {_display(b.pmHx_relevant, 'PMHx')}",
            f"PSHx relevant: {_display(b.psHx_relevant, 'PSHx')}",
            f"Medications: {_display(b.meds_relevant, 'medications')}",
            f"Allergies/intolerances: {_display(b.allergies, 'allergies')}",
            "",
            "6) CONTEXT / LOGISTICS",
            f"Comorbidities affecting care: {_display(l.high_risk_context, 'high-risk context')}",
        ]
    )

    if (l.barriers or "").strip():
        lines.append(f"Barriers: {l.barriers}")
    if (l.patient_goals or "").strip():
        lines.append(f"Patient goals / expectations: {l.patient_goals}")

    lines.extend(
        [
            "",
            "7) SAFETY",
            _display(draft.safety.advice_line, "safety advice"),
            "",
            "Thank you for assessing. Please advise on diagnosis and management, and whether you recommend assuming ongoing specialty follow-up.",
            "",
            "Sincerely,",
            (ref.signature_block or ""),
        ]
    )
    return "\n".join(lines).strip()
