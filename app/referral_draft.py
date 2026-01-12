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
REFERRAL_AUDIT_MODEL = os.getenv("REFERRAL_AUDIT_MODEL", REFERRAL_MODEL)


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
REFERRAL_AUDIT_TEMPERATURE = _float_env("REFERRAL_AUDIT_TEMPERATURE", 0.1)

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
    relevant_history: str = ""
    social_history: str = ""
    functional_impact: str = ""


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


_SUMMARY_KEYS = [
    "specialty_name",
    "subspecialty_or_clinic",
    "reason_short",
    "consult_question",
    "summary_symptoms",
    "key_positives",
    "key_negatives_and_redflags",
    "pertinent_exam",
    "relevant_history",
    "social_history",
    "functional_impact",
    "treatments_tried",
    "pending_items",
    "working_dx_and_ddx",
    "patient_goals",
    "target_timeframe",
    "objective_labs",
    "objective_imaging",
    "objective_pathology",
    "safety_advice",
]

_UNWANTED_PHRASES = [
    "report not available",
    "not available in emr",
    "date not listed",
    "not listed",
]

_SUMMARY_STOPWORDS = {
    "the", "and", "or", "of", "to", "with", "for", "in", "on", "at", "by",
    "a", "an", "is", "are", "was", "were", "be", "as", "from", "that", "this",
    "these", "those", "it", "its", "his", "her", "their", "patient", "pt",
}


def _normalize_list_text(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return ""
        try:
            data = json.loads(v)
            if isinstance(data, list):
                items = [str(x or "").strip() for x in data if str(x or "").strip()]
                return "; ".join(items)
        except Exception:
            parts = [p.strip().strip("\"'") for p in inner.split(",")]
            parts = [p for p in parts if p]
            return "; ".join(parts)
    return v


def _sanitize_summary_text(value: str) -> str:
    v = _normalize_list_text(value)
    if not v:
        return ""
    lowered = v.lower()
    if any(p in lowered for p in _UNWANTED_PHRASES):
        chunks = re.split(r"[;\n]+", v)
        keep = []
        for c in chunks:
            c_strip = c.strip()
            if not c_strip:
                continue
            if any(p in c_strip.lower() for p in _UNWANTED_PHRASES):
                continue
            keep.append(c_strip)
        v = "; ".join(keep)
    return v.strip()


def _text_supported(text: str, source: str) -> bool:
    if not text:
        return False
    src = (source or "").lower()
    if not src:
        return False
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    tokens = [t for t in tokens if t not in _SUMMARY_STOPWORDS]
    if not tokens:
        return False
    uniq = list(dict.fromkeys(tokens))
    matches = sum(1 for t in uniq if t in src)
    if len(uniq) <= 3:
        return matches >= 1
    if len(uniq) <= 6:
        return matches >= 2
    return matches >= 3


def _merge_audited(generator: Dict[str, str], audited: Dict[str, str], source: str) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for key in _SUMMARY_KEYS:
        gen_val = (generator.get(key) or "").strip()
        aud_val = (audited.get(key) or "").strip()
        if aud_val:
            merged[key] = aud_val
            continue
        if gen_val and _text_supported(gen_val, source):
            merged[key] = gen_val
            continue
        merged[key] = ""
    return merged


def _infer_specialty_from_text(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return ""
    rules = [
        (["colonoscopy", "fit positive", "positive fit", "bowel habit", "constipation", "diarrhea", "gi bleed", "melena", "hematochezia"], "Gastroenterology"),
        (["skin lesion", "mole", "rash", "eczema", "psoriasis"], "Dermatology"),
        (["chest pain", "palpitations", "atrial fibrillation", "cardiology", "tavi", "murmur"], "Cardiology"),
        (["asthma", "copd", "pft", "pulmonary", "respirology", "dyspnea"], "Respirology"),
        (["sinus", "nasal polyps", "otitis", "ent", "hearing loss", "tonsil"], "ENT"),
        (["depression", "anxiety", "bipolar", "ptsd", "adhd"], "Psychiatry"),
        (["seizure", "stroke", "tremor", "neurology", "neuro"], "Neurology"),
        (["knee", "hip", "shoulder", "fracture", "ortho", "msk"], "Orthopedics"),
        (["diabetes", "thyroid", "endocrine", "a1c", "insulin"], "Endocrinology"),
        (["ckd", "dialysis", "proteinuria", "nephrology"], "Nephrology"),
        (["hematuria", "bph", "prostate", "urology"], "Urology"),
        (["pregnancy", "prenatal", "obgyn", "gynecology"], "Obstetrics/Gynecology"),
        (["pain clinic", "chronic pain", "rfa", "facet", "si joint"], "Pain Clinic"),
        (["hematology", "anemia", "low ferritin", "macrocytosis"], "Hematology"),
        (["oncology", "cancer", "malignancy"], "Oncology"),
        (["rheumatoid", "lupus", "vasculitis", "rheumatology"], "Rheumatology"),
    ]
    for keywords, specialty in rules:
        if any(k in t for k in keywords):
            return specialty
    return ""


def _split_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0].strip()


def _pick_lines_with_keywords(text: str, keywords: List[str], limit: int = 3) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    lower_keywords = [k.lower() for k in keywords]
    for ln in _split_lines(text):
        lnl = ln.lower()
        if any(k in lnl for k in lower_keywords):
            out.append(ln)
        if len(out) >= limit:
            break
    return out


def _format_paragraph_from_lines(lines: List[str], max_chars: int = 420) -> str:
    if not lines:
        return ""
    cleaned = []
    for ln in lines:
        ln = ln.lstrip("•- ").strip()
        if not ln:
            continue
        cleaned.append(_first_sentence(ln))
    merged = "; ".join(dict.fromkeys(cleaned))
    if len(merged) > max_chars:
        merged = merged[: max_chars - 1].rstrip() + "…"
    return merged.strip()


def _fallback_summary_from_emr(transcript: str, emr_text: str, netcare_text: str) -> Dict[str, str]:
    bg = (emr_text or "").strip()
    tx = (transcript or "").strip()
    nc = (netcare_text or "").strip()
    patient_notes = _extract_section_block_last(bg, ["Patient Notes"], max_lines=200)
    clinical_notes = _extract_section_block_last(bg, ["Clinical Notes"], max_lines=80)
    plan_block = _extract_section_block_last(bg, ["Plan", "Plan (Problem-Based)", "A/P", "AP"], max_lines=50)
    assessment_block = _extract_section_block_last(bg, ["Assessment", "Impression"], max_lines=40)
    objective_block = _extract_section_block_last(bg, ["Objective Data", "Objective", "O/E", "Vitals"], max_lines=40)
    ros_block = _extract_section_block_last(bg, ["Review of Systems", "ROS"], max_lines=30)

    source = "\n".join([tx, patient_notes, clinical_notes, assessment_block, plan_block, objective_block, ros_block]).strip()
    if not source:
        return {}

    reason_lines = _pick_lines_with_keywords(
        source,
        ["presents", "presented", "presenting", "primary", "reason", "for", "complain", "reports", "evaluation of"],
        limit=2,
    )
    summary_symptoms = _format_paragraph_from_lines(reason_lines, max_chars=360)

    key_pos_lines = _pick_lines_with_keywords(
        source,
        ["positive", "noted", "significant", "history of", "change", "abnormal", "pressure", "constipation", "pain"],
        limit=3,
    )
    key_positives = _format_paragraph_from_lines(key_pos_lines, max_chars=360)

    key_neg_lines = _pick_lines_with_keywords(
        source,
        ["no ", "denies", "without", "not reported", "negative"],
        limit=3,
    )
    key_negatives = _format_paragraph_from_lines(key_neg_lines, max_chars=260)

    exam_lines = _pick_lines_with_keywords(
        "\n".join([objective_block, source]),
        ["exam", "ultrasound", "bp", "vitals", "o/e", "tender", "hydronephrosis", "fecal loading"],
        limit=3,
    )
    pertinent_exam = _format_paragraph_from_lines(exam_lines, max_chars=280)

    treatments = _format_paragraph_from_lines(_split_lines(plan_block), max_chars=420)
    working_dx = _format_paragraph_from_lines(_split_lines(assessment_block), max_chars=360)

    labs_lines = _pick_lines_with_keywords(
        "\n".join([patient_notes, clinical_notes, bg]),
        ["a1c", "ldl", "hdl", "creatinine", "egfr", "acr", "hemoglobin", "mcv", "ferritin", "c-reactive", "crp"],
        limit=4,
    )
    imaging_lines = _pick_lines_with_keywords(
        "\n".join([patient_notes, clinical_notes, bg]),
        ["ecg", "echo", "ultrasound", "ct", "mri", "x-ray", "pft", "scan", "biopsy"],
        limit=4,
    )

    return {
        "reason_short": summary_symptoms,
        "summary_symptoms": summary_symptoms,
        "key_positives": key_positives,
        "key_negatives_and_redflags": key_negatives,
        "pertinent_exam": pertinent_exam,
        "treatments_tried": treatments,
        "pending_items": "",
        "working_dx_and_ddx": working_dx,
        "objective_labs": _format_paragraph_from_lines(labs_lines, max_chars=360),
        "objective_imaging": _format_paragraph_from_lines(imaging_lines, max_chars=360),
    }


def _clean_summary_dict(data: Dict[str, Any]) -> Dict[str, str]:
    cleaned: Dict[str, str] = {}
    for key in _SUMMARY_KEYS:
        raw = data.get(key, "")
        cleaned[key] = _sanitize_summary_text(str(raw or ""))
    return cleaned


def _audit_referral_summary(
    draft: Dict[str, str],
    transcript: str,
    emr_text: str,
    netcare_text: str,
    focus_text: str,
) -> Dict[str, str]:
    if not draft:
        return {}
    draft_json = json.dumps({k: draft.get(k, "") for k in _SUMMARY_KEYS})
    prompt = f"""
You are a medical documentation auditor.
Return strict JSON with the SAME keys as the draft.

Rules:
- Keep statements ONLY if explicitly supported by transcript or EMR/Netcare.
- Do NOT add new facts.
- If a field contains unsupported content, delete it (empty string).
- Do NOT include phrases like "not documented", "date not listed", "report not available".

TRANSCRIPT:
{_clip_text(transcript, max_chars=6000) or "[none]"}

EMR:
{_clip_text(emr_text, max_chars=4000) or "[none]"}

NETCARE:
{_clip_text(netcare_text, max_chars=3000) or "[none]"}

EMR FOCUS:
{focus_text or "[none]"}

DRAFT JSON:
{draft_json}
""".strip()
    try:
        resp = client.chat.completions.create(
            model=REFERRAL_AUDIT_MODEL,
            messages=[
                {"role": "system", "content": "Return strict JSON only. Remove unsupported content."},
                {"role": "user", "content": prompt},
            ],
            temperature=REFERRAL_AUDIT_TEMPERATURE,
            top_p=1.0,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        if isinstance(data, dict):
            return _clean_summary_dict(data)
    except Exception as exc:
        logger.warning("Referral audit failed: %s", exc)
    return _clean_summary_dict(draft)


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
relevant_history, social_history, functional_impact,
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
- relevant_history: only PMHx/PSHx/treatments clearly related to the referral reason.
- social_history: only lifestyle/occupation/habits explicitly stated and relevant.
- functional_impact: only if daily-life impact is explicitly stated.
- objective_labs: list lab name + value + units + date if available; if only test name/date, leave value empty.
- objective_imaging: list test + date + brief result summary if present; if report not present, leave empty.
- objective_pathology: only if pathology present; otherwise empty string.
- safety_advice: issue-specific red flags and what to do (urgent care/ED) based on the presenting concern.
- Do NOT add phrases like "date not listed", "report not available in EMR", or "not documented"—leave field empty instead.
- Output plain text only, not list syntax like ["..."].

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
            cleaned = _clean_summary_dict(data)
            audited = _audit_referral_summary(cleaned, t, bg, nc, focus_text)
            source_text = " ".join([t, bg, nc, focus_text])
            merged = _merge_audited(cleaned, audited, source_text)
            fallback = _fallback_summary_from_emr(t, bg, nc)
            for key, val in fallback.items():
                if not merged.get(key):
                    merged[key] = val
            if not merged.get("specialty_name"):
                merged["specialty_name"] = _infer_specialty_from_text(source_text)
            return merged
    except Exception as exc:
        logger.warning("Referral summary failed: %s", exc)
    return {}


def _suggest_urgency(text: str) -> Tuple[str, str]:
    t = (text or "").lower()
    if not t:
        return ("Routine", "No red flags identified; routine triage.")

    def has_positive(term: str) -> bool:
        term_re = re.escape(term)
        for m in re.finditer(term_re, t):
            start = max(0, m.start() - 24)
            window = t[start:m.start()]
            if re.search(r"\b(no|denies|without|negative for)\b", window):
                continue
            return True
        return False

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
        "uncontrolled",
        "suicidal",
        "self-harm",
        "acute vision loss",
        "seizure",
    ]
    urgent_hits = [term for term in urgent_terms if has_positive(term)]
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
    semi_hits = [term for term in semi_terms if has_positive(term)]
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

    if "diarrhea" in reason or "loose stool" in reason:
        if not re.search(r"\b(week|weeks|month|months|year|years)\b", text):
            warnings.append("Chronic diarrhea trigger: duration not documented.")
        if "weight loss" not in text:
            warnings.append("Chronic diarrhea trigger: weight loss status missing.")
        if "blood" not in text and "melena" not in text and "hematochezia" not in text:
            warnings.append("Chronic diarrhea trigger: blood in stool status missing.")

    if "dysphagia" in reason or "swallow" in reason or "food stuck" in reason:
        if "solid" not in text and "liquid" not in text:
            warnings.append("Dysphagia trigger: solids vs liquids not documented.")
        if "weight loss" not in text:
            warnings.append("Dysphagia trigger: weight loss status missing.")
        if "odynophagia" not in text and "painful swallow" not in text:
            warnings.append("Dysphagia trigger: odynophagia status missing.")


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
        cpsa=overrides.get("cpsa") or os.getenv("REFERRER_CPSA", "032698"),
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
        relevant_history=summary.get("relevant_history", ""),
        social_history=summary.get("social_history", "")
        or _extract_section_block(emr_text, ["Social History", "Social Hx"], max_lines=8),
        functional_impact=summary.get("functional_impact", ""),
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

    objective_labs = (summary.get("objective_labs", "") or "").strip()
    objective_imaging = (summary.get("objective_imaging", "") or "").strip()
    objective_pathology = (summary.get("objective_pathology", "") or "").strip()

    labs_block = objective_labs or _extract_section_block_last(
        emr_text, ["Labs", "Laboratory", "Calculated Results"], max_lines=16
    )
    imaging_block = objective_imaging or _extract_section_block_last(
        emr_text, ["Investigations", "Imaging"], max_lines=16
    )
    pathology_block = objective_pathology or _extract_section_block_last(
        emr_text, ["Pathology", "Biopsy"], max_lines=10
    )
    labs_block = _sanitize_summary_text(labs_block)
    imaging_block = _sanitize_summary_text(imaging_block)
    pathology_block = _sanitize_summary_text(pathology_block)

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

    attachments = AttachmentsBlock(items=[], list_block="")

    logistics = LogisticsBlock(
        high_risk_context=_collect_high_risk_context(f"{emr_text}\n{netcare_text}"),
        barriers="",
        patient_goals=summary.get("patient_goals", ""),
    )

    safety_text = (summary.get("safety_advice", "") or "").strip()
    if not safety_text:
        safety_text = "Seek urgent care/ED for worsening symptoms or new red-flag features related to the presenting concern."

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
        safety=SafetyBlock(advice_line=safety_text),
        quality=quality,
    )

    missing_critical = []
    if not draft.referral.specialty_name:
        missing_critical.append("referral.specialty_name")
    if not draft.referral.reason_short:
        missing_critical.append("referral.reason_short")
    if not draft.referral.urgency_label:
        missing_critical.append("referral.urgency_label")

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


def _format_dr_name(value: str, label: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return _display(raw, label)
    if raw.lower().startswith("dr "):
        return raw
    return f"Dr. {raw}"


def _bullet_lines(items: List[str]) -> List[str]:
    bullets = [f"- {item.strip()}" for item in items if (item or "").strip()]
    return bullets or ["- Not documented."]


def _bullets_from_text(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    lines = _split_lines(t)
    fragments = lines if len(lines) > 1 else _split_into_sentences(t)
    out: List[str] = []
    for frag in fragments:
        frag = (frag or "").strip()
        if not frag:
            continue
        out.append(_sentence_case(frag))
    return out


def _display_soft(value: str, fallback: str = "Not documented.") -> str:
    v = (value or "").strip()
    return v if v else fallback


def _normalize_summary_fragment(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(
        r"^(presenting symptoms|key positives?|key negatives? / red flags|key negatives?|red flags|pertinent exam|exam|objective)\s*[:\-]\s*",
        "",
        t,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", t).strip()


def _split_summary_fragments(text: str) -> List[str]:
    if not text:
        return []
    raw = re.split(r"[;\n]+", text)
    out = []
    for chunk in raw:
        norm = _normalize_summary_fragment(chunk)
        if norm:
            out.append(norm)
    return out


def _dedupe_fragments(fragments: List[str]) -> List[str]:
    deduped: List[str] = []
    for frag in fragments:
        frag_norm = frag.lower()
        replaced = False
        for i, existing in enumerate(deduped):
            ex_norm = existing.lower()
            if frag_norm == ex_norm or frag_norm in ex_norm:
                replaced = True
                break
            if ex_norm in frag_norm:
                deduped[i] = frag
                replaced = True
                break
        if not replaced:
            deduped.append(frag)
    return deduped


def _dedupe_against(base: List[str], candidates: List[str]) -> List[str]:
    if not base:
        return candidates[:]
    base_norm = [b.lower() for b in base]
    out: List[str] = []
    for c in candidates:
        c_norm = c.lower()
        if any(c_norm in b or b in c_norm for b in base_norm):
            continue
        out.append(c)
    return out


def _join_fragments(fragments: List[str]) -> str:
    if not fragments:
        return ""
    return "; ".join(fragments).strip()


def _sentence(prefix: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    text = f"{prefix}{body}"
    if not re.search(r"[.!?]$", text):
        text += "."
    return text


def _sentence_case(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if not re.search(r"[.!?]$", s):
        s += "."
    return s


def _split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"[;\n]+", text) if p.strip()]
    return [_sentence_case(p) for p in parts if p]


def _sentence_from_fragment(fragment: str, prefix: str) -> str:
    f = (fragment or "").strip()
    if not f:
        return ""
    if re.match(r"^(the patient|patient|he|she|they|no\b|denies\b|reports\b|exam\b|on exam\b|physical exam\b)", f, flags=re.IGNORECASE):
        return _sentence_case(f)
    return _sentence_case(f"{prefix}{f}")


def _build_clinical_summary_paragraph(c: ClinicalBlock) -> str:
    summary_frags = _dedupe_fragments(_split_summary_fragments(c.summary_symptoms))
    positives_frags = _dedupe_fragments(_split_summary_fragments(c.key_positives))
    negatives_frags = _dedupe_fragments(_split_summary_fragments(c.key_negatives_and_redflags))
    exam_frags = _dedupe_fragments(_split_summary_fragments(c.pertinent_exam))

    positives_frags = _dedupe_against(summary_frags, positives_frags)
    combined_frags = _dedupe_fragments(summary_frags + positives_frags)
    negatives_frags = _dedupe_against(combined_frags, negatives_frags)
    exam_frags = _dedupe_against(combined_frags + negatives_frags, exam_frags)

    sentences: List[str] = []
    for frag in combined_frags:
        sentences.append(_sentence_from_fragment(frag, "The patient reports "))
    for frag in negatives_frags:
        sentences.append(_sentence_from_fragment(frag, "Negatives include "))
    for frag in exam_frags:
        sentences.append(_sentence_from_fragment(frag, "Exam findings include "))

    paragraph = " ".join([s for s in sentences if s]).strip()
    return paragraph or "Not documented."


def _normalize_management_clause(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    low = t.lower()
    if low.startswith("start "):
        return "starting " + t[6:].strip()
    if low.startswith("continue "):
        return "continuing " + t[9:].strip()
    if low.startswith("stop "):
        return "stopping " + t[5:].strip()
    if low.startswith("trial "):
        return "a trial of " + t[6:].strip()
    if low.startswith("add "):
        return "adding " + t[4:].strip()
    return t


def _build_management_paragraph(m: ManagementBlock, a: AssessmentBlock) -> str:
    sentences: List[str] = []
    tried = (m.tried_block or "").strip()
    pending = (m.pending_block or "").strip()
    working = (a.working_dx_and_ddx or "").strip()

    if tried:
        tried_sentences = _split_into_sentences(tried)
        if tried_sentences:
            first = _normalize_management_clause(tried_sentences[0].rstrip("."))
            sentences.append(_sentence_case(f"Treatments to date include {first}"))
            for extra in tried_sentences[1:]:
                sentences.append(_sentence_case(_normalize_management_clause(extra.rstrip("."))))
    if pending:
        pending_sentences = _split_into_sentences(pending)
        if pending_sentences:
            first = pending_sentences[0].rstrip(".")
            sentences.append(_sentence_case(f"Pending investigations or referrals include {first}"))
            sentences.extend(pending_sentences[1:])
    if working:
        working_sentences = _split_into_sentences(working)
        if working_sentences:
            first = working_sentences[0].rstrip(".")
            sentences.append(_sentence_case(f"The working diagnosis/differential is {first}"))
            for extra in working_sentences[1:]:
                sentences.append(_sentence_case(f"Additional considerations include {extra.rstrip('.')}"))

    paragraph = " ".join(sentences).strip()
    return paragraph or "Not documented."


def _compact_medications_block(text: str) -> str:
    lines = _split_lines(text)
    if not lines:
        return ""
    entries: List[str] = []
    current: List[str] = []
    for ln in lines:
        is_continuation = bool(re.match(r"^(?:\d+|tab|tabs|tablet|caps?|po|inh|iv|im|sc|q\d|bid|tid|qid|prn)\b", ln.strip(), flags=re.IGNORECASE))
        if is_continuation and current:
            current.append(ln.strip())
            continue
        if current:
            entries.append(" ".join(current).strip())
        current = [ln.strip()]
    if current:
        entries.append(" ".join(current).strip())

    compact: List[str] = []
    for entry in entries:
        entry = re.sub(r"\s+", " ", entry).strip()
        if entry:
            compact.append(entry)
    return "\n".join(compact)


def _build_background_paragraph(b: BackgroundBlock) -> str:
    lines: List[str] = []
    if (b.pmHx_relevant or "").strip():
        lines.append(f"PMHx: {b.pmHx_relevant.strip()}")
    if (b.psHx_relevant or "").strip():
        lines.append(f"PSHx: {b.psHx_relevant.strip()}")
    if (b.meds_relevant or "").strip():
        meds = _compact_medications_block(b.meds_relevant)
        lines.append("Medications:")
        if meds:
            lines.append(meds)
        else:
            lines.append(b.meds_relevant.strip())
    if (b.allergies or "").strip():
        lines.append(f"Allergies/intolerances: {b.allergies.strip()}")
    return "\n".join(lines).strip() or "Not documented."


def _build_context_paragraph(l: LogisticsBlock) -> str:
    parts: List[str] = []
    if (l.high_risk_context or "").strip():
        parts.append(f"Relevant comorbidities affecting care include {l.high_risk_context.strip()}.")
    if (l.barriers or "").strip():
        parts.append(f"Barriers include {l.barriers.strip()}.")
    if (l.patient_goals or "").strip():
        parts.append(f"Patient goals include {l.patient_goals.strip()}.")
    return " ".join(parts).strip() or "Not documented."


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
    date_line = _display(draft.meta.generated_at.split("T")[0], "date")
    referrer_line = _format_dr_name(ref.name, "referring provider")
    specialist_raw = (r.specialty_name or "").strip()
    specialist_line = _format_dr_name(specialist_raw, "specialist name") if specialist_raw else _display("", "specialist name")
    patient_line = _display(p.full_name, "patient name")
    dob_line = _display(p.dob, "DOB")

    reason = (r.reason_short or "").strip() or (c.summary_symptoms or "").strip()
    reason_line = _display(reason, "reason for referral")

    relevant_history = (c.relevant_history or "").strip()
    if not relevant_history:
        relevant_history = " ".join([b.pmHx_relevant.strip(), b.psHx_relevant.strip(), m.tried_block.strip()]).strip()
    relevant_history_items = _bullets_from_text(relevant_history)

    social_history = (c.social_history or "").strip()
    social_items = _bullets_from_text(social_history)

    clinical_items: List[str] = []
    clinical_summary = _build_clinical_summary_paragraph(c)
    if clinical_summary and clinical_summary != "Not documented.":
        clinical_items.append(clinical_summary)
    if (o.labs_block or "").strip():
        clinical_items.append(f"Labs: {o.labs_block.strip()}")
    if (o.imaging_block or "").strip():
        clinical_items.append(f"Imaging: {o.imaging_block.strip()}")
    if (o.pathology_block or "").strip():
        clinical_items.append(f"Pathology: {o.pathology_block.strip()}")
    clinical_items = _bullets_from_text("\n".join(clinical_items)) if clinical_items else []

    impact_items = _bullets_from_text(c.functional_impact)

    goals_items: List[str] = []
    if (a.working_dx_and_ddx or "").strip():
        goals_items.append(f"Diagnostic considerations: {a.working_dx_and_ddx.strip()}")
    if (l.patient_goals or "").strip():
        goals_items.append(f"Desired outcomes: {l.patient_goals.strip()}")
    if (r.consult_question or "").strip():
        goals_items.append(f"Specific questions: {r.consult_question.strip()}")
    goals_items = _bullets_from_text("\n".join(goals_items)) if goals_items else []

    follow_up_interval = (r.target_timeframe or "").strip()
    follow_up_line = (
        f"I see the patient every {follow_up_interval}, and will discuss your recommendations accordingly."
        if follow_up_interval
        else "I will discuss your recommendations with the patient at follow-up."
    )

    signature = (ref.signature_block or ref.name or "").strip() or _display("", "signature name")

    lines = [
        f"Date: {date_line}",
        f"Referring Provider: {referrer_line}",
        f"Receiving Specialist: {specialist_line}",
        f"Patient: {patient_line}",
        f"Date of Birth: {dob_line}",
        "",
        "Reason for Referral",
        _sentence_case(reason_line),
        "",
        "Relevant Medical History",
        *_bullet_lines(relevant_history_items),
        "",
        "Social History",
        *_bullet_lines(social_items),
        "",
        "Clinical Evaluation Summary",
        *_bullet_lines(clinical_items),
        "Impact on daily life",
        *_bullet_lines(impact_items),
        "",
        "Goals of Referral",
        *_bullet_lines(goals_items),
        "",
        "Follow-Up and Communication",
        "- Please provide your consultation notes and recommendations.",
        f"- {follow_up_line}",
        "",
        "Thank you for your assistance in managing this patient.",
        "",
        signature,
    ]

    return "\n".join(lines).strip()
