from __future__ import annotations

from typing import Any, Dict, List, Optional

from .schema import SoapStructured


def _strip_bullets(line: str) -> str:
    raw = (line or "").lstrip()
    if raw.startswith("- "):
        return raw[2:].strip()
    if raw.startswith("• "):
        return raw[2:].strip()
    if raw.startswith("•"):
        return raw[1:].strip()
    return raw.strip()


def _filter_placeholders(lines: List[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        clean = _strip_bullets(ln)
        if not clean:
            continue
        low = clean.lower()
        if low in {"none", "n/a"}:
            continue
        if "not mentioned" in low or "not documented" in low or "not discussed" in low:
            continue
        out.append(clean)
    return out


def _item_to_dict(item: Any) -> Dict:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "dict"):
        return item.dict()
    return item


def _render_issue_lines(items: List[Dict], issue_map: Dict[int, int]) -> List[str]:
    out: List[str] = []
    for item in items:
        item = _item_to_dict(item)
        try:
            issue_num = int(item.get("issue_number", 0))
        except Exception:
            issue_num = 0
        display_num = issue_map.get(issue_num, issue_num or 0)
        for line in item.get("lines", []) or []:
            clean = _strip_bullets(line)
            if not clean:
                continue
            if display_num > 0:
                out.append(f"{display_num}. {clean}")
            else:
                out.append(clean)
    return _filter_placeholders(out)


def render_soap(soap: SoapStructured) -> str:
    issues = soap.issues or []
    issue_map: Dict[int, int] = {}
    rendered_issues: List[str] = []
    for idx, issue in enumerate(issues, start=1):
        issue_map[int(issue.number)] = idx
        title = _strip_bullets(issue.title)
        if title:
            rendered_issues.append(f"{idx}. {title}")

    subjective_lines = _render_issue_lines(list(soap.subjective or []), issue_map)
    assessment_lines = _render_issue_lines(list(soap.assessment or []), issue_map)
    plan_lines = _render_issue_lines(list(soap.plan or []), issue_map)

    safety_lines: List[str] = []
    if isinstance(soap.safety_red_flags, str):
        safety_lines = ["none"] if soap.safety_red_flags.strip().lower() == "none" else [soap.safety_red_flags.strip()]
    else:
        safety_lines = _filter_placeholders(list(soap.safety_red_flags or []))
    if not safety_lines:
        safety_lines = ["none"]

    social_lines = _filter_placeholders(list(soap.social_hx or []))
    objective_lines = _filter_placeholders(list(soap.objective or []))

    procedure_lines: Optional[List[str]] = None
    if soap.procedure and soap.procedure.lines:
        procedure_lines = _filter_placeholders(list(soap.procedure.lines or []))
        if not procedure_lines:
            procedure_lines = None

    sections: List[str] = []

    sections.append("\n".join(["**Issues:**"] + rendered_issues))
    sections.append("\n".join(["**Subjective:**"] + subjective_lines))
    sections.append("\n".join(["**Safety / Red Flags:**"] + safety_lines))
    sections.append("\n".join(["**Social Hx:**"] + social_lines))
    sections.append("\n".join(["**Objective:**"] + objective_lines))
    sections.append("\n".join(["**Assessment:**"] + assessment_lines))
    if procedure_lines is not None:
        sections.append("\n".join(["**Procedure:**"] + procedure_lines))
    sections.append("\n".join(["**Plan:**"] + plan_lines))

    return "\n\n".join(sections).strip()
