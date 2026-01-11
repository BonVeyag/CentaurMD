from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.local_kb import search_guideline_graphs, get_guideline_graph


def _extract_basic_variables(text: str) -> Dict[str, Any]:
    t = (text or "")
    out: Dict[str, Any] = {}
    m = re.search(r"\b(\d{1,3})\s*(?:years|yrs|yr|y)\s*old\b", t, flags=re.IGNORECASE)
    if m:
        try:
            out["age_years"] = int(m.group(1))
        except Exception:
            pass
    if re.search(r"\bmale\b|\bman\b", t, flags=re.IGNORECASE):
        out["sex"] = "male"
    elif re.search(r"\bfemale\b|\bwoman\b", t, flags=re.IGNORECASE):
        out["sex"] = "female"
    if re.search(r"\bpregnan\w+\b", t, flags=re.IGNORECASE):
        out["pregnant"] = True
    return out


def _normalize_var_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (name or "").lower()).strip("_")


def _eval_logic(cond: Dict[str, Any], variables: Dict[str, Any]) -> Optional[bool]:
    if not cond:
        return None
    if "any_of" in cond:
        results = [_eval_logic(c, variables) for c in cond.get("any_of") or []]
        if any(r is True for r in results):
            return True
        if all(r is False for r in results if r is not None):
            return False
        return None
    if "all_of" in cond:
        results = [_eval_logic(c, variables) for c in cond.get("all_of") or []]
        if any(r is False for r in results):
            return False
        if all(r is True for r in results if r is not None):
            return True
        return None
    if "not" in cond:
        res = _eval_logic(cond.get("not") or {}, variables)
        return None if res is None else (not res)

    var = cond.get("var") or cond.get("variable")
    if not var:
        return None
    key = _normalize_var_name(var)
    if key not in variables:
        return None
    val = variables.get(key)
    op = (cond.get("op") or cond.get("operator") or "").lower()
    target = cond.get("value")
    try:
        if op in {"eq", "="}:
            return val == target
        if op in {"ne", "!="}:
            return val != target
        if op in {"gt", ">"}:
            return float(val) > float(target)
        if op in {"lt", "<"}:
            return float(val) < float(target)
        if op in {"gte", ">="}:
            return float(val) >= float(target)
        if op in {"lte", "<="}:
            return float(val) <= float(target)
        if op == "in":
            return val in (target or [])
    except Exception:
        return None
    return None


def _extract_condition_vars(cond: Dict[str, Any]) -> List[str]:
    if not cond:
        return []
    if "any_of" in cond:
        out: List[str] = []
        for c in cond.get("any_of") or []:
            out.extend(_extract_condition_vars(c))
        return out
    if "all_of" in cond:
        out = []
        for c in cond.get("all_of") or []:
            out.extend(_extract_condition_vars(c))
        return out
    if "not" in cond:
        return _extract_condition_vars(cond.get("not") or {})
    var = cond.get("var") or cond.get("variable")
    return [str(var)] if var else []


def run_guideline_runner(query: str, context_text: str) -> Optional[Dict[str, Any]]:
    candidates = search_guideline_graphs(query, limit=3)
    if not candidates:
        return None

    selected = candidates[0]
    guideline_id = selected.get("guideline_id")
    graph = get_guideline_graph(guideline_id)
    if not graph:
        return None

    variables = _extract_basic_variables(context_text)
    nodes = {n.get("id"): n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])

    entry_nodes = [n for n in graph.get("nodes", []) if (n.get("type") == "entry")]
    start_nodes = entry_nodes if entry_nodes else (graph.get("nodes", [])[:1])

    decision_trace = []
    next_steps = []
    missing_inputs = []

    visited = set()
    queue = [n.get("id") for n in start_nodes if n.get("id")]

    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        node = nodes.get(node_id)
        if not node:
            continue
        decision_trace.append(
            {
                "node_id": node_id,
                "label": node.get("label", ""),
                "evidence_spans": node.get("evidence_spans", []),
            }
        )
        if node.get("type") in {"action", "investigation", "referral"}:
            next_steps.append(
                {
                    "action_text": (node.get("actions") or [node.get("label", "")])[0],
                    "node_id": node_id,
                    "evidence_spans": node.get("evidence_spans", []),
                }
            )
        for edge in [e for e in edges if e.get("from") == node_id]:
            cond_logic = edge.get("condition_logic")
            res = _eval_logic(cond_logic, variables)
            if res is True:
                queue.append(edge.get("to"))
                continue
            if res is False:
                continue
            cond_vars = _extract_condition_vars(cond_logic or {})
            for var in cond_vars:
                norm = _normalize_var_name(var)
                if norm not in variables:
                    missing_inputs.append({"variable": var, "question_to_user": f"Please clarify: {var}"})
            if not cond_vars and edge.get("condition_text"):
                missing_inputs.append(
                    {
                        "variable": "condition",
                        "question_to_user": edge.get("condition_text"),
                    }
                )

    return {
        "selected_guideline": {
            "guideline_id": guideline_id,
            "title": graph.get("title", ""),
            "jurisdiction": graph.get("jurisdiction", ""),
            "version_date": graph.get("version_date", ""),
            "source_url": graph.get("source_url", ""),
        },
        "decision_trace": decision_trace,
        "next_steps": next_steps,
        "missing_inputs": missing_inputs,
    }
