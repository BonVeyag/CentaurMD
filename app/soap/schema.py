from __future__ import annotations

from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class Issue(BaseModel):
    number: int
    title: str


class IssueLines(BaseModel):
    issue_number: int = Field(..., alias="issue_number")
    lines: List[str]


class ProcedureBlock(BaseModel):
    lines: List[str]


class SoapStructured(BaseModel):
    issues: List[Issue]
    subjective: List[IssueLines]
    safety_red_flags: List[str]
    social_hx: List[str]
    objective: List[str]
    assessment: List[IssueLines]
    procedure: Optional[ProcedureBlock] = None
    plan: List[IssueLines]


def soap_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "number": {"type": "integer"},
                        "title": {"type": "string"},
                    },
                    "required": ["number", "title"],
                },
            },
            "subjective": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "issue_number": {"type": "integer"},
                        "lines": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["issue_number", "lines"],
                },
            },
            "safety_red_flags": {"type": "array", "items": {"type": "string"}},
            "social_hx": {"type": "array", "items": {"type": "string"}},
            "objective": {"type": "array", "items": {"type": "string"}},
            "assessment": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "issue_number": {"type": "integer"},
                        "lines": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["issue_number", "lines"],
                },
            },
            "procedure": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "lines": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["lines"],
            },
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "issue_number": {"type": "integer"},
                        "lines": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["issue_number", "lines"],
                },
            },
        },
        "required": [
            "issues",
            "subjective",
            "safety_red_flags",
            "social_hx",
            "objective",
            "assessment",
            "plan",
        ],
    }
