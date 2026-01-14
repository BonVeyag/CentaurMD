import os
from typing import List, Tuple

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import PyPDF2  # type: ignore
except Exception:
    PyPDF2 = None


def extract_pdf_pages(path: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    if fitz is not None:
        try:
            with fitz.open(path) as doc:  # type: ignore
                for i, page in enumerate(doc):
                    text = page.get_text("text") or ""
                    pages.append((i + 1, text))
                return pages
        except Exception:
            pass
    if PyPDF2 is not None:
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    pages.append((i + 1, text))
        except Exception:
            pass
    return pages
