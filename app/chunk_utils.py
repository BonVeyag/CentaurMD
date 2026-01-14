import re
from typing import List, Tuple


def chunk_text(text: str, min_size: int = 500, max_size: int = 1200) -> List[str]:
    blocks = re.split(r"\n\s*\n", text.strip())
    chunks: List[str] = []
    current = ""
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) >= max_size:
            # hard split by sentence if needed
            parts = re.split(r"(?<=[.!?])\s+", block)
            for part in parts:
                if not part.strip():
                    continue
                if len(part) > max_size:
                    chunks.append(part.strip())
                else:
                    if len(current) + len(part) + 1 > max_size and current:
                        chunks.append(current.strip())
                        current = part
                    else:
                        current = f"{current}\n{part}" if current else part
            continue
        if len(current) + len(block) + 1 <= max_size:
            current = f"{current}\n{block}" if current else block
        else:
            if current:
                chunks.append(current.strip())
            current = block
        if len(current) >= min_size:
            chunks.append(current.strip())
            current = ""
    if current.strip():
        chunks.append(current.strip())
    return chunks


def group_code_rows(text: str) -> List[str]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    grouped: List[str] = []
    current = []
    for ln in lines:
        if re.match(r"^[0-9]{2}\.[0-9]{2}[A-Z]?", ln):
            if current:
                grouped.append("\n".join(current).strip())
                current = []
        current.append(ln)
    if current:
        grouped.append("\n".join(current).strip())
    return grouped
