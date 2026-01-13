from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class DocChunk:
    chunk_id: str
    text: str


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 150) -> List[DocChunk]:
    parts = re.split(r"\n(?=\d+\.\s)|\n(?=[A-Z][^\n]{0,60}\n)", text)
    parts = [p.strip() for p in parts if p and p.strip()]

    if len(parts) < 4:
        parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    merged: List[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)

    chunks: List[DocChunk] = []
    for i, m in enumerate(merged):
        if i > 0 and overlap > 0:
            prev = merged[i - 1]
            m = (prev[-overlap:] + "\n\n" + m).strip()
        chunks.append(DocChunk(chunk_id=f"doc_{i:04d}", text=m))

    return chunks
