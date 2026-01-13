from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from pypdf import PdfReader


@dataclass
class CommentRecord:
    comment_text: str
    post_url: str | None = None
    comment_url: str | None = None
    posted_from: str | None = None
    feedback: str | None = None


def _clean_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    return s


def load_comments_from_xlsx(xlsx_path: Path) -> List[CommentRecord]:
    df = pd.read_excel(xlsx_path)

    def safe_str(x) -> str:
        return "" if pd.isna(x) else str(x)

    out: List[CommentRecord] = []
    for _, row in df.iterrows():
        mention = _clean_text(safe_str(row.get("Mention text")))
        if not mention:
            continue

        post_url = safe_str(row.get("Reddit Thread")).strip() or None
        comment_url = safe_str(row.get("Link to the mention")).strip() or None
        posted_from = safe_str(row.get("Posted from account")).strip() or None
        feedback = safe_str(row.get("Feedback from Darko")).strip() or None

        if post_url and not post_url.startswith("http"):
            post_url = None
        if comment_url and not comment_url.startswith("http"):
            comment_url = None

        out.append(
            CommentRecord(
                comment_text=mention,
                post_url=post_url,
                comment_url=comment_url,
                posted_from=posted_from,
                feedback=feedback,
            )
        )

    seen = set()
    deduped: List[CommentRecord] = []
    for r in out:
        key = re.sub(r"\s+", " ", r.comment_text).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    return deduped


def load_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return _clean_text("\n\n".join(pages))
