from pathlib import Path

from ingest import load_comments_from_xlsx, load_pdf_text
from chunking import chunk_text
from retrieval import Embedder, save_index

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

# Make sure data dir exists
DATA.mkdir(exist_ok=True)


def main():
    embedder = Embedder()

    # =========================
    # 1. COMMENT STYLE INDEX
    # =========================
    comments = load_comments_from_xlsx(DATA / "KiloCode Comments.xlsx")

    comment_texts = [c.comment_text for c in comments]
    comment_meta = [c.__dict__ for c in comments]

    comment_vectors = embedder.embed(comment_texts)

    save_index(
        vectors=comment_vectors,
        meta=comment_meta,
        name="comments"
    )

    # =========================
    # 2. DOCUMENTATION INDEX
    # =========================
    doc_text = load_pdf_text(DATA / "KiloCode_Full_Documentation_Guide.pdf")

    chunks = chunk_text(doc_text)
    chunk_texts = [c.text for c in chunks]
    chunk_meta = [c.__dict__ for c in chunks]

    doc_vectors = embedder.embed(chunk_texts)

    save_index(
        vectors=doc_vectors,
        meta=chunk_meta,
        name="docs"
    )

    print("Indexes built successfully (FAISS-free)")


if __name__ == "__main__":
    main()
