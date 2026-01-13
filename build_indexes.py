from pathlib import Path

from ingest import load_comments_from_xlsx, load_pdf_text
from chunking import chunk_text
from retrieval import Embedder, build_faiss_index, save_index, save_jsonl

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
INDEXES = BASE / "indexes"


def main():
    embedder = Embedder()

    comments = load_comments_from_xlsx(DATA / "KiloCode Comments.xlsx")
    comment_texts = [c.comment_text for c in comments]
    style_vecs = embedder.embed(comment_texts)
    save_index(build_faiss_index(style_vecs), INDEXES / "style.faiss")
    save_jsonl([c.__dict__ for c in comments], INDEXES / "style_meta.jsonl")

    doc_text = load_pdf_text(DATA / "KiloCode_Full_Documentation_Guide.pdf")
    chunks = chunk_text(doc_text)
    doc_vecs = embedder.embed([c.text for c in chunks])
    save_index(build_faiss_index(doc_vecs), INDEXES / "docs.faiss")
    save_jsonl([c.__dict__ for c in chunks], INDEXES / "docs_meta.jsonl")

    print("Indexes built successfully")


if __name__ == "__main__":
    main()
