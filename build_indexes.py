"""
Build vector indexes using Gemini embeddings (FREE tier).

IMPORTANT: This script now uses Gemini API embeddings instead of local models.
- Old model: all-MiniLM-L6-v2 (384 dimensions)
- New model: Gemini text-embedding-004 (768 dimensions)

This means existing indexes MUST be rebuilt after switching to Gemini.
The script requires GEMINI_API_KEY environment variable to be set.
"""
from pathlib import Path

from ingest import load_comments_from_xlsx, load_pdf_text
from chunking import chunk_text
from retrieval import Embedder, save_index

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

# Make sure data dir exists
DATA.mkdir(exist_ok=True)


def main():
    """
    Build indexes using Gemini embeddings.
    
    This will:
    1. Load comment examples from Excel
    2. Load documentation from PDF
    3. Embed all content using Gemini API (FREE tier)
    4. Save indexes to data/ directory
    
    NOTE: Requires GEMINI_API_KEY environment variable.
    """
    print("Building indexes with Gemini embeddings (FREE tier)...")
    print("Model: text-embedding-004 (768 dimensions)")
    
    embedder = Embedder()

    # =========================
    # 1. COMMENT STYLE INDEX
    # =========================
    print("\n[1/2] Building comment style index...")
    comments = load_comments_from_xlsx(DATA / "KiloCode Comments.xlsx")
    print(f"Loaded {len(comments)} comments")

    comment_texts = [c.comment_text for c in comments]
    comment_meta = [c.__dict__ for c in comments]

    print("Embedding comments via Gemini API...")
    comment_vectors = embedder.embed(comment_texts, batch_size=20)
    print(f"Generated embeddings: shape={comment_vectors.shape}")

    save_index(
        vectors=comment_vectors,
        meta=comment_meta,
        name="comments"
    )
    print("Comment index saved ✓")

    # =========================
    # 2. DOCUMENTATION INDEX
    # =========================
    print("\n[2/2] Building documentation index...")
    doc_text = load_pdf_text(DATA / "KiloCode_Full_Documentation_Guide.pdf")
    print(f"Loaded PDF: {len(doc_text)} characters")

    chunks = chunk_text(doc_text)
    chunk_texts = [c.text for c in chunks]
    chunk_meta = [c.__dict__ for c in chunks]
    print(f"Created {len(chunks)} chunks")

    print("Embedding documentation via Gemini API...")
    doc_vectors = embedder.embed(chunk_texts, batch_size=20)
    print(f"Generated embeddings: shape={doc_vectors.shape}")

    save_index(
        vectors=doc_vectors,
        meta=chunk_meta,
        name="docs"
    )
    print("Documentation index saved ✓")

    print("\n✅ Indexes built successfully!")
    print("   - No local model loading (zero RAM usage)")
    print("   - Using Gemini FREE tier API")
    print("   - Ready for Render deployment")


if __name__ == "__main__":
    main()
