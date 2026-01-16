from retrieval import search_by_name
from generation.prompt_builder import build_comment

def generate_comment(post, embedder, top_k_style=5, top_k_docs=5):
    query_text = f"{post.title}\n\n{post.content}".strip()

    style_examples = search_by_name(
        query=query_text,
        index_name="comments",
        embedder=embedder,
        top_k=top_k_style,
    )

    doc_facts = search_by_name(
        query=query_text,
        index_name="docs",
        embedder=embedder,
        top_k=top_k_docs,
    )

    return build_comment(
        post=post,
        style_examples=style_examples,
        doc_facts=doc_facts,
    )
