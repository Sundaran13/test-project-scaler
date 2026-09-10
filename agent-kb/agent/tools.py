import os
from langchain.tools import tool
from knowledge_base.vector_store import search


def format_source(metadata: dict) -> str:
    """
    Turns raw metadata into a readable citation.
    URLs pass through as-is; file paths get reduced to a filename,
    with the page number appended when available.
    """
    source = metadata.get("source", "unknown")
    page = metadata.get("page")

    if source.startswith("http"):
        return source

    name = os.path.basename(source)
    if page is not None:
        return f"{name}, page {page + 1}"
    return name


@tool
def retrieve_knowledge_base(query: str) -> str:
    """
    Searches the knowledge base for information relevant to the query.
    Use this whenever you need facts, context, or details to answer
    the user's question — do not answer from memory alone if the
    question is about content that may be in the knowledge base.
    """
    results = search(query, k=4)

    if not results:
        return "No relevant information found in the knowledge base."

    formatted = []
    for i, doc in enumerate(results, start=1):
        src = format_source(doc.metadata)
        formatted.append(f"[{i}] source: {src}\n{doc.page_content}")

    return "\n\n".join(formatted)