from langchain_chroma import Chroma
from langchain_core.documents import Document
from ingestion.embedder import get_embedder
import os
import time

PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chroma_db"
)

_vector_store_instance = None

RELEVANCE_THRESHOLD = None


def get_vector_store() -> Chroma:
    """
    Returns a singleton Chroma vector store instance,
    persisted to disk at PERSIST_DIR so the knowledge base
    survives across restarts.
    """
    global _vector_store_instance

    if _vector_store_instance is None:
        print(f"\n[STORE] opening chroma at {PERSIST_DIR}")
        _vector_store_instance = Chroma(
            collection_name="knowledge_base",
            embedding_function=get_embedder(),
            persist_directory=PERSIST_DIR
        )
        print(f"[STORE] collection=knowledge_base ready")

    return _vector_store_instance


def add_documents(chunks: list[Document]) -> list[str]:
    """
    Embeds and stores a list of chunked Documents into the vector store.
    Returns the list of generated ids for those chunks.
    """
    print(f"\n[STORE] embedding and storing {len(chunks)} chunks...")
    start = time.time()

    store = get_vector_store()
    ids = store.add_documents(chunks)

    print(f"[STORE] stored {len(ids)} chunks in {time.time() - start:.2f}s")
    for i, (cid, c) in enumerate(zip(ids, chunks)):
        print(f"[STORE]   [{i}] id={cid} meta={c.metadata}")

    return ids


def search(query: str, k: int = 4) -> list[Document]:
    """
    Searches the knowledge base for the k most relevant chunks.
    Logs the distance score for each so a threshold can be calibrated.
    """
    print(f"\n[SEARCH] query=\"{query}\"  k={k}")
    start = time.time()

    store = get_vector_store()
    scored = store.similarity_search_with_score(query, k=k)

    print(f"[SEARCH] returned {len(scored)} chunks in {time.time() - start:.3f}s")

    kept = []
    for i, (doc, score) in enumerate(scored, start=1):
        if RELEVANCE_THRESHOLD is None:
            status = "----"
            keep = True
        else:
            keep = score <= RELEVANCE_THRESHOLD
            status = "KEEP" if keep else "DROP"

        preview = doc.page_content[:60].replace("\n", " ")
        print(f"[SEARCH]   [{i}] {status} distance={score:.4f}")
        print(f"[SEARCH]       metadata={doc.metadata}")
        print(f"[SEARCH]       text=\"{preview}...\"")

        if keep:
            kept.append(doc)

    if RELEVANCE_THRESHOLD is not None:
        print(f"[SEARCH] kept {len(kept)}/{len(scored)} "
              f"(threshold={RELEVANCE_THRESHOLD})")

    return kept