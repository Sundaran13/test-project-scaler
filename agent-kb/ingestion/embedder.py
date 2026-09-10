from langchain_huggingface import HuggingFaceEmbeddings

_embedder_instance = None

def get_embedder() -> HuggingFaceEmbeddings:
    """
    Returns a singleton embedding model instance.
    Uses a small, free, local sentence-transformers model —
    no API key, no internet call after the first download,
    runs on CPU.
    """
    global _embedder_instance

    if _embedder_instance is None:
        _embedder_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return _embedder_instance