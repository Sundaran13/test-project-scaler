from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
import os


def load_document(file_path: str) -> list[Document]:
    """
    Loads a file from disk and returns a list of LangChain Document objects.
    Supports .txt and .pdf for now.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()

    print(f"\n[LOADER] file={os.path.basename(file_path)} ext={ext}")
    print(f"[LOADER] langchain documents created: {len(docs)}")
    for i, d in enumerate(docs):
        print(f"[LOADER]   doc[{i}] chars={len(d.page_content)} metadata={d.metadata}")

    return docs


def load_raw_text(text: str, source_name: str = "raw_input") -> list[Document]:
    """
    Wraps a plain text string into a LangChain Document.
    """
    docs = [Document(page_content=text, metadata={"source": source_name})]

    print(f"\n[LOADER] raw text source={source_name}")
    print(f"[LOADER] langchain documents created: {len(docs)}")
    print(f"[LOADER]   doc[0] chars={len(text)} metadata={docs[0].metadata}")

    return docs