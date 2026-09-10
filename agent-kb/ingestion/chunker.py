from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[Document]:
    """
    Splits documents into smaller overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    print(f"\n[CHUNKER] chunk_size={chunk_size} overlap={chunk_overlap}")
    print(f"[CHUNKER] input documents: {len(documents)}")

    all_chunks = []
    for i, doc in enumerate(documents):
        doc_chunks = splitter.split_documents([doc])
        all_chunks.extend(doc_chunks)

        page = doc.metadata.get("page", "n/a")
        source = doc.metadata.get("source", "unknown")

        print(f"[CHUNKER]   doc[{i}] (page={page}) "
              f"{len(doc.page_content)} chars -> {len(doc_chunks)} chunks")
        for j, c in enumerate(doc_chunks):
            preview = c.page_content[:60].replace("\n", " ")
            print(f"[CHUNKER]       chunk[{j}] {len(c.page_content)} chars "
                  f"| meta={c.metadata} | \"{preview}...\"")

    print(f"[CHUNKER] total chunks: {len(all_chunks)}")
    if all_chunks:
        avg = sum(len(c.page_content) for c in all_chunks) // len(all_chunks)
        print(f"[CHUNKER] avg chunk size: {avg} chars")

    return all_chunks