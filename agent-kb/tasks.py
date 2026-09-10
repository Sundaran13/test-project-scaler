import os
import time
from datetime import datetime

from celery_app import celery_app
from ingestion.loader import load_document, load_raw_text
from ingestion.chunker import chunk_documents
from knowledge_base.vector_store import add_documents


@celery_app.task(name="ingest_file_task")
def ingest_file_task(file_path: str, original_filename: str) -> dict:
    start = time.time()
    print("\n" + "=" * 70)
    print(f"[INGEST] START  file={original_filename}")
    print(f"[INGEST] start_time={datetime.now().isoformat()}")
    print("=" * 70)

    docs = load_document(file_path)
    t_load = time.time()

    chunks = chunk_documents(docs)
    t_chunk = time.time()

    ids = add_documents(chunks)
    t_embed = time.time()

    os.remove(file_path)

    print("\n" + "=" * 70)
    print(f"[INGEST] END    file={original_filename}")
    print(f"[INGEST] end_time={datetime.now().isoformat()}")
    print(f"[INGEST] documents={len(docs)}  chunks={len(chunks)}")
    print(f"[INGEST] timing  load={t_load - start:.2f}s  "
          f"chunk={t_chunk - t_load:.2f}s  embed={t_embed - t_chunk:.2f}s")
    print(f"[INGEST] TOTAL={time.time() - start:.2f}s")
    print("=" * 70 + "\n")

    return {
        "filename": original_filename,
        "chunks_added": len(chunks),
        "ids": ids
    }


@celery_app.task(name="ingest_text_task")
def ingest_text_task(text: str, source_name: str) -> dict:
    start = time.time()
    print("\n" + "=" * 70)
    print(f"[INGEST] START  source={source_name}")
    print(f"[INGEST] start_time={datetime.now().isoformat()}")
    print("=" * 70)

    docs = load_raw_text(text, source_name)
    t_load = time.time()

    chunks = chunk_documents(docs)
    t_chunk = time.time()

    ids = add_documents(chunks)
    t_embed = time.time()

    print("\n" + "=" * 70)
    print(f"[INGEST] END    source={source_name}")
    print(f"[INGEST] end_time={datetime.now().isoformat()}")
    print(f"[INGEST] documents={len(docs)}  chunks={len(chunks)}")
    print(f"[INGEST] timing  load={t_load - start:.2f}s  "
          f"chunk={t_chunk - t_load:.2f}s  embed={t_embed - t_chunk:.2f}s")
    print(f"[INGEST] TOTAL={time.time() - start:.2f}s")
    print("=" * 70 + "\n")

    return {
        "source_name": source_name,
        "chunks_added": len(chunks),
        "ids": ids
    }