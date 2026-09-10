from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from celery.result import AsyncResult
import shutil
import os
import uuid

from tasks import ingest_file_task, ingest_text_task
from celery_app import celery_app

router = APIRouter()

UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "uploads"
)
os.makedirs(UPLOAD_DIR, exist_ok=True)


class RawTextRequest(BaseModel):
    text: str
    source_name: str = "raw_input"


@router.post("/ingest/text")
def ingest_text(request: RawTextRequest):
    """
    Queues raw text for ingestion. Returns immediately with a task id.
    """
    task = ingest_text_task.delay(request.text, request.source_name)

    return {
        "status": "queued",
        "task_id": task.id
    }


@router.post("/ingest/file")
def ingest_file(file: UploadFile = File(...)):
    """
    Saves an uploaded file to disk, then queues it for ingestion.
    Returns immediately with a task id.
    """
    file_ext = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{file_ext}")

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task = ingest_file_task.delay(temp_path, file.filename)

    return {
        "status": "queued",
        "filename": file.filename,
        "task_id": task.id
    }


@router.get("/ingest/status/{task_id}")
def get_ingest_status(task_id: str):
    """
    Polls the status of a queued ingestion task.
    """
    result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "state": result.state
    }

    if result.successful():
        response["result"] = result.result
    elif result.failed():
        response["error"] = str(result.result)

    return response