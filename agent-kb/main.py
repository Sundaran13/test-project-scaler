from fastapi import FastAPI
from api.ingest_api import router as ingest_router
from api.ask_api import router as ask_router

app = FastAPI(title="Knowledge Base Agent API")

app.include_router(ingest_router)
app.include_router(ask_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}