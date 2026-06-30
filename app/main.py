from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from rag import ask, search

app = FastAPI(title="PhilosophIA", version="1.0.0")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Question(BaseModel):
    query: str


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/ask")
def ask_endpoint(body: Question):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Question vide")
    try:
        result = ask(body.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
def search_endpoint(q: str, n: int = 5):
    return {"results": search(q, n)}


@app.get("/health")
def health():
    return {"status": "ok"}
