import logging
import os
import sys

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
from rag import ask, search  # noqa: E402  (vérifie aussi GROQ_API_KEY au démarrage)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("philosophia")

app = FastAPI(title="PhilosophIA", version="1.1.0")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Question(BaseModel):
    query: str = Field(max_length=1000)


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/ask")
def ask_endpoint(body: Question):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Question vide")
    try:
        return ask(body.query)
    except Exception:
        log.exception("Erreur dans /ask")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne (base non construite ? clé Groq invalide ?). Voir les logs du serveur.",
        )


@app.get("/search")
def search_endpoint(q: str, n: int = Query(5, ge=1, le=20)):
    try:
        return {"results": search(q, n)}
    except Exception:
        log.exception("Erreur dans /search")
        raise HTTPException(status_code=500, detail="Erreur interne. Voir les logs du serveur.")


@app.get("/health")
def health():
    return {"status": "ok"}