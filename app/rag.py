"""
Moteur RAG : recherche dans ChromaDB + génération via Groq (Llama 3).
"""
import logging
import os

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

log = logging.getLogger("philosophia")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY manquante. Exporte-la dans ton terminal : "
        "export GROQ_API_KEY=gsk_..."
    )

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "philosophia"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TRANSLATE_MODEL = os.getenv("GROQ_TRANSLATE_MODEL", "openai/gpt-oss-20b")  # léger et rapide, suffisant pour traduire
N_RESULTS = 6
# Distance cosinus (0 = identique, 1 = sans rapport). Au-delà, l'extrait est écarté.
# À calibrer avec /search : regarde les distances de bonnes et de mauvaises réponses, 0.65 semble etre le plus pertinent après tests
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.65"))

NO_ANSWER = (
    "Je ne trouve pas d'extraits suffisamment pertinents dans le corpus pour "
    "répondre à cette question. Essaie de la reformuler ou de citer un auteur ou un thème précis."
)

_groq = Groq()
_collection = None


def get_collection():
    global _collection
    if _collection is None:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)
    return _collection


def to_english(query: str) -> str:
    """Traduit la question en anglais (le corpus est en anglais). Repli : question d'origine."""
    try:
        resp = _groq.chat.completions.create(
            model=TRANSLATE_MODEL,
            temperature=0,
            max_tokens=500,
            extra_body={"reasoning_effort": "low"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the user's question into English. Output only the "
                        "translation, nothing else. If it is already in English, "
                        "return it unchanged."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        return resp.choices[0].message.content.strip() or query
    except Exception:
        log.warning("Traduction impossible, requête d'origine utilisée", exc_info=True)
        return query


def _retrieve(query_en: str, n: int) -> list[dict]:
    col = get_collection()
    results = col.query(query_texts=[query_en], n_results=n)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [
        {
            "text": d,
            "author": m["author"],
            "title": m["title"],
            "chunk": m.get("chunk"),
            "distance": round(dist, 3),
        }
        for d, m, dist in zip(docs, metas, dists)
    ]


def search(query: str, n: int = N_RESULTS) -> list[dict]:
    """Recherche brute (sans filtre de distance) : pratique pour calibrer le retrieval."""
    return _retrieve(to_english(query), n)


SYSTEM_PROMPT = """Tu es PhilosophIA, un assistant spécialisé en philosophie et histoire.

Règles :
- Appuie-toi UNIQUEMENT sur les extraits fournis. N'ajoute aucune connaissance extérieure.
- Les extraits sont des traductions anglaises (souvent anciennes) : réponds toujours en français, en reformulant les idées.
- Cite tes sources au format [Auteur, Œuvre] après chaque affirmation qui s'appuie sur un extrait.
- Si les extraits ne permettent pas de répondre, dis-le clairement au lieu d'inventer.
- Si la question porte sur un sujet moderne, relie-la aux idées des extraits en précisant que c'est une interprétation.
- Réponse claire et structurée, sans paraphraser tout l'extrait."""


def ask(question: str) -> dict:
    query_en = to_english(question)
    chunks = [c for c in _retrieve(query_en, N_RESULTS) if c["distance"] <= MAX_DISTANCE]

    if not chunks:
        return {"answer": NO_ANSWER, "sources": [], "chunks": [], "search_query": query_en}

    context_parts = []
    sources = []
    seen = set()
    for c in chunks:
        context_parts.append(f"[{c['author']}, {c['title']}]\n{c['text']}")
        key = (c["author"], c["title"])
        if key not in seen:
            sources.append({"author": c["author"], "title": c["title"]})
            seen.add(key)

    context = "\n\n---\n\n".join(context_parts)
    user_prompt = f"EXTRAITS :\n{context}\n\nQUESTION : {question}"

    response = _groq.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        max_tokens=2048,
        extra_body={"reasoning_effort": "low"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = response.choices[0].message.content

    return {"answer": answer, "sources": sources, "chunks": chunks, "search_query": query_en}