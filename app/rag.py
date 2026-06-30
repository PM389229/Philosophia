"""
Moteur RAG : recherche dans ChromaDB + génération via Groq (Llama 3).
"""
import os
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "philosophia"
MODEL = "llama-3.3-70b-versatile"
N_RESULTS = 5

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


def search(query: str, n: int = N_RESULTS) -> list[dict]:
    col = get_collection()
    results = col.query(query_texts=[query], n_results=n)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [{"text": d, "author": m["author"], "title": m["title"]} for d, m in zip(docs, metas)]


def ask(question: str) -> dict:
    chunks = search(question)

    context_parts = []
    sources = []
    seen = set()
    for c in chunks:
        context_parts.append(f"[{c['author']} — {c['title']}]\n{c['text']}")
        key = (c["author"], c["title"])
        if key not in seen:
            sources.append({"author": c["author"], "title": c["title"]})
            seen.add(key)

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""Tu es PhilosophIA, un assistant spécialisé en philosophie et histoire.
Réponds à la question en t'appuyant UNIQUEMENT sur les extraits fournis.
Cite explicitement les auteurs. Si les extraits ne suffisent pas, dis-le.
Réponds en français, de façon claire et structurée.

EXTRAITS :
{context}

QUESTION : {question}

RÉPONSE :"""

    response = _groq.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    answer = response.choices[0].message.content

    return {"answer": answer, "sources": sources, "chunks": chunks}
