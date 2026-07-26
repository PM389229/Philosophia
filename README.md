# PhilosophIA

Moteur de recherche sémantique sur corpus philosophique et historique.
RAG (Retrieval-Augmented Generation) : ChromaDB + Groq (Llama 3).

## Stack

- **FastAPI** — API REST
- **ChromaDB** — vector store local
- **sentence-transformers** — embeddings (all-MiniLM-L6-v2)
- **Groq (Llama 3)** — LLM cloud
- **HTML vanilla** — interface web

## Corpus

Platon, Nietzsche, Descartes, Machiavel, Thucydide, Marc Aurèle, Aristote, Hobbes
(Project Gutenberg — domaine public)

## Installation

```bash
# 1. Clé API Groq (gratuit sur console.groq.com)
export GROQ_API_KEY=gsk_...

# 2. Dépendances Python
pip install -r requirements.txt

# 3. Construire l'index (télécharge et indexe les textes ~5 min)
python3 app/ingest.py

# 4. Lancer le serveur
python3 -m uvicorn app.main:app --reload --port 8000
```

Ouvrir http://localhost:8000

## Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Interface web |
| POST | `/ask` | Question → réponse RAG + sources |
| GET | `/search?q=...` | Recherche sémantique brute |
| GET | `/health` | Statut |
