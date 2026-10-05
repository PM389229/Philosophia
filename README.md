# PhilosophIA

Moteur de recherche sémantique sur corpus philosophique et historique.
RAG (Retrieval-Augmented Generation) : ChromaDB + Groq (GPT-OSS).

## Stack

- **FastAPI** — API REST
- **ChromaDB** — vector store local (similarité cosinus)
- **sentence-transformers** — embeddings (all-MiniLM-L6-v2)
- **Groq (GPT-OSS 120B et 20B)** — LLM cloud
- **HTML vanilla** — interface web

## Corpus

Platon, Nietzsche, Descartes (Discours de la méthode, Méditations), Machiavel, Thucydide, Marc Aurèle, Aristote, Hobbes
(Project Gutenberg, traductions anglaises — domaine public)

## Comment ça marche

1. La question (en français) est traduite en anglais par un petit modèle, car le corpus est en anglais.
2. Les 6 extraits les plus proches sont récupérés dans ChromaDB.
3. Les extraits trop éloignés (distance cosinus > 0.65) sont écartés. S'il n'en reste aucun, l'app répond qu'elle ne trouve pas, sans appeler le modèle.
4. Le modèle répond en français, uniquement à partir des extraits, avec des citations `[Auteur, Œuvre]`.

## Installation

```bash
# 1. Clé API Groq (gratuit sur console.groq.com)
export GROQ_API_KEY=gsk_...

# 2. Dépendances Python
python3 -m pip install -r requirements.txt

# 3. Construire l'index (télécharge et indexe les textes ~5 min)
python3 app/ingest.py

# 4. Lancer le serveur
python3 -m uvicorn app.main:app --reload --port 8000
```

Ouvrir http://localhost:8000

## Configuration (optionnel)

| Variable | Défaut | Rôle |
|----------|--------|------|
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Modèle de génération |
| `GROQ_TRANSLATE_MODEL` | `openai/gpt-oss-20b` | Modèle de traduction des questions |
| `MAX_DISTANCE` | `0.65` | Seuil de distance cosinus au-delà duquel un extrait est écarté |

Les modèles Llama 3.x (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) ont été retirés par Groq le 16 août 2026, d'où la migration vers GPT-OSS. Les noms de modèles sont configurables pour faciliter les prochaines migrations.

## Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Interface web |
| POST | `/ask` | Question → réponse RAG + sources |
| GET | `/search?q=...&n=5` | Recherche sémantique brute (renvoie la distance de chaque extrait, `n` entre 1 et 20) |
| GET | `/health` | Statut |