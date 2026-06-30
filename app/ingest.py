"""
Ingestion pipeline : télécharge des textes depuis Project Gutenberg,
les découpe en chunks, les embed et les stocke dans ChromaDB.
"""
import os
import re
import requests
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "texts")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "philosophia"

# Textes Gutenberg : (auteur, titre, url_id, langue)
CORPUS = [
    ("Platon",      "La République",               "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",  "en"),
    ("Nietzsche",   "Par-delà le bien et le mal",  "https://www.gutenberg.org/cache/epub/4363/pg4363.txt",  "en"),
    ("Descartes",   "Méditations Métaphysiques",    "https://www.gutenberg.org/cache/epub/59/pg59.txt",      "en"),
    ("Machiavel",   "Le Prince",                   "https://www.gutenberg.org/cache/epub/1232/pg1232.txt",  "en"),
    ("Thucydide",   "Guerre du Péloponnèse",        "https://www.gutenberg.org/cache/epub/7142/pg7142.txt",  "en"),
    ("Marc Aurèle", "Pensées pour moi-même",        "https://www.gutenberg.org/cache/epub/2680/pg2680.txt",  "en"),
    ("Aristote",    "La Politique",                 "https://www.gutenberg.org/cache/epub/6762/pg6762.txt",  "en"),
    ("Hobbes",      "Le Léviathan",                 "https://www.gutenberg.org/cache/epub/3207/pg3207.txt",  "en"),
]


def download_text(url: str, dest_path: str) -> str:
    if os.path.exists(dest_path):
        with open(dest_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    print(f"  Téléchargement : {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    text = r.text
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def clean_gutenberg(text: str) -> str:
    """Supprime les headers/footers Gutenberg."""
    start = re.search(r"\*\*\* START OF .+? \*\*\*", text)
    end = re.search(r"\*\*\* END OF .+? \*\*\*", text)
    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start()]
    return text.strip()


def build_index():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Recrée la collection à chaque ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, embedding_function=ef)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )

    total = 0
    for author, title, url, lang in CORPUS:
        print(f"\n[{author}] {title}")
        fname = re.sub(r"[^a-z0-9]", "_", title.lower()) + ".txt"
        dest = os.path.join(DATA_DIR, fname)
        try:
            raw = download_text(url, dest)
            text = clean_gutenberg(raw)
            chunks = splitter.split_text(text)
            print(f"  {len(chunks)} chunks")

            ids = [f"{author}_{i}" for i in range(len(chunks))]
            metas = [{"author": author, "title": title, "lang": lang} for _ in chunks]
            collection.add(documents=chunks, ids=ids, metadatas=metas)
            total += len(chunks)
        except Exception as e:
            print(f"  ⚠ Erreur : {e}")

    print(f"\n✓ Index construit : {total} chunks dans ChromaDB")


if __name__ == "__main__":
    build_index()
