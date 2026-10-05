"""
Ingestion pipeline : télécharge des textes depuis Project Gutenberg,
les découpe en chunks, les embed et les stocke dans ChromaDB.
"""
import os
import re

import chromadb
import requests
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "texts")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "philosophia"
BATCH_SIZE = 500

# Textes Gutenberg : (auteur, titre affiché, url, langue du texte)
# Attention : le titre affiché est celui que verra l'utilisateur, il doit
# correspondre au vrai contenu du fichier (voir la ligne "Titre Gutenberg" à l'ingestion).
CORPUS = [
    ("Platon",      "La République",               "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",  "en"),
    ("Nietzsche",   "Par-delà le bien et le mal",  "https://www.gutenberg.org/cache/epub/4363/pg4363.txt",  "en"),
    ("Descartes",   "Discours de la méthode",      "https://www.gutenberg.org/cache/epub/59/pg59.txt",      "en"),
    ("Descartes",   "Méditations métaphysiques",   "https://www.gutenberg.org/cache/epub/70091/pg70091.txt", "en"),
    ("Machiavel",   "Le Prince",                   "https://www.gutenberg.org/cache/epub/1232/pg1232.txt",  "en"),
    ("Thucydide",   "Guerre du Péloponnèse",       "https://www.gutenberg.org/cache/epub/7142/pg7142.txt",  "en"),
    ("Marc Aurèle", "Pensées pour moi-même",       "https://www.gutenberg.org/cache/epub/2680/pg2680.txt",  "en"),
    ("Aristote",    "La Politique",                "https://www.gutenberg.org/cache/epub/6762/pg6762.txt",  "en"),
    ("Hobbes",      "Le Léviathan",                "https://www.gutenberg.org/cache/epub/3207/pg3207.txt",  "en"),
]


def download_text(url: str, dest_path: str) -> str:
    if os.path.exists(dest_path):
        with open(dest_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    print(f"  Téléchargement : {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    text = r.text
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def gutenberg_title(text: str) -> str:
    """Titre déclaré dans l'en-tête Gutenberg (pour vérifier le corpus)."""
    m = re.search(r"^Title:\s*(.+)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else "?"


def clean_gutenberg(text: str) -> str:
    """Supprime les headers/footers Gutenberg."""
    text = text.replace("\r\n", "\n")
    start = re.search(r"\*\*\* START OF .+? \*\*\*", text)
    end = re.search(r"\*\*\* END OF .+? \*\*\*", text)
    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start()]
    return text.strip()


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


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
    collection = client.create_collection(
        COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # distance = 1 - similarité cosinus
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )

    total = 0
    for author, title, url, lang in CORPUS:
        print(f"\n[{author}] {title}")
        slug = slugify(f"{author}_{title}")
        dest = os.path.join(DATA_DIR, slug + ".txt")
        try:
            raw = download_text(url, dest)
            print(f"  Titre Gutenberg : {gutenberg_title(raw)}")
            text = clean_gutenberg(raw)
            chunks = splitter.split_text(text)
            print(f"  {len(chunks)} chunks")

            ids = [f"{slug}_{i}" for i in range(len(chunks))]
            metas = [
                {"author": author, "title": title, "lang": lang, "chunk": i}
                for i in range(len(chunks))
            ]
            for s in range(0, len(chunks), BATCH_SIZE):
                collection.add(
                    documents=chunks[s:s + BATCH_SIZE],
                    ids=ids[s:s + BATCH_SIZE],
                    metadatas=metas[s:s + BATCH_SIZE],
                )
            total += len(chunks)
        except Exception as e:
            print(f"  ⚠ Erreur : {e}")

    print(f"\n✓ Index construit : {total} chunks dans ChromaDB")


if __name__ == "__main__":
    build_index()