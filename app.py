from contextlib import asynccontextmanager
from pathlib import Path
import math
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from vector_store import VectorStore
from ollama_client import OllamaClient

BASE_DIR = Path(__file__).resolve().parent
store = VectorStore()
ollama = OllamaClient()

DEMO_ITEMS = [
    ("Binary tree", "CS", [0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]),
    ("Graph traversal", "CS", [0.85,0.82,0.72,0.58,0.12,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]),
    ("Dynamic programming", "CS", [0.88,0.75,0.8,0.55,0.12,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]),
    ("Calculus", "Math", [0.1,0.1,0.1,0.1,0.9,0.8,0.75,0.7,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]),
    ("Linear algebra", "Math", [0.1,0.1,0.1,0.1,0.85,0.82,0.78,0.72,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]),
    ("Probability", "Math", [0.1,0.1,0.1,0.1,0.8,0.85,0.72,0.76,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]),
    ("Sushi", "Food", [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1]),
    ("Pizza", "Food", [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.85,0.82,0.72,0.58,0.1,0.1,0.1,0.1]),
    ("Basketball", "Sports", [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9,0.8,0.7,0.6]),
    ("Football", "Sports", [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.85,0.82,0.72,0.58]),
]
if not store.demo:
    for title, category, vector in DEMO_ITEMS:
        store.insert_demo(title, category, vector)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Your Own AI — Python Edition", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

class SearchRequest(BaseModel):
    vector: list[float] = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=100)
    metric: str = "cosine"
    algo: str = "brute"

class InsertRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    category: str = "Custom"
    vector: list[float] = Field(..., min_length=1)

class DocumentRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1)

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    k: int = Field(default=3, ge=1, le=10)

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.get("/status")
def status():
    return ollama.status()

@app.get("/items")
def items():
    return store.list_demo()

@app.get("/stats")
def stats():
    return {"demo_vectors": len(store.demo), "documents": len(store.documents),
            "dimensions": len(store.demo[0]["vector"]) if store.demo else 0}

@app.post("/insert")
def insert(req: InsertRequest):
    if not all(math.isfinite(x) for x in req.vector):
        raise HTTPException(400, "Vector values must be finite numbers")
    dimensions = len(store.demo[0]["vector"]) if store.demo else len(req.vector)
    if len(req.vector) != dimensions:
        raise HTTPException(400, f"Vector must have {dimensions} dimensions")
    item = store.insert_demo(req.title, req.category, req.vector)
    return {"ok": True, "item": item}

@app.delete("/delete/{item_id}")
def delete(item_id: int):
    if not store.delete_demo(item_id):
        raise HTTPException(404, "Item not found")
    return {"ok": True}

@app.post("/search")
def search(req: SearchRequest):
    try:
        return {"results": store.search(req.vector, req.k, req.metric, req.algo)}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/search")
def search_get(v: str, k: int = 5, metric: str = "cosine", algo: str = "brute"):
    try:
        vector = [float(x) for x in v.split(",")]
        if not all(math.isfinite(x) for x in vector):
            raise ValueError("Vector values must be finite numbers")
        return {"results": store.search(vector, k, metric, algo)}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/benchmark")
def benchmark(v: str, k: int = 5, metric: str = "cosine"):
    try:
        vector = [float(x) for x in v.split(",")]
        if not all(math.isfinite(x) for x in vector):
            raise ValueError("Vector values must be finite numbers")
        return store.benchmark(vector, k, metric)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/hnsw-info")
def hnsw_info():
    return {"implementation": "exact search (HNSW is not implemented)",
            "indexed_demo_vectors": len(store.demo), "note": "The hnsw option currently uses exact ranking."}

@app.post("/doc/insert")
def insert_document(req: DocumentRequest):
    if not req.text.strip():
        raise HTTPException(400, "Document text cannot be empty")
    chunks = store.chunk_text(req.text, chunk_words=250, overlap_words=40)
    try:
        embedded = [(req.title, i, chunk, ollama.embed(chunk)) for i, chunk in enumerate(chunks)]
        saved = store.insert_documents(embedded)
    except requests.RequestException as e:
        raise HTTPException(503, f"Ollama is unavailable: {e}")
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise HTTPException(502, f"Could not create document embeddings: {e}")
    return {"ok": True, "title": req.title, "chunks_added": len(saved), "chunks": saved}

@app.get("/doc/list")
def list_documents():
    return store.list_documents()

@app.delete("/doc/delete/{doc_id}")
def delete_document(doc_id: int):
    if not store.delete_document(doc_id):
        raise HTTPException(404, "Document chunk not found")
    return {"ok": True}

@app.post("/doc/ask")
def ask(req: AskRequest):
    if not store.documents:
        raise HTTPException(400, "Add a document first using POST /doc/insert")
    try:
        if not req.question.strip():
            raise HTTPException(400, "Question cannot be empty")
        qvec = ollama.embed(req.question)
        contexts = store.search_documents(qvec, req.k)
        context_text = "\n\n".join(
            f"[Source: {c['title']} | chunk {c['chunk_index']}]\n{c['text']}"
            for c in contexts
        )
        prompt = (
            "Answer the question using only the supplied context. If the answer is not in "
            "the context, say you could not find it in the documents. Cite source titles.\n\n"
            f"CONTEXT:\n{context_text}\n\nQUESTION: {req.question}"
        )
        answer = ollama.generate(prompt)
        return {"answer": answer, "sources": contexts}
    except requests.RequestException as e:
        raise HTTPException(503, f"Ollama is unavailable: {e}")

@app.get("/visualization")
def visualization():
    items = store.demo
    if len(items) < 2:
        return {"points": []}
    X = np.array([i["vector"] for i in items], dtype=float)
    coords = PCA(n_components=2).fit_transform(X)
    return {"points": [
        {"id": item["id"], "title": item["title"], "category": item["category"],
         "x": float(coords[idx, 0]), "y": float(coords[idx, 1])}
        for idx, item in enumerate(items)
    ]}
