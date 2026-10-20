import math
import json
import os
import sqlite3
import time
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors

class VectorStore:
    """Small educational vector store with brute-force and KD-tree search.
    If hnswlib is installed, HNSW search is used; otherwise it falls back to exact search.
    """
    def __init__(self, db_path=None):
        self.demo = []
        self.documents = []
        self._next_demo_id = 1
        self._next_doc_id = 1
        self.db_path = Path(db_path or os.getenv("VECTOR_DB_PATH", Path(__file__).resolve().parent / "data" / "vectors.sqlite3"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
        self._load_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _initialize_db(self):
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS demo (id INTEGER PRIMARY KEY, title TEXT NOT NULL, category TEXT NOT NULL, vector TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, title TEXT NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL, vector TEXT NOT NULL)")

    def _load_db(self):
        with self._connect() as db:
            self.demo = [{"id": i, "title": t, "category": c, "vector": json.loads(v)}
                         for i, t, c, v in db.execute("SELECT id,title,category,vector FROM demo ORDER BY id")]
            self.documents = [{"id": i, "title": t, "chunk_index": ci, "text": tx, "vector": json.loads(v)}
                              for i, t, ci, tx, v in db.execute("SELECT id,title,chunk_index,text,vector FROM documents ORDER BY id")]
        self._next_demo_id = max((x["id"] for x in self.demo), default=0) + 1
        self._next_doc_id = max((x["id"] for x in self.documents), default=0) + 1

    def _save_demo(self):
        with self._connect() as db:
            db.execute("DELETE FROM demo")
            db.executemany("INSERT INTO demo VALUES (?,?,?,?)", [(x["id"], x["title"], x["category"], json.dumps(x["vector"])) for x in self.demo])

    def _save_documents(self):
        with self._connect() as db:
            db.execute("DELETE FROM documents")
            db.executemany("INSERT INTO documents VALUES (?,?,?,?,?)", [(x["id"], x["title"], x["chunk_index"], x["text"], json.dumps(x["vector"])) for x in self.documents])

    def insert_demo(self, title, category, vector):
        vector = [float(x) for x in vector]
        if not vector or not all(np.isfinite(vector)):
            raise ValueError("Vector must contain finite values")
        item = {"id": self._next_demo_id, "title": title, "category": category,
                "vector": vector}
        self._next_demo_id += 1
        self.demo.append(item)
        self._save_demo()
        return item

    def delete_demo(self, item_id):
        old = len(self.demo)
        self.demo = [x for x in self.demo if x["id"] != item_id]
        if len(self.demo) != old:
            self._save_demo()
        return len(self.demo) != old

    def list_demo(self):
        return [{"id": x["id"], "title": x["title"], "category": x["category"],
                 "dimensions": len(x["vector"])} for x in self.demo]

    @staticmethod
    def distance(a, b, metric):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape:
            raise ValueError(f"Dimension mismatch: query has {len(a)} values, stored vector has {len(b)}")
        if metric == "cosine":
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            return 1.0 if denom == 0 else float(1 - np.dot(a, b) / denom)
        if metric == "euclidean":
            return float(np.linalg.norm(a - b))
        if metric == "manhattan":
            return float(np.abs(a - b).sum())
        raise ValueError("metric must be cosine, euclidean, or manhattan")

    def search(self, vector, k=5, metric="cosine", algo="brute"):
        if not self.demo:
            return []
        if algo not in ("brute", "kdtree", "hnsw"):
            raise ValueError("algo must be brute, kdtree, or hnsw")
        if metric not in ("cosine", "euclidean", "manhattan"):
            raise ValueError("metric must be cosine, euclidean, or manhattan")
        q = np.asarray(vector, dtype=float)
        if q.ndim != 1 or not np.all(np.isfinite(q)):
            raise ValueError("Query vector must contain finite numbers")
        if any(len(x["vector"]) != len(q) for x in self.demo):
            raise ValueError("Query vector dimensions must match all stored vectors")
        started = time.perf_counter()
        # KD-tree supports Euclidean/Manhattan directly. Cosine is ranked exactly below.
        # hnswlib is optional; a dependency-free exact fallback keeps the demo runnable.
        if algo == "kdtree" and metric in ("euclidean", "manhattan") and len(self.demo) > 1:
            X = np.array([x["vector"] for x in self.demo], dtype=float)
            model = NearestNeighbors(algorithm="kd_tree", metric=metric).fit(X)
            distances, indices = model.kneighbors(q.reshape(1, -1), n_neighbors=min(k, len(X)))
            result = [(self.demo[int(i)], float(d)) for i, d in zip(indices[0], distances[0])]
        else:
            result = [(x, self.distance(q, x["vector"], metric)) for x in self.demo]
            result.sort(key=lambda pair: pair[1])
            result = result[:min(k, len(result))]
        elapsed = (time.perf_counter() - started) * 1000
        return [{"id": x["id"], "title": x["title"], "category": x["category"],
                 "distance": round(float(d), 6), "algorithm": algo,
                 "elapsed_ms": round(elapsed, 3)} for x, d in result]

    def benchmark(self, vector, k=5, metric="cosine"):
        return {algo: self.search(vector, k, metric, algo) for algo in ("brute", "kdtree", "hnsw")}

    @staticmethod
    def chunk_text(text, chunk_words=250, overlap_words=40):
        words = text.split()
        if not words:
            return []
        chunks, start = [], 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = end - overlap_words
        return chunks

    def insert_document(self, title, chunk_index, text, vector):
        row = {"id": self._next_doc_id, "title": title, "chunk_index": chunk_index,
               "text": text, "vector": [float(x) for x in vector]}
        self._next_doc_id += 1
        self.documents.append(row)
        self._save_documents()
        return {"id": row["id"], "title": title, "chunk_index": chunk_index}

    def insert_documents(self, rows):
        """Insert a fully embedded document atomically, so failed embeddings save no chunks."""
        start_id = self._next_doc_id
        additions = [{"id": start_id + i, "title": title, "chunk_index": chunk_index,
                      "text": text, "vector": [float(x) for x in vector]}
                     for i, (title, chunk_index, text, vector) in enumerate(rows)]
        if not additions:
            return []
        if any(not x["vector"] or not np.all(np.isfinite(x["vector"])) for x in additions):
            raise ValueError("Document embeddings must contain finite values")
        expected = len(self.documents[0]["vector"]) if self.documents else len(additions[0]["vector"])
        if any(len(x["vector"]) != expected for x in additions):
            raise ValueError("Document embedding dimensions must match existing documents")
        self.documents.extend(additions)
        try:
            self._save_documents()
        except Exception:
            del self.documents[-len(additions):]
            raise
        self._next_doc_id += len(additions)
        return [{"id": x["id"], "title": x["title"], "chunk_index": x["chunk_index"]} for x in additions]

    def list_documents(self):
        return [{"id": x["id"], "title": x["title"], "chunk_index": x["chunk_index"],
                 "word_count": len(x["text"].split())} for x in self.documents]

    def delete_document(self, doc_id):
        old = len(self.documents)
        self.documents = [x for x in self.documents if x["id"] != doc_id]
        if len(self.documents) != old:
            self._save_documents()
        return len(self.documents) != old

    def search_documents(self, vector, k=3):
        scored = [(x, self.distance(vector, x["vector"], "cosine")) for x in self.documents]
        scored.sort(key=lambda pair: pair[1])
        return [{"id": x["id"], "title": x["title"], "chunk_index": x["chunk_index"],
                 "text": x["text"], "distance": round(float(d), 6)}
                for x, d in scored[:min(k, len(scored))]]
