# Your Own AI — Python Edition

 It recreates the main concepts and endpoints: vector similarity search, multiple distance metrics, document chunking, local Ollama embeddings, and retrieval-augmented generation (RAG).

> This is a Python port/reimplementation based on the repository's documented behaviorsource.

## Features

- REST API powered by FastAPI
- Brute-force nearest-neighbor search and KD-tree search
- Cosine, Euclidean, and Manhattan distance metrics
- Sample semantic vectors across CS, Math, Food, and Sports
- Document chunking with overlap
- Ollama embeddings using `nomic-embed-text`
- RAG answers generated locally with `llama3.2`
- Simple browser UI at `http://127.0.0.1:8000`
- PCA visualization data endpoint at `/visualization`
- SQLite persistence for demo vectors and document chunks
- Browser controls for listing/deleting saved chunks and viewing the PCA plot

 That algorithm can be added separately using `hnswlib` or a custom implementation. KD-tree is meaningful for Euclidean/Manhattan metrics; cosine search uses exact ranking.

## Requirements

- Python 3.10+
- Ollama for document embedding and RAG (optional for demo-vector search)
- Internet is not required at runtime after Python packages and Ollama models are installed.

## Windows setup

Open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install Ollama from https://ollama.com if you want document embeddings and RAG. Then download the models:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

Copy `.env.example` to `.env` to customize the Ollama URL and model names. Environment variables already set in PowerShell take precedence. Set `VECTOR_DB_PATH` to choose a different SQLite database location; by default the database is `data/vectors.sqlite3` in the project folder. The database directory is created automatically.

Start the app:

```powershell
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

## API examples

### Search demo vectors

```powershell
curl.exe -X POST http://127.0.0.1:8000/search `
  -H "Content-Type: application/json" `
  -d "{\"vector\":[0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],\"k\":3,\"metric\":\"cosine\",\"algo\":\"brute\"}"
```

### Insert a document

```json
POST /doc/insert
{
  "title": "Operating Systems Notes",
  "text": "A process is a program in execution..."
}
```

### Ask a question

```json
POST /doc/ask
{
  "question": "What is a process?",
  "k": 3
}
```

## Project structure

```text
your_own_ai_python/
├── app.py              # FastAPI routes and application entry point
├── vector_store.py     # Vector store, distance functions, search and chunking
├── ollama_client.py    # Local Ollama HTTP client
├── static/
│   └── index.html      # Minimal web interface
├── requirements.txt
├── .env.example
└── README.md
```

## Limitations and next steps

- SQLite persists demo vectors and document chunks across restarts.
- HNSW graph implementation is not included; choosing `hnsw` runs exact search, and `/hnsw-info` reports this accurately.
- Document embeddings are computed before chunks are committed, avoiding partial saves when Ollama fails.
- Add automated tests, authentication, and a real HNSW index before presenting this as production-ready.

frontend demo-https://my-own-ai-sneha.vercel.app/
