import os
from pathlib import Path

import requests


def _load_env_file():
    """Load simple KEY=value entries from the project .env without overriding the shell."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_env_file()

class OllamaClient:
    def __init__(self, base_url=None, embedding_model=None, generation_model=None):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.embedding_model = embedding_model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.generation_model = generation_model or os.getenv("OLLAMA_GENERATION_MODEL", "llama3.2")

    def status(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]
            return {"online": True, "models": models,
                    "embedding_model": self.embedding_model,
                    "generation_model": self.generation_model}
        except requests.RequestException:
            return {"online": False, "models": [],
                    "embedding_model": self.embedding_model,
                    "generation_model": self.generation_model}

    def embed(self, text):
        # Newer Ollama endpoint. Older versions may use /api/embeddings.
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.embedding_model, "input": text},
            timeout=120,
        )
        if response.status_code == 404:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["embedding"]
        response.raise_for_status()
        data = response.json()
        if "embeddings" in data:
            return data["embeddings"][0]
        return data["embedding"]

    def generate(self, prompt):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.generation_model, "prompt": prompt, "stream": False},
            timeout=180,
        )
        response.raise_for_status()
        return response.json().get("response", "")
