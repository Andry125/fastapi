import os
from fastapi import FastAPI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Charger depuis les variables d'environnement
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

@app.get("/search")
def search(q: str):
    query_vector = model.encode(q).tolist()
    response = client.query_points(collection_name="pdf_docs", query=query_vector, limit=5)
    return [
        {"score": sp.score, "page": sp.payload.get("page", "?"), "text": sp.payload.get("text", "")}
        for sp in response.points
    ]
