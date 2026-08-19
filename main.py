import os
from fastapi import FastAPI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Charger les secrets depuis les variables d'environnement Render
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialiser Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Lazy load du modèle (chargé seulement à la première requête)
model = None

@app.get("/search")
def search(q: str):
    global model
    if model is None:
        # Modèle compact (~90MB, adapté au plan gratuit Render)
        model = SentenceTransformer("all-MiniLM-L6-v2")

    query_vector = model.encode(q).tolist()
    response = client.query_points(
        collection_name="pdf_docs",
        query=query_vector,
        limit=5
    )

    return [
        {
            "score": sp.score,
            "page": sp.payload.get("page", "?"),
            "text": sp.payload.get("text", "")
        }
        for sp in response.points
    ]