import os
import uuid
import logging
import io
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import qdrant_client
from qdrant_client.http import models as qdrant_models
from fastembed import TextEmbedding  # ← remplace sentence-transformers
from pypdf import PdfReader

# ----- Configuration -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables d'environnement (à définir sur Render)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")       # optionnel
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))   # en mots
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Modèle FastEmbed (léger : ~100 Mo RAM, 384 dimensions)
# Pour le français, vous pouvez utiliser "intfloat/multilingual-e5-small" (384 dims)
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
VECTOR_SIZE = 384  # pour ce modèle (vérifier selon le modèle choisi)

# Connexion à Qdrant Cloud
qdrant = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

# ----- Initialisation de FastAPI -----
app = FastAPI(
    title="PDF Ingestion & Semantic Search API",
    description="API avec FastEmbed pour une faible consommation mémoire",
    version="2.0.0"
)

# CORS (à restreindre en production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Modèles Pydantic -----
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5
    filter: Optional[dict] = None

class SearchResult(BaseModel):
    id: str
    score: float
    payload: dict

# ----- Chargement du modèle FastEmbed (global, une seule fois) -----
embedding_model = None

@app.on_event("startup")
def startup_event():
    global embedding_model
    logger.info(f"Chargement du modèle FastEmbed : {MODEL_NAME}")
    # FastEmbed télécharge le modèle au premier appel si non présent
    embedding_model = TextEmbedding(model_name=MODEL_NAME)
    ensure_collection()
    logger.info("Modèle chargé et collection prête.")

# ----- Vérification/création de la collection -----
def ensure_collection():
    collections = qdrant.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        logger.info(f"Création de la collection '{COLLECTION_NAME}'")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=VECTOR_SIZE,
                distance=qdrant_models.Distance.COSINE
            )
        )
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' existe déjà.")

# ----- Fonctions utilitaires -----
def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """Extrait le texte d'un PDF."""
    try:
        content = pdf_file.file.read()
        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            raise HTTPException(400, "Le PDF ne contient aucun texte extractible.")
        return text
    except Exception as e:
        logger.error(f"Erreur extraction PDF : {e}")
        raise HTTPException(500, f"Erreur de lecture du PDF : {str(e)}")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Découpe le texte en chunks (par mots) avec chevauchement."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def generate_embeddings(chunks: List[str]):
    """Génère les embeddings avec FastEmbed. Retourne une liste de listes de floats."""
    # FastEmbed.embed() retourne un générateur de numpy arrays
    embeddings_generator = embedding_model.embed(chunks)
    # Convertir en listes de floats
    embeddings = [emb.tolist() for emb in embeddings_generator]
    return embeddings

def upsert_chunks(filename: str, chunks: List[str], vectors: List[List[float]]):
    """Insère les chunks et leurs vecteurs dans Qdrant."""
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        point_id = str(uuid.uuid4())
        points.append(
            qdrant_models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "filename": filename,
                    "chunk_index": i,
                    "text": chunk,
                    "total_chunks": len(chunks)
                }
            )
        )
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    return len(points)

# ----- Endpoints -----
@app.get("/health")
async def health_check():
    return {"status": "ok", "qdrant": "connected" if qdrant else "error"}

@app.post("/upload-pdf/")
async def upload_pdf(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None)
):
    if API_SECRET_KEY and x_api_key != API_SECRET_KEY:
        raise HTTPException(403, "Clé API invalide")

    if file.content_type != "application/pdf":
        raise HTTPException(400, "Seuls les PDF sont acceptés")

    # Extraction
    text = extract_text_from_pdf(file)
    # Découpage
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "Aucun texte valide après découpage")

    # Embeddings
    vectors = generate_embeddings(chunks)

    # Insertion dans Qdrant
    nb_inserted = upsert_chunks(file.filename, chunks, vectors)

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_inserted": nb_inserted,
        "total_chunks": len(chunks)
    }

@app.post("/search/", response_model=List[SearchResult])
async def search_documents(
    search: SearchQuery,
    x_api_key: Optional[str] = Header(None)
):
    if API_SECRET_KEY and x_api_key != API_SECRET_KEY:
        raise HTTPException(403, "Clé API invalide")

    try:
        logger.info(f"🔍 Recherche : {search.query}")
        
        # Vérification du modèle
        if embedding_model is None:
            raise HTTPException(500, "Modèle d'embeddings non chargé")
        
        # Génération de l'embedding
        query_embedding = list(embedding_model.embed([search.query]))[0].tolist()
        logger.info(f"✅ Embedding généré ({len(query_embedding)} dimensions)")
        
        # Construction du filtre Qdrant (si fourni)
        qdrant_filter = None
        if search.filter:
            qdrant_filter = qdrant_models.Filter(**search.filter)
        
        # ✅ Nouvelle API Qdrant (>= 1.10.0)
        try:
            # Essayer d'abord avec query_points (version moderne)
            response = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,  # ← Utiliser query_embedding
                limit=search.top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=qdrant_filter
            )
            results = response.points
        except AttributeError:
            # Fallback : ancienne API
            results = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,  # ← Utiliser query_embedding
                limit=search.top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=qdrant_filter
            )
        
        logger.info(f"✅ {len(results)} résultats trouvés")
        
        return [
            SearchResult(
                id=hit.id,
                score=hit.score,
                payload=hit.payload
            )
            for hit in results
        ]
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Erreur interne : {str(e)}")

@app.delete("/clear-collection/")
async def clear_collection(
    x_api_key: Optional[str] = Header(None)
):
    if API_SECRET_KEY and x_api_key != API_SECRET_KEY:
        raise HTTPException(403, "Clé API invalide")
    qdrant.delete_collection(collection_name=COLLECTION_NAME)
    ensure_collection()
    return {"status": "collection cleared and recreated"}
