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
from fastembed import TextEmbedding
from pypdf import PdfReader

# ----- Configuration -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
VECTOR_SIZE = 384

qdrant = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

app = FastAPI(
    title="PDF Ingestion & Semantic Search API",
    description="API avec FastEmbed + recherche hybride (dense + BM25)",
    version="3.0.0"
)

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

# ----- Modèle global -----
embedding_model = None

@app.on_event("startup")
def startup_event():
    global embedding_model
    logger.info(f"Chargement du modèle FastEmbed : {MODEL_NAME}")
    embedding_model = TextEmbedding(model_name=MODEL_NAME)
    ensure_collection()
    logger.info("✅ Modèle chargé et collection prête.")

# ----- Vérification/création de la collection -----
def ensure_collection():
    collections = qdrant.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        logger.info(f"Création de la collection '{COLLECTION_NAME}'...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": qdrant_models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=qdrant_models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "bm25": qdrant_models.SparseVectorParams(
                    modifier=qdrant_models.Modifier.IDF
                )
            }
        )
        logger.info("✅ Collection créée.")
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' existe déjà.")

# ----- Fonctions utilitaires -----
def extract_text_from_pdf(pdf_file: UploadFile) -> str:
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur extraction PDF : {e}")
        raise HTTPException(500, f"Erreur de lecture du PDF : {str(e)}")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    embeddings_generator = embedding_model.embed(chunks)
    return [emb.tolist() for emb in embeddings_generator]

def upsert_chunks(filename: str, chunks: List[str], dense_vectors: List[List[float]]) -> int:
    """Insère les chunks avec vecteurs dense + BM25 et retourne le nombre inséré."""
    points = []
    for i, (chunk, dense_vector) in enumerate(zip(chunks, dense_vectors)):
        points.append(
            qdrant_models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vector,
                    "bm25": qdrant_models.Document(
                        text=chunk,
                        model="qdrant/bm25"
                    )
                },
                payload={
                    "filename": filename,
                    "chunk_index": i,
                    "text": chunk,
                    "total_chunks": len(chunks)
                }
            )
        )
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)

# ----- Endpoints -----
@app.get("/health")
async def health_check():
    try:
        qdrant.get_collections()
        return {"status": "ok", "qdrant": "connected"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/upload-pdf/")
async def upload_pdf(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None)
):
    if API_SECRET_KEY and x_api_key != API_SECRET_KEY:
        raise HTTPException(403, "Clé API invalide")

    if file.content_type != "application/pdf":
        raise HTTPException(400, "Seuls les PDF sont acceptés")

    logger.info(f"📄 Upload : {file.filename}")
    text = extract_text_from_pdf(file)
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "Aucun texte valide après découpage")

    logger.info(f"🔮 Génération des embeddings pour {len(chunks)} chunks...")
    vectors = generate_embeddings(chunks)

    logger.info("💾 Insertion dans Qdrant...")
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

        # Vecteur dense de la requête
        dense_query_vector = list(embedding_model.embed([search.query]))[0].tolist()

        # Filtre optionnel
        qdrant_filter = None
        if search.filter:
            qdrant_filter = qdrant_models.Filter(**search.filter)

        # Recherche hybride
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                qdrant_models.Prefetch(
                    query=dense_query_vector,
                    using="dense",
                    limit=search.top_k * 2,
                    filter=qdrant_filter
                ),
                qdrant_models.Prefetch(
                    query=qdrant_models.Document(text=search.query, model="qdrant/bm25"),
                    using="bm25",
                    limit=search.top_k * 2,
                    filter=qdrant_filter
                ),
            ],
            query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
            limit=search.top_k,
            with_payload=True,
            with_vectors=False,
        )
        results = response.points

        logger.info(f"✅ {len(results)} résultats trouvés")

        return [
            SearchResult(id=hit.id, score=hit.score, payload=hit.payload)
            for hit in results
        ]

    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Erreur interne : {str(e)}")

@app.delete("/clear-collection/")
async def clear_collection(x_api_key: Optional[str] = Header(None)):
    if API_SECRET_KEY and x_api_key != API_SECRET_KEY:
        raise HTTPException(403, "Clé API invalide")
    qdrant.delete_collection(collection_name=COLLECTION_NAME)
    ensure_collection()
    return {"status": "collection cleared and recreated"}
