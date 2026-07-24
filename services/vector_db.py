import chromadb
from chromadb.utils import embedding_functions

from config import settings
from core.logging import get_logger

logger = get_logger(__name__)

chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=settings.OPENAI_API_KEY,
    model_name="text-embedding-3-small",
)

drill_collection = chroma_client.get_or_create_collection(
    name="drill_library",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},
)

baseline_collection = chroma_client.get_or_create_collection(
    name="player_baselines",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},
)

coach_preference_collection = chroma_client.get_or_create_collection(
    name="coach_preferences",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},
)

logger.info("Chroma collections ready | path=%s", settings.CHROMA_DB_PATH)
