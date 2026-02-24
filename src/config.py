import os

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


GCP_PROJECT_ID = _required("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
BQ_DATASET = _required("BQ_DATASET")
BQ_TABLE = os.getenv("BQ_TABLE", "slide_chunks")
GCS_BUCKET = _required("GCS_BUCKET")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-005")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.0-flash")
GEN_FALLBACK_MODELS = [
    m.strip() for m in os.getenv("GEN_FALLBACK_MODELS", "").split(",") if m.strip()
]
MEDIA_GEN_MODEL = os.getenv("MEDIA_GEN_MODEL", GEN_MODEL)
MEDIA_FALLBACK_MODELS = [
    m.strip() for m in os.getenv("MEDIA_FALLBACK_MODELS", "").split(",") if m.strip()
]
VIDEO_ENABLE_VISUAL_ANALYSIS = os.getenv("VIDEO_ENABLE_VISUAL_ANALYSIS", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

VERTEX_MAX_RETRIES = int(os.getenv("VERTEX_MAX_RETRIES", "4"))
VERTEX_RETRY_INITIAL_SECONDS = float(os.getenv("VERTEX_RETRY_INITIAL_SECONDS", "2.0"))
VERTEX_RETRY_MAX_SECONDS = float(os.getenv("VERTEX_RETRY_MAX_SECONDS", "20.0"))
