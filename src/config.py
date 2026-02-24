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

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
