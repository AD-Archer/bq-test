# BigQuery + GCP Slide RAG Starter

This project gives you a working starter to:

1. Upload one or many files (`.pptx`, `.pdf`, images, audio, video)
2. Ingest slide files directly from Google Drive (file ID or folder ID)
3. Parse and chunk slide text
4. Create embeddings with Vertex AI
5. Store chunks + embeddings in BigQuery
6. Query with semantic search from BigQuery and generate an answer with Gemini

## Architecture

- `POST /upload-slides`: accepts a file, uploads to GCS, parses text, writes chunks to BigQuery.
- `POST /query`: embeds your question, retrieves nearest chunks from BigQuery `VECTOR_SEARCH`, then asks Gemini to answer from those chunks.

## Prerequisites

- A GCP project with billing enabled
- `uv` installed: https://docs.astral.sh/uv/getting-started/installation/
- Python 3.11+
- Enabled APIs:
  - BigQuery API
  - Cloud Storage API
  - Vertex AI API
- Application Default Credentials available locally:
  - `gcloud auth application-default login`

## Setup

1. Authenticate and select your GCP project:

```bash
just auth
just project YOUR_PROJECT_ID
just enable-apis
just adc-quota YOUR_PROJECT_ID
just project-show
```

2. Install dependencies with `uv`:

```bash
uv sync
```

3. Configure environment:

```bash
cp .env.example .env
```

Set values in `.env`.

4. Create BigQuery schema/table:

```bash
# Edit PROJECT_ID and DATASET in infra/schema.sql first.
bq query --use_legacy_sql=false < infra/schema.sql
```

5. Create GCS bucket if you do not already have one:

```bash
just bucket-create YOUR_BUCKET_NAME us-central1
```

6. Run the API:

```bash
uv run --env-file .env uvicorn src.main:app --reload --port 8080
```

Open docs at: `http://127.0.0.1:8080/docs`

## Just Commands

```bash
just sync
just run
just auth
just project YOUR_PROJECT_ID
just enable-apis
just adc-quota YOUR_PROJECT_ID
just project-show
just bucket-create YOUR_BUCKET_NAME us-central1
just schema
just upload /absolute/path/to/deck.pptx
just ingest-drive-file DRIVE_FILE_ID
just ingest-drive-folder DRIVE_FOLDER_ID 50
just docs-list 20
just doc-read SOURCE_ID
just mcp-search "What changed in summer 2025?" 5
just mcp-read SOURCE_ID
just find-word SOURCE_ID hello 50
just ask "What are the rollout risks?" 5
```

## Example usage

Upload slides:

```bash
curl -X POST "http://127.0.0.1:8080/upload-slides" \
  -F "file=@/absolute/path/to/deck.pptx"
```

Ask a question:

```bash
curl -X POST "http://127.0.0.1:8080/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the rollout risks?", "top_k":5}'
```

Ingest from Google Drive:

```bash
curl -X POST "http://127.0.0.1:8080/ingest-drive" \
  -H "Content-Type: application/json" \
  -d '{"folder_id":"YOUR_DRIVE_FOLDER_ID","max_files":50}'
```

Read full document by ID:

```bash
curl "http://127.0.0.1:8080/documents?limit=20"
curl "http://127.0.0.1:8080/documents/SOURCE_ID"
curl "http://127.0.0.1:8080/documents/SOURCE_ID/images?max_images=30"
```

MCP-style tool endpoints:

```bash
curl -X POST "http://127.0.0.1:8080/mcp/tools/search" \
  -H "Content-Type: application/json" \
  -d '{"question":"What changed in summer 2025?","top_k":5}'

curl -X POST "http://127.0.0.1:8080/mcp/tools/read_document" \
  -H "Content-Type: application/json" \
  -d '{"source_id":"SOURCE_ID"}'

curl -X POST "http://127.0.0.1:8080/mcp/tools/list_images" \
  -H "Content-Type: application/json" \
  -d '{"source_id":"SOURCE_ID"}'

curl -X POST "http://127.0.0.1:8080/mcp/tools/get_media_source" \
  -H "Content-Type: application/json" \
  -d '{"source_id":"SOURCE_ID"}'

curl "http://127.0.0.1:8080/videos/SOURCE_ID/words/hello?max_hits=50"
```

## Important notes

- Current ingestion supports `.pptx`, `.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.mp3`, `.wav`, `.m4a`, `.mp4`, `.mov`, `.webm`, `.mkv`.
- New metadata is stored per chunk: source name/system, content type, modalities, and detected date.
- Audio/video chunks include `media_start_seconds`, `media_end_seconds`, `speech_style`, and `word_timestamps_json`.
- Video frame extraction for `/documents/{source_id}/images` requires `ffmpeg` on PATH.
- To reduce `429 Resource exhausted` for media uploads:
  - Set `VIDEO_ENABLE_VISUAL_ANALYSIS=false` (default) to lower model load.
  - Set `MEDIA_GEN_MODEL` and optional `MEDIA_FALLBACK_MODELS` for automatic fallback on throttling.
  - Increase retry settings (`VERTEX_MAX_RETRIES`, `VERTEX_RETRY_INITIAL_SECONDS`, `VERTEX_RETRY_MAX_SECONDS`) as needed.
  - Increase `EMBED_BATCH_SIZE` to reduce embedding request count.
- Each file gets a stable `source_id`, and full text can be retrieved by that ID.
- You can also retrieve extracted document images using the same `source_id`.
- Run `just schema-migrate` once to add new metadata columns to an existing table.

## MCP Server Setup

Start API first:

```bash
just run
```

In another terminal start MCP stdio server:

```bash
just mcp-server
```

Tool names exposed: `search`, `list_documents`, `read_document`, `list_images`, `get_media_source`, `find_word_occurrences`.

If your client needs URL-based MCP (HTTP), run:

```bash
just mcp-http
```

Then use MCP URL:

```text
http://127.0.0.1:8000/mcp
```
- For Google Drive ingestion, rerun `just auth` so ADC includes `drive.readonly` scope.
- Retrieval uses BigQuery `VECTOR_SEARCH` on `embedding ARRAY<FLOAT64>`.
- For large datasets, add a BigQuery vector index to improve latency.
- The generated answer is constrained to retrieved context, but you should still validate critical outputs.
