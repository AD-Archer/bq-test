# BigQuery RAG Research Handoff

## Objective
Build a GCP-native RAG pipeline that can:
- Ingest files (`.pptx`, `.pdf`, images, audio) and Google Drive content.
- Store embeddings + metadata in BigQuery.
- Answer questions with BigQuery vector search + Gemini.
- Expose agent-friendly tools via MCP.

## System Implemented

### Core stack
- API framework: FastAPI
- Vector store + metadata: BigQuery table `rag_demo.slide_chunks`
- Blob storage: GCS bucket (`slides/` + extracted assets)
- Embeddings: Vertex `text-embedding-005`
- Generation/vision/audio handling: Gemini (`GEN_MODEL` from `.env`)
- MCP bridge: FastMCP server (`src/mcp_server.py`)

### Main components
- `src/main.py`: HTTP API + root GUI
- `src/rag_service.py`: ingestion, extraction, embeddings, retrieval, document reconstruction
- `src/mcp_server.py`: MCP tools mapped to API endpoints
- `infra/table_schema.json`: BigQuery schema
- `Justfile`: bootstrap, schema ops, run/debug commands

## Data Model (BigQuery)
Table: `${GCP_PROJECT_ID}.${BQ_DATASET}.${BQ_TABLE}`

Key fields:
- `chunk_id`: unique row id
- `source_id`: stable per-document id (UUIDv5 derived from `source_uri`)
- `source_uri`: original GCS object URI
- `source_name`, `source_system`: file name + ingest source (`upload`/`drive`)
- `slide_number`, `chunk_index`: reconstruction order
- `content_type`: `slide` / `pdf_page` / `image` / `audio`
- `modalities`: e.g. `["text","image"]`
- `detected_date`: extracted date for temporal filters
- `chunk_text`: canonical searchable text
- `embedding`: vector for semantic retrieval

## Ingestion Paths

### Local upload
- Single: `POST /upload-slides`
- Batch: `POST /upload-slides-batch`

### Google Drive
- `POST /ingest-drive` with `file_id` or `folder_id`
- Uses ADC + Drive readonly scope

### Multimodal behavior
- `.pptx/.pdf`: text extraction + image summaries
- images: Gemini description
- audio: Gemini transcription + summary

## Retrieval + QA
- Query endpoint: `POST /query`
- Flow:
1. Embed query (`RETRIEVAL_QUERY`)
2. BigQuery `VECTOR_SEARCH` over `embedding`
3. Optional date filter inferred from question (e.g. “summer 2025”)
4. Prompt Gemini with retrieved context

## Full-Document Access by ID
- List docs: `GET /documents?limit=...`
- Read doc: `GET /documents/{source_id}`
- Image retrieval: `GET /documents/{source_id}/images`
  - Returns signed URLs
  - For PPTX/PDF extracts embedded images to `gs://.../extracted/{source_id}/...`

## MCP Exposure
Two modes:
- `stdio`: `just mcp-server`
- URL-based: `just mcp-http`, endpoint `http://127.0.0.1:8000/mcp`

Tools exposed:
- `search`
- `list_documents`
- `read_document`
- `list_images`

## Operational Learnings / Gotchas
- ADC quota project must be set:
  - `gcloud auth application-default set-quota-project <PROJECT_ID>`
- Required APIs:
  - BigQuery, Storage, Vertex AI, Drive
- BigQuery schema drift is common:
  - `just schema-migrate` updates existing table safely
- `source_id` vs upload UUID:
  - Use `source_id` for canonical reads
  - Fallback supports old UUID from GCS filename
- MCP stdio appears “stuck” when run manually; this is expected.

## Commands Used Most
- `just run`
- `just schema-migrate`
- `just docs-list 20`
- `just doc-read <SOURCE_ID>`
- `just mcp-http`

## Integration Plan for Existing Project
1. Copy service logic:
   - Port `SlideRAGService` methods from `src/rag_service.py`.
2. Reuse schema:
   - Apply `infra/table_schema.json` to your dataset/table.
3. Map auth/bootstrap:
   - Ensure ADC + quota project + API enablement in your environment.
4. Integrate endpoints:
   - Add `/query`, `/documents/{source_id}`, `/documents/{source_id}/images`.
5. Connect agent layer:
   - Reuse `src/mcp_server.py` or call `/mcp/tools/*` directly.
6. Re-ingest critical corpora:
   - Needed to backfill new metadata fields.

## Current Limitations
- Image/audio understanding uses model summaries (not OCR+ASR specialized pipelines).
- Date extraction is heuristic and English-centric.
- No auth layer on API/MCP endpoints yet (local/dev posture).
- For large scale, add:
  - vector index tuning
  - async ingestion queue
  - signed URL lifecycle policy
  - explicit tenant/project isolation

## Research Value Summary
This implementation validates that a BigQuery-centric RAG stack can support:
- multimodal ingestion,
- metadata-aware retrieval,
- full-document reconstruction by stable id,
- and direct AI-agent tool access via MCP.

