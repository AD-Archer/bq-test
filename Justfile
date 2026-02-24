set dotenv-load := true

default:
    @just --list

check-tools:
    @command -v uv >/dev/null || (echo "uv is required" && exit 1)
    @command -v just >/dev/null || (echo "just is required" && exit 1)
    @command -v gcloud >/dev/null || (echo "gcloud is required" && exit 1)
    @command -v bq >/dev/null || (echo "bq is required" && exit 1)
    @command -v gsutil >/dev/null || (echo "gsutil is required" && exit 1)

auth:
    gcloud auth login
    gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/drive.readonly

adc-quota project_id:
    gcloud auth application-default set-quota-project {{project_id}}

project project_id:
    gcloud config set project {{project_id}}

project-show:
    gcloud config get-value project

enable-apis:
    gcloud services enable \
      bigquery.googleapis.com \
      storage.googleapis.com \
      aiplatform.googleapis.com \
      drive.googleapis.com

gcp-setup project_id:
    just project {{project_id}}
    just enable-apis
    just adc-quota {{project_id}}

bucket-create bucket location="us-central1":
    gsutil mb -l {{location}} gs://{{bucket}}

sync: check-tools
    uv sync

run: sync
    uv run --env-file .env uvicorn src.main:app --reload --port 8080

schema:
    bq --location=${GCP_LOCATION} mk --dataset ${GCP_PROJECT_ID}:${BQ_DATASET} || true
    bq mk --table ${GCP_PROJECT_ID}:${BQ_DATASET}.${BQ_TABLE} infra/table_schema.json || true

schema-reset:
    bq --location=${GCP_LOCATION} mk --dataset ${GCP_PROJECT_ID}:${BQ_DATASET} || true
    bq rm -f -t ${GCP_PROJECT_ID}:${BQ_DATASET}.${BQ_TABLE} || true
    bq mk --table ${GCP_PROJECT_ID}:${BQ_DATASET}.${BQ_TABLE} infra/table_schema.json

schema-show:
    bq show --schema --format=prettyjson ${GCP_PROJECT_ID}:${BQ_DATASET}.${BQ_TABLE}

schema-migrate:
    bq query --use_legacy_sql=false "ALTER TABLE \`${GCP_PROJECT_ID}.${BQ_DATASET}.${BQ_TABLE}\` ADD COLUMN IF NOT EXISTS source_id STRING"
    bq query --use_legacy_sql=false "ALTER TABLE \`${GCP_PROJECT_ID}.${BQ_DATASET}.${BQ_TABLE}\` ADD COLUMN IF NOT EXISTS source_name STRING"
    bq query --use_legacy_sql=false "ALTER TABLE \`${GCP_PROJECT_ID}.${BQ_DATASET}.${BQ_TABLE}\` ADD COLUMN IF NOT EXISTS source_system STRING"
    bq query --use_legacy_sql=false "ALTER TABLE \`${GCP_PROJECT_ID}.${BQ_DATASET}.${BQ_TABLE}\` ADD COLUMN IF NOT EXISTS chunk_index INT64"
    bq query --use_legacy_sql=false "ALTER TABLE \`${GCP_PROJECT_ID}.${BQ_DATASET}.${BQ_TABLE}\` ADD COLUMN IF NOT EXISTS content_type STRING"
    bq query --use_legacy_sql=false "ALTER TABLE \`${GCP_PROJECT_ID}.${BQ_DATASET}.${BQ_TABLE}\` ADD COLUMN IF NOT EXISTS modalities ARRAY<STRING>"
    bq query --use_legacy_sql=false "ALTER TABLE \`${GCP_PROJECT_ID}.${BQ_DATASET}.${BQ_TABLE}\` ADD COLUMN IF NOT EXISTS detected_date DATE"

health:
    curl -sS http://127.0.0.1:8080/health

upload file:
    curl -sS -X POST "http://127.0.0.1:8080/upload-slides" \
      -F "file=@{{file}}"

ask question top_k="5":
    curl -sS -X POST "http://127.0.0.1:8080/query" \
      -H "Content-Type: application/json" \
      -d '{"question":"{{question}}","top_k":{{top_k}}}'

ingest-drive-file file_id:
    curl -sS -X POST "http://127.0.0.1:8080/ingest-drive" \
      -H "Content-Type: application/json" \
      -d '{"file_id":"{{file_id}}"}'

ingest-drive-folder folder_id max_files="50":
    curl -sS -X POST "http://127.0.0.1:8080/ingest-drive" \
      -H "Content-Type: application/json" \
      -d '{"folder_id":"{{folder_id}}","max_files":{{max_files}}}'

docs-list limit="20":
    curl -sS "http://127.0.0.1:8080/documents?limit={{limit}}"

doc-read source_id:
    curl -sS "http://127.0.0.1:8080/documents/{{source_id}}"

mcp-search question top_k="5":
    curl -sS -X POST "http://127.0.0.1:8080/mcp/tools/search" \
      -H "Content-Type: application/json" \
      -d '{"question":"{{question}}","top_k":{{top_k}}}'

mcp-read source_id:
    curl -sS -X POST "http://127.0.0.1:8080/mcp/tools/read_document" \
      -H "Content-Type: application/json" \
      -d '{"source_id":"{{source_id}}"}'

mcp-images source_id:
    curl -sS -X POST "http://127.0.0.1:8080/mcp/tools/list_images" \
      -H "Content-Type: application/json" \
      -d '{"source_id":"{{source_id}}"}'

mcp-server:
    uv run --env-file .env python3 -m src.mcp_server

mcp-http:
    MCP_TRANSPORT=streamable-http MCP_HOST=127.0.0.1 MCP_PORT=8000 uv run --env-file .env python3 -m src.mcp_server
