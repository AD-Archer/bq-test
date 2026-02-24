-- Replace PROJECT_ID and DATASET before running.
CREATE SCHEMA IF NOT EXISTS `PROJECT_ID.DATASET`;

CREATE TABLE IF NOT EXISTS `PROJECT_ID.DATASET.slide_chunks` (
  chunk_id STRING NOT NULL,
  source_id STRING,
  source_uri STRING NOT NULL,
  source_name STRING,
  source_system STRING,
  title STRING,
  slide_number INT64,
  chunk_index INT64,
  content_type STRING,
  modalities ARRAY<STRING>,
  detected_date DATE,
  media_start_seconds FLOAT64,
  media_end_seconds FLOAT64,
  speech_style STRING,
  word_timestamps_json STRING,
  chunk_text STRING NOT NULL,
  embedding ARRAY<FLOAT64> NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
);
