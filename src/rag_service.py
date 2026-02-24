from __future__ import annotations

import io
import re
import textwrap
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from google.auth import default as google_auth_default
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.cloud import bigquery
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from pypdf import PdfReader
from pptx import Presentation

from .config import (
    BQ_DATASET,
    BQ_TABLE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    GCP_LOCATION,
    GCP_PROJECT_ID,
    GCS_BUCKET,
    GEN_MODEL,
)


@dataclass
class SlideChunk:
    source_id: str
    source_uri: str
    source_name: str
    source_system: str
    title: str | None
    slide_number: int
    chunk_index: int
    content_type: str
    modalities: list[str]
    detected_date: str | None
    chunk_text: str


class SlideRAGService:
    def __init__(self) -> None:
        self.bq = bigquery.Client(project=GCP_PROJECT_ID)
        self.storage = storage.Client(project=GCP_PROJECT_ID)
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        self.embedder = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
        self.generator = GenerativeModel(GEN_MODEL)

    def upload_to_gcs(self, filename: str, content: bytes) -> str:
        safe_name = f"{uuid.uuid4()}-{Path(filename).name}"
        bucket = self.storage.bucket(GCS_BUCKET)
        blob = bucket.blob(f"slides/{safe_name}")
        blob.upload_from_string(content)
        return f"gs://{GCS_BUCKET}/slides/{safe_name}"

    def ingest_bytes(self, filename: str, content: bytes, source_system: str = "upload") -> dict:
        source_uri = self.upload_to_gcs(filename, content)
        chunks = self.extract_chunks(filename, content, source_uri, source_system)
        inserted = self.upsert_chunks(chunks)
        return {
            "source_id": self._source_id_from_uri(source_uri),
            "source_uri": source_uri,
            "chunks_inserted": inserted,
        }

    def extract_chunks(
        self, filename: str, content: bytes, source_uri: str, source_system: str = "upload"
    ) -> list[SlideChunk]:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pptx":
            return self._extract_from_pptx(filename, content, source_uri, source_system)
        if suffix == ".pdf":
            return self._extract_from_pdf(filename, content, source_uri, source_system)
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return self._extract_from_image(filename, content, source_uri, source_system, 1)
        if suffix in {".mp3", ".wav", ".m4a"}:
            return self._extract_from_audio(filename, content, source_uri, source_system)
        raise ValueError("Supported: .pptx, .pdf, .png, .jpg, .jpeg, .webp, .mp3, .wav, .m4a.")

    def upsert_chunks(self, chunks: list[SlideChunk]) -> int:
        rows = []
        for chunk in chunks:
            embedding = self._embed_document(chunk.chunk_text)
            rows.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "source_id": chunk.source_id,
                    "source_uri": chunk.source_uri,
                    "source_name": chunk.source_name,
                    "source_system": chunk.source_system,
                    "title": chunk.title,
                    "slide_number": chunk.slide_number,
                    "chunk_index": chunk.chunk_index,
                    "content_type": chunk.content_type,
                    "modalities": chunk.modalities,
                    "detected_date": chunk.detected_date,
                    "chunk_text": chunk.chunk_text,
                    "embedding": embedding,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        table = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
        errors = self.bq.insert_rows_json(table, rows)
        if errors:
            raise RuntimeError(f"BigQuery insert errors: {errors}")
        return len(rows)

    def answer_question(self, question: str, top_k: int = 5) -> dict:
        query_embedding = self._embed_query(question)
        date_start, date_end = self._extract_date_filter(question)
        search_k = min(max(top_k * 5, top_k), 100)
        sql = f"""
        WITH hits AS (
          SELECT
            base.source_uri AS source_uri,
            base.source_id AS source_id,
            base.source_name AS source_name,
            base.source_system AS source_system,
            base.title AS title,
            base.slide_number AS slide_number,
            base.chunk_index AS chunk_index,
            base.content_type AS content_type,
            base.modalities AS modalities,
            base.detected_date AS detected_date,
            base.chunk_text AS chunk_text,
            distance
          FROM VECTOR_SEARCH(
            TABLE `{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`,
            'embedding',
            (SELECT @query_embedding AS embedding),
            top_k => @search_k,
            distance_type => 'COSINE'
          )
        )
        SELECT *
        FROM hits
        WHERE (@date_start IS NULL OR detected_date >= @date_start)
          AND (@date_end IS NULL OR detected_date <= @date_end)
        ORDER BY distance
        LIMIT @top_k
        """
        params = [
            bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", query_embedding),
            bigquery.ScalarQueryParameter("search_k", "INT64", search_k),
            bigquery.ScalarQueryParameter("top_k", "INT64", top_k),
            bigquery.ScalarQueryParameter("date_start", "DATE", date_start),
            bigquery.ScalarQueryParameter("date_end", "DATE", date_end),
        ]
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        hits = list(self.bq.query(sql, job_config=job_config).result())

        if not hits:
            return {"answer": "No relevant content found in indexed slides.", "citations": []}

        context_lines = []
        citations = []
        for h in hits:
            context_lines.append(
                f"[source={h.source_uri}, slide={h.slide_number}] {h.chunk_text}"
            )
            citations.append(
                {
                    "source_uri": h.source_uri,
                    "source_id": h.source_id,
                    "source_name": h.source_name,
                    "source_system": h.source_system,
                    "slide_number": h.slide_number,
                    "chunk_index": h.chunk_index,
                    "title": h.title,
                    "content_type": h.content_type,
                    "modalities": h.modalities,
                    "detected_date": h.detected_date,
                    "distance": h.distance,
                }
            )

        prompt = textwrap.dedent(
            f"""
            You are a precise assistant answering questions from slide content.
            Use only the provided context. If context is insufficient, say so clearly.
            Question: {question}

            Context:
            {'\n'.join(context_lines)}
            """
        )
        response = self.generator.generate_content(prompt)
        answer = response.text if hasattr(response, "text") and response.text else str(response)
        return {"answer": answer, "citations": citations}

    def ingest_drive(
        self,
        file_id: str | None = None,
        folder_id: str | None = None,
        max_files: int = 50,
    ) -> dict:
        if not file_id and not folder_id:
            raise ValueError("Provide file_id or folder_id.")
        if file_id and folder_id:
            raise ValueError("Provide only one of file_id or folder_id.")

        drive = self._drive_service()
        candidates = [self._drive_file_meta(drive, file_id)] if file_id else self._drive_list_folder(
            drive, folder_id, max_files
        )

        indexed = []
        skipped = []
        total_chunks = 0
        for item in candidates:
            normalized = self._normalize_drive_file(item)
            if not normalized:
                skipped.append({"file_id": item["id"], "name": item["name"], "reason": "unsupported file type"})
                continue

            target_name, content = self._download_drive_file(drive, item, normalized)
            result = self.ingest_bytes(target_name, content, source_system="drive")
            total_chunks += result["chunks_inserted"]
            indexed.append(
                {
                    "file_id": item["id"],
                    "name": item["name"],
                    "source_id": result["source_id"],
                    "source_uri": result["source_uri"],
                    "chunks_inserted": result["chunks_inserted"],
                }
            )

        return {
            "indexed_files": len(indexed),
            "indexed_chunks": total_chunks,
            "indexed": indexed,
            "skipped": skipped,
        }

    def _extract_from_pptx(
        self, filename: str, content: bytes, source_uri: str, source_system: str
    ) -> list[SlideChunk]:
        deck = Presentation(io.BytesIO(content))
        chunks: list[SlideChunk] = []
        title = None
        source_name = Path(filename).name

        for idx, slide in enumerate(deck.slides, start=1):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    texts.append(shape.text.strip())
                if hasattr(shape, "image") and getattr(shape, "image", None):
                    image_caption = self._describe_binary(
                        shape.image.blob,
                        "image/jpeg",
                        "Describe this slide image in 3-6 concise bullets.",
                    )
                    texts.append(f"[Image summary] {image_caption}")
            slide_text = "\n".join(t for t in texts if t)
            if idx == 1 and texts:
                title = texts[0]
            chunks.extend(
                self._chunk_slide_text(
                    text=slide_text,
                    source_uri=source_uri,
                    source_name=source_name,
                    source_system=source_system,
                    title=title,
                    slide_number=idx,
                    content_type="slide",
                    modalities=["text", "image"],
                )
            )
        return chunks

    def _extract_from_pdf(
        self, filename: str, content: bytes, source_uri: str, source_system: str
    ) -> list[SlideChunk]:
        reader = PdfReader(io.BytesIO(content))
        chunks: list[SlideChunk] = []
        title = None
        source_name = Path(filename).name

        for idx, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            image_texts = []
            try:
                for img in getattr(page, "images", []):
                    image_texts.append(
                        self._describe_binary(
                            img.data,
                            "image/jpeg",
                            "Describe this page image in 2-5 concise bullets.",
                        )
                    )
            except Exception:
                pass
            if image_texts:
                page_text = f"{page_text}\n[Page image summaries]\n" + "\n".join(image_texts)
            if idx == 1 and page_text:
                title = page_text.splitlines()[0][:180]
            chunks.extend(
                self._chunk_slide_text(
                    text=page_text,
                    source_uri=source_uri,
                    source_name=source_name,
                    source_system=source_system,
                    title=title,
                    slide_number=idx,
                    content_type="pdf_page",
                    modalities=["text", "image"],
                )
            )
        return chunks

    def _extract_from_image(
        self,
        filename: str,
        content: bytes,
        source_uri: str,
        source_system: str,
        slide_number: int,
    ) -> list[SlideChunk]:
        source_name = Path(filename).name
        suffix = Path(filename).suffix.lower()
        mime = "image/jpeg"
        if suffix == ".png":
            mime = "image/png"
        if suffix == ".webp":
            mime = "image/webp"
        text = self._describe_binary(
            content,
            mime,
            "Describe the image in detail. Include visible text, entities, charts, and key scene context.",
        )
        return self._chunk_slide_text(
            text=text,
            source_uri=source_uri,
            source_name=source_name,
            source_system=source_system,
            title=source_name,
            slide_number=slide_number,
            content_type="image",
            modalities=["image", "text"],
        )

    def _extract_from_audio(
        self, filename: str, content: bytes, source_uri: str, source_system: str
    ) -> list[SlideChunk]:
        source_name = Path(filename).name
        suffix = Path(filename).suffix.lower()
        mime_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
        }
        mime = mime_map.get(suffix, "audio/mpeg")
        text = self._describe_binary(
            content,
            mime,
            "Transcribe the audio. Then add a concise summary and any key dates mentioned.",
        )
        return self._chunk_slide_text(
            text=text,
            source_uri=source_uri,
            source_name=source_name,
            source_system=source_system,
            title=source_name,
            slide_number=1,
            content_type="audio",
            modalities=["audio", "text"],
        )

    def _chunk_slide_text(
        self,
        text: str,
        source_uri: str,
        source_name: str,
        source_system: str,
        title: str | None,
        slide_number: int,
        content_type: str,
        modalities: list[str],
    ) -> list[SlideChunk]:
        text = " ".join(text.split())
        if not text:
            return []

        source_id = self._source_id_from_uri(source_uri)
        chunks = []
        for chunk_index, segment in enumerate(self._windowed_chunks(text), start=1):
            chunks.append(
                SlideChunk(
                    source_id=source_id,
                    source_uri=source_uri,
                    source_name=source_name,
                    source_system=source_system,
                    title=title,
                    slide_number=slide_number,
                    chunk_index=chunk_index,
                    content_type=content_type,
                    modalities=modalities,
                    detected_date=self._extract_date_from_text(segment) or self._extract_date_from_text(source_name),
                    chunk_text=segment,
                )
            )
        return chunks

    def _windowed_chunks(self, text: str) -> Iterable[str]:
        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        for start in range(0, len(text), step):
            chunk = text[start : start + CHUNK_SIZE].strip()
            if chunk:
                yield chunk

    def list_documents(self, limit: int = 100) -> list[dict]:
        sql = f"""
        SELECT
          COALESCE(source_id, source_uri) AS source_id,
          ANY_VALUE(source_uri) AS source_uri,
          ANY_VALUE(source_name) AS source_name,
          ANY_VALUE(source_system) AS source_system,
          ANY_VALUE(content_type) AS content_type,
          ARRAY(
            SELECT DISTINCT m
            FROM UNNEST(ARRAY_CONCAT_AGG(IFNULL(modalities, []))) AS m
          ) AS modalities,
          MIN(detected_date) AS first_detected_date,
          MAX(detected_date) AS last_detected_date,
          COUNT(*) AS chunks
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
        GROUP BY COALESCE(source_id, source_uri)
        ORDER BY MAX(created_at) DESC
        LIMIT @limit
        """
        params = [bigquery.ScalarQueryParameter("limit", "INT64", limit)]
        rows = self.bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        return [dict(r) for r in rows]

    def get_document(self, source_id: str) -> dict:
        sql = f"""
        SELECT
          source_id, source_uri, source_name, source_system, title,
          slide_number, chunk_index, content_type, modalities, detected_date, chunk_text
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE source_id = @source_id
        ORDER BY slide_number, chunk_index
        """
        params = [bigquery.ScalarQueryParameter("source_id", "STRING", source_id)]
        rows = list(self.bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result())
        if not rows:
            fallback_sql = f"""
            SELECT
              source_id, source_uri, source_name, source_system, title,
              slide_number, chunk_index, content_type, modalities, detected_date, chunk_text
            FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
            WHERE source_uri = @source_uri
            ORDER BY slide_number, chunk_index
            """
            fallback_params = [bigquery.ScalarQueryParameter("source_uri", "STRING", source_id)]
            rows = list(
                self.bq.query(
                    fallback_sql,
                    job_config=bigquery.QueryJobConfig(query_parameters=fallback_params),
                ).result()
            )
        if not rows and re.fullmatch(r"[0-9a-fA-F-]{36}", source_id):
            upload_id_sql = f"""
            SELECT
              source_id, source_uri, source_name, source_system, title,
              slide_number, chunk_index, content_type, modalities, detected_date, chunk_text
            FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
            WHERE REGEXP_CONTAINS(source_uri, @pattern)
            ORDER BY slide_number, chunk_index
            """
            pattern = rf"/slides/{re.escape(source_id)}-"
            rows = list(
                self.bq.query(
                    upload_id_sql,
                    job_config=bigquery.QueryJobConfig(
                        query_parameters=[bigquery.ScalarQueryParameter("pattern", "STRING", pattern)]
                    ),
                ).result()
            )
        if not rows:
            raise ValueError(f"No document found for source_id={source_id}")

        sections = []
        current_slide = None
        current_texts: list[str] = []
        for r in rows:
            if current_slide is None:
                current_slide = r.slide_number
            if r.slide_number != current_slide:
                sections.append({"slide_number": current_slide, "text": " ".join(current_texts).strip()})
                current_slide = r.slide_number
                current_texts = []
            current_texts.append(r.chunk_text)
        if current_slide is not None:
            sections.append({"slide_number": current_slide, "text": " ".join(current_texts).strip()})

        full_text = "\n\n".join(
            f"[slide {s['slide_number']}]\n{s['text']}" if s["slide_number"] else s["text"] for s in sections
        ).strip()
        head = rows[0]
        return {
            "source_id": head.source_id or self._source_id_from_uri(head.source_uri),
            "source_uri": head.source_uri,
            "source_name": head.source_name,
            "source_system": head.source_system,
            "title": head.title,
            "content_type": head.content_type,
            "sections": sections,
            "full_text": full_text,
            "chunk_count": len(rows),
        }

    def list_document_images(self, source_id: str, max_images: int = 50) -> dict:
        doc = self.get_document(source_id)
        source_uri = doc["source_uri"]
        source_name = doc["source_name"] or "source"
        suffix = Path(source_name).suffix.lower()

        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return {
                "source_id": doc["source_id"],
                "images": [self._signed_asset(source_uri, "source-image")],
            }

        content = self._download_gcs_uri(source_uri)
        source_id_norm = doc["source_id"]
        images: list[dict] = []

        if suffix == ".pptx":
            deck = Presentation(io.BytesIO(content))
            for slide_number, slide in enumerate(deck.slides, start=1):
                image_index = 0
                for shape in slide.shapes:
                    if not (hasattr(shape, "image") and getattr(shape, "image", None)):
                        continue
                    image_index += 1
                    ext = (shape.image.ext or "jpg").lower()
                    if ext == "jpeg":
                        ext = "jpg"
                    blob_path = f"extracted/{source_id_norm}/slide-{slide_number}-img-{image_index}.{ext}"
                    gs_uri = self._upload_extracted(blob_path, shape.image.blob)
                    images.append(self._signed_asset(gs_uri, f"slide-{slide_number}-img-{image_index}"))
                    if len(images) >= max_images:
                        break
                if len(images) >= max_images:
                    break

        elif suffix == ".pdf":
            reader = PdfReader(io.BytesIO(content))
            for page_number, page in enumerate(reader.pages, start=1):
                page_images = getattr(page, "images", [])
                for image_index, img in enumerate(page_images, start=1):
                    blob_path = f"extracted/{source_id_norm}/page-{page_number}-img-{image_index}.jpg"
                    gs_uri = self._upload_extracted(blob_path, img.data)
                    images.append(self._signed_asset(gs_uri, f"page-{page_number}-img-{image_index}"))
                    if len(images) >= max_images:
                        break
                if len(images) >= max_images:
                    break

        return {"source_id": doc["source_id"], "images": images}

    def _embed_document(self, text: str) -> list[float]:
        inputs = [TextEmbeddingInput(text=text, task_type="RETRIEVAL_DOCUMENT")]
        output = self.embedder.get_embeddings(inputs)[0]
        return output.values

    def _embed_query(self, text: str) -> list[float]:
        inputs = [TextEmbeddingInput(text=text, task_type="RETRIEVAL_QUERY")]
        output = self.embedder.get_embeddings(inputs)[0]
        return output.values

    def _drive_service(self):
        creds, _ = google_auth_default(
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    def _drive_file_meta(self, drive, file_id: str) -> dict:
        return (
            drive.files()
            .get(
                fileId=file_id,
                fields="id,name,mimeType",
                supportsAllDrives=True,
            )
            .execute()
        )

    def _drive_list_folder(self, drive, folder_id: str, max_files: int) -> list[dict]:
        query = f"'{folder_id}' in parents and trashed=false"
        items = []
        page_token = None
        while len(items) < max_files:
            resp = (
                drive.files()
                .list(
                    q=query,
                    fields="nextPageToken,files(id,name,mimeType)",
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                    pageSize=min(100, max_files - len(items)),
                    pageToken=page_token,
                )
                .execute()
            )
            items.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return items

    def _normalize_drive_file(self, item: dict) -> str | None:
        mime = item["mimeType"]
        name = item["name"]
        if mime == "application/vnd.google-apps.presentation":
            return f"{name}.pptx" if not name.lower().endswith(".pptx") else name
        if mime == "application/pdf":
            return name if name.lower().endswith(".pdf") else f"{name}.pdf"
        if (
            mime
            == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        ):
            return name if name.lower().endswith(".pptx") else f"{name}.pptx"
        return None

    def _download_drive_file(self, drive, item: dict, target_name: str) -> tuple[str, bytes]:
        mime = item["mimeType"]
        file_id = item["id"]
        if mime == "application/vnd.google-apps.presentation":
            req = (
                drive.files()
                .export_media(
                    fileId=file_id,
                    mimeType="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
            )
        else:
            req = drive.files().get_media(fileId=file_id, supportsAllDrives=True)

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return target_name, buffer.getvalue()

    def _describe_binary(self, data: bytes, mime_type: str, prompt: str) -> str:
        response = self.generator.generate_content([prompt, Part.from_data(data=data, mime_type=mime_type)])
        text = response.text if hasattr(response, "text") and response.text else str(response)
        return text.strip()

    def _extract_date_from_text(self, text: str) -> str | None:
        if not text:
            return None
        for pattern in [
            r"\b(20\d{2})-(\d{2})-(\d{2})\b",
            r"\b(20\d{2})/(\d{2})/(\d{2})\b",
            r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b",
        ]:
            m = re.search(pattern, text)
            if not m:
                continue
            try:
                if pattern.startswith(r"\b(\d{1,2})"):
                    mm, dd, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
                else:
                    yyyy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return date(yyyy, mm, dd).isoformat()
            except ValueError:
                pass

        month_match = re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2})\b",
            text,
            flags=re.IGNORECASE,
        )
        if month_match:
            month = {
                "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
            }[month_match.group(1).lower()]
            year = int(month_match.group(2))
            return date(year, month, 1).isoformat()
        return None

    def _extract_date_filter(self, question: str) -> tuple[str | None, str | None]:
        q = question.lower()
        year_match = re.search(r"\b(20\d{2})\b", q)
        year = int(year_match.group(1)) if year_match else None

        season_match = re.search(r"\b(beginning of\s+)?(spring|summer|fall|autumn|winter)\s+(20\d{2})\b", q)
        if season_match:
            beginning = bool(season_match.group(1))
            season = season_match.group(2)
            yr = int(season_match.group(3))
            ranges = {
                "spring": (date(yr, 3, 1), date(yr, 5, 31)),
                "summer": (date(yr, 6, 1), date(yr, 8, 31)),
                "fall": (date(yr, 9, 1), date(yr, 11, 30)),
                "autumn": (date(yr, 9, 1), date(yr, 11, 30)),
                "winter": (date(yr, 12, 1), date(yr + 1, 2, 28)),
            }
            start, end = ranges[season]
            if beginning:
                end = min(end, start.replace(day=1) + (date(start.year, start.month, 28) - date(start.year, start.month, 1)))
                end = date(start.year, min(start.month + 1, 12), 15)
            return start.isoformat(), end.isoformat()

        month_year = re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2})\b",
            q,
        )
        if month_year:
            month = {
                "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
            }[month_year.group(1)]
            yr = int(month_year.group(2))
            start = date(yr, month, 1)
            end = date(yr, month, 28)
            return start.isoformat(), end.isoformat()

        if year and ("in " + str(year) in q or "from " + str(year) in q):
            return date(year, 1, 1).isoformat(), date(year, 12, 31).isoformat()
        return None, None

    def _source_id_from_uri(self, source_uri: str) -> str:
        return uuid.uuid5(uuid.NAMESPACE_URL, source_uri).hex

    def _download_gcs_uri(self, gs_uri: str) -> bytes:
        bucket_name, blob_name = self._parse_gs_uri(gs_uri)
        blob = self.storage.bucket(bucket_name).blob(blob_name)
        return blob.download_as_bytes()

    def _upload_extracted(self, blob_path: str, data: bytes) -> str:
        bucket = self.storage.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_path)
        if not blob.exists():
            blob.upload_from_string(data)
        return f"gs://{GCS_BUCKET}/{blob_path}"

    def _signed_asset(self, gs_uri: str, label: str) -> dict:
        bucket_name, blob_name = self._parse_gs_uri(gs_uri)
        blob = self.storage.bucket(bucket_name).blob(blob_name)
        url = blob.generate_signed_url(version="v4", expiration=timedelta(hours=1), method="GET")
        return {"label": label, "gs_uri": gs_uri, "signed_url": url}

    def _parse_gs_uri(self, gs_uri: str) -> tuple[str, str]:
        parsed = urlparse(gs_uri)
        if parsed.scheme != "gs":
            raise ValueError(f"Invalid gs URI: {gs_uri}")
        return parsed.netloc, parsed.path.lstrip("/")
