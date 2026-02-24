from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse

from .rag_service import SlideRAGService

app = FastAPI(title="BigQuery Slide RAG")

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BigQuery Slide RAG</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #13223b;
      --muted: #4a5a79;
      --accent: #006a6a;
      --accent-2: #0d8c8c;
      --border: #dce3f1;
      --ok: #1e7b34;
      --err: #a11b2e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at 10% 10%, #deecff 0%, transparent 45%),
        radial-gradient(circle at 90% 0%, #d2f5ef 0%, transparent 35%),
        var(--bg);
    }
    .wrap {
      max-width: 980px;
      margin: 0 auto;
      padding: 24px 16px 48px;
    }
    h1 {
      margin: 4px 0 8px;
      font-size: 34px;
      letter-spacing: 0.2px;
    }
    .subtitle {
      margin: 0 0 22px;
      color: var(--muted);
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 8px 24px rgba(16, 35, 67, 0.06);
    }
    .row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    input[type="file"],
    input[type="number"],
    input[type="text"],
    textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
      color: var(--text);
      background: #fff;
    }
    textarea {
      min-height: 92px;
      resize: vertical;
    }
    .btn {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      font-weight: 600;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
    }
    .btn:hover { background: var(--accent-2); }
    .small {
      font-size: 13px;
      color: var(--muted);
    }
    .status {
      margin-top: 10px;
      min-height: 20px;
      font-size: 14px;
    }
    .ok { color: var(--ok); }
    .err { color: var(--err); }
    .answer {
      white-space: pre-wrap;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      background: #fdfefe;
      min-height: 120px;
    }
    .citations {
      margin-top: 12px;
      font-size: 13px;
      color: var(--muted);
    }
    .citation {
      padding: 6px 0;
      border-bottom: 1px dashed var(--border);
    }
    .citation:last-child {
      border-bottom: 0;
    }
    @media (max-width: 640px) {
      h1 { font-size: 28px; }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <h1>BigQuery Slide RAG</h1>
    <p class="subtitle">Upload one or many slide decks, or ingest directly from Google Drive.</p>

    <section class="grid">
      <div class="card">
        <h2>1) Upload Slides</h2>
        <p class="small">Supported: .pptx, .pdf, .png, .jpg, .jpeg, .webp, .mp3, .wav, .m4a</p>
        <div class="row">
          <input id="fileInput" type="file" accept=".pptx,.pdf,.png,.jpg,.jpeg,.webp,.mp3,.wav,.m4a" multiple />
          <button id="uploadBtn" class="btn" type="button">Upload and Index File(s)</button>
        </div>
        <div id="uploadStatus" class="status"></div>
        <hr style="border:0;border-top:1px solid var(--border);margin:14px 0;">
        <p class="small">Google Drive ingest: provide a file ID or folder ID.</p>
        <div class="row">
          <input id="driveFileId" type="text" placeholder="Drive file ID (optional)" />
          <input id="driveFolderId" type="text" placeholder="Drive folder ID (optional)" />
          <input id="driveMax" type="number" min="1" max="500" value="50" style="width: 96px;" />
          <button id="driveBtn" class="btn" type="button">Ingest from Drive</button>
        </div>
        <div id="driveStatus" class="status"></div>
      </div>

      <div class="card">
        <h2>2) Ask Questions</h2>
        <div class="row">
          <label for="topK">Top K:</label>
          <input id="topK" type="number" min="1" max="20" value="5" style="width: 96px;" />
        </div>
        <textarea id="question" placeholder="Ask something about your uploaded slides..."></textarea>
        <div class="row">
          <button id="askBtn" class="btn" type="button">Ask</button>
        </div>
        <div id="queryStatus" class="status"></div>
        <h3>Answer</h3>
        <div id="answer" class="answer"></div>
        <div id="citations" class="citations"></div>
      </div>

      <div class="card">
        <h2>3) Read Full Document by ID</h2>
        <p class="small">Use the stable <code>source_id</code> returned by upload or drive ingest.</p>
        <div class="row">
          <input id="docIdInput" type="text" placeholder="source_id" />
          <button id="readDocBtn" class="btn" type="button">Read Document</button>
          <button id="listDocsBtn" class="btn" type="button">List Recent IDs</button>
          <button id="listImagesBtn" class="btn" type="button">Get Images</button>
        </div>
        <div id="docStatus" class="status"></div>
        <div id="docResult" class="answer"></div>
        <div id="imageGallery" class="citations"></div>
      </div>
    </section>
  </main>

  <script>
    const fileInput = document.getElementById("fileInput");
    const uploadBtn = document.getElementById("uploadBtn");
    const uploadStatus = document.getElementById("uploadStatus");
    const askBtn = document.getElementById("askBtn");
    const driveBtn = document.getElementById("driveBtn");
    const driveFileIdEl = document.getElementById("driveFileId");
    const driveFolderIdEl = document.getElementById("driveFolderId");
    const driveMaxEl = document.getElementById("driveMax");
    const driveStatus = document.getElementById("driveStatus");
    const questionEl = document.getElementById("question");
    const topKEl = document.getElementById("topK");
    const queryStatus = document.getElementById("queryStatus");
    const answerEl = document.getElementById("answer");
    const citationsEl = document.getElementById("citations");
    const docIdInput = document.getElementById("docIdInput");
    const readDocBtn = document.getElementById("readDocBtn");
    const listDocsBtn = document.getElementById("listDocsBtn");
    const listImagesBtn = document.getElementById("listImagesBtn");
    const docStatus = document.getElementById("docStatus");
    const docResult = document.getElementById("docResult");
    const imageGallery = document.getElementById("imageGallery");

    function setStatus(el, text, ok = true) {
      el.textContent = text;
      el.className = "status " + (ok ? "ok" : "err");
    }

    uploadBtn.addEventListener("click", async () => {
      const files = Array.from(fileInput.files || []);
      if (!files.length) {
        setStatus(uploadStatus, "Choose one or more supported files first.", false);
        return;
      }

      const form = new FormData();
      for (const file of files) form.append(files.length > 1 ? "files" : "file", file);
      setStatus(uploadStatus, "Uploading and indexing " + files.length + " file(s)...", true);

      try {
        const endpoint = files.length > 1 ? "/upload-slides-batch" : "/upload-slides";
        const res = await fetch(endpoint, { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) {
          setStatus(uploadStatus, data.detail || "Upload failed.", false);
          return;
        }
        if (files.length > 1) {
          setStatus(uploadStatus, "Indexed " + data.chunks_indexed + " chunks across " + data.files_indexed + " files.", true);
          const ids = (data.results || []).map((r) => r.source_id).filter(Boolean);
          if (ids.length) {
            docResult.textContent = "source_id values:\\n" + ids.join("\\n");
          }
        } else {
          setStatus(uploadStatus, "Indexed " + data.chunks_inserted + " chunks from " + (data.source_uri || files[0].name), true);
          if (data.source_id) {
            docResult.textContent = "source_id:\\n" + data.source_id;
          }
        }
        const result = files.length > 1 ? (data.results || [])[0] : data;
        if (result && result.source_id) {
          docIdInput.value = result.source_id;
        }
      } catch (err) {
        setStatus(uploadStatus, "Upload error: " + err.message, false);
      }
    });

    driveBtn.addEventListener("click", async () => {
      const file_id = driveFileIdEl.value.trim() || null;
      const folder_id = driveFolderIdEl.value.trim() || null;
      const max_files = Number(driveMaxEl.value || "50");
      if (!file_id && !folder_id) {
        setStatus(driveStatus, "Provide a Drive file ID or folder ID.", false);
        return;
      }
      if (file_id && folder_id) {
        setStatus(driveStatus, "Use file ID or folder ID, not both.", false);
        return;
      }
      setStatus(driveStatus, "Downloading from Drive and indexing...", true);
      try {
        const res = await fetch("/ingest-drive", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file_id, folder_id, max_files }),
        });
        const data = await res.json();
        if (!res.ok) {
          setStatus(driveStatus, data.detail || "Drive ingest failed.", false);
          return;
        }
        setStatus(driveStatus, "Indexed " + data.indexed_chunks + " chunks from " + data.indexed_files + " Drive file(s).", true);
        if (data.indexed && data.indexed.length && data.indexed[0].source_id) {
          docIdInput.value = data.indexed[0].source_id;
        }
      } catch (err) {
        setStatus(driveStatus, "Drive ingest error: " + err.message, false);
      }
    });

    askBtn.addEventListener("click", async () => {
      const question = questionEl.value.trim();
      const topK = Number(topKEl.value || "5");
      if (question.length < 3) {
        setStatus(queryStatus, "Enter a longer question.", false);
        return;
      }
      setStatus(queryStatus, "Querying...", true);
      answerEl.textContent = "";
      citationsEl.innerHTML = "";

      try {
        const res = await fetch("/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, top_k: topK }),
        });
        const data = await res.json();
        if (!res.ok) {
          setStatus(queryStatus, data.detail || "Query failed.", false);
          return;
        }

        setStatus(queryStatus, "Done.", true);
        answerEl.textContent = data.answer || "";
        const citations = data.citations || [];
        if (!citations.length) {
          citationsEl.textContent = "No citations.";
          return;
        }
          citationsEl.innerHTML = "<h3>Citations</h3>" + citations.map((c) =>
          '<div class="citation">' +
          "<strong>Slide " + (c.slide_number ?? "?") + "</strong> - " +
          (c.title || "Untitled") + "<br/>" +
          "<span>source_id=" + (c.source_id || "n/a") + "</span><br/>" +
          "<span>type=" + (c.content_type || "unknown") + ", date=" + (c.detected_date || "n/a") + "</span><br/>" +
          '<code>' + (c.source_uri || "") + "</code>" +
          "</div>"
        ).join("");
        if (citations[0] && citations[0].source_id) {
          docIdInput.value = citations[0].source_id;
        }
      } catch (err) {
        setStatus(queryStatus, "Query error: " + err.message, false);
      }
    });

    readDocBtn.addEventListener("click", async () => {
      const sourceId = docIdInput.value.trim();
      if (!sourceId) {
        setStatus(docStatus, "Enter a source_id first.", false);
        return;
      }
      setStatus(docStatus, "Loading full document...", true);
      try {
        const res = await fetch("/documents/" + encodeURIComponent(sourceId));
        const data = await res.json();
        if (!res.ok) {
          setStatus(docStatus, data.detail || "Failed to load document.", false);
          return;
        }
        setStatus(docStatus, "Loaded " + data.chunk_count + " chunks from " + (data.source_name || sourceId), true);
        docResult.textContent = data.full_text || "";
        imageGallery.innerHTML = "";
      } catch (err) {
        setStatus(docStatus, "Read error: " + err.message, false);
      }
    });

    listDocsBtn.addEventListener("click", async () => {
      setStatus(docStatus, "Loading recent documents...", true);
      try {
        const res = await fetch("/documents?limit=20");
        const data = await res.json();
        if (!res.ok) {
          setStatus(docStatus, data.detail || "Failed to list documents.", false);
          return;
        }
        const docs = data.documents || [];
        if (!docs.length) {
          setStatus(docStatus, "No documents yet.", false);
          return;
        }
        const preview = docs.map((d) => d.source_id + " | " + (d.source_name || "unknown")).join("\\n");
        docResult.textContent = preview;
        imageGallery.innerHTML = "";
        setStatus(docStatus, "Loaded " + docs.length + " document IDs.", true);
      } catch (err) {
        setStatus(docStatus, "List error: " + err.message, false);
      }
    });

    listImagesBtn.addEventListener("click", async () => {
      const sourceId = docIdInput.value.trim();
      if (!sourceId) {
        setStatus(docStatus, "Enter a source_id first.", false);
        return;
      }
      setStatus(docStatus, "Loading document images...", true);
      imageGallery.innerHTML = "";
      try {
        const res = await fetch("/documents/" + encodeURIComponent(sourceId) + "/images?max_images=30");
        const data = await res.json();
        if (!res.ok) {
          setStatus(docStatus, data.detail || "Failed to load images.", false);
          return;
        }
        const imgs = data.images || [];
        if (!imgs.length) {
          setStatus(docStatus, "No images found for this source.", false);
          return;
        }
        imageGallery.innerHTML = "<h3>Images</h3>" + imgs.map((img) =>
          '<div class="citation">' +
          "<strong>" + (img.label || "image") + "</strong><br/>" +
          '<a href="' + img.signed_url + '" target="_blank" rel="noopener">open image</a><br/>' +
          '<code>' + (img.gs_uri || "") + "</code>" +
          "</div>"
        ).join("");
        setStatus(docStatus, "Loaded " + imgs.length + " images.", true);
      } catch (err) {
        setStatus(docStatus, "Image load error: " + err.message, false);
      }
    });
  </script>
</body>
</html>
"""


@lru_cache
def get_rag() -> SlideRAGService:
    return SlideRAGService()


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


class DriveIngestRequest(BaseModel):
    file_id: str | None = None
    folder_id: str | None = None
    max_files: int = Field(default=50, ge=1, le=500)


class DocumentReadRequest(BaseModel):
    source_id: str = Field(min_length=8)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return INDEX_HTML


@app.post("/upload-slides")
async def upload_slides(file: UploadFile = File(...)) -> dict:
    return await _ingest_uploaded_file(file)


@app.post("/upload-slides-batch")
async def upload_slides_batch(files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    results = []
    total_chunks = 0
    for file in files:
        result = await _ingest_uploaded_file(file)
        results.append(result)
        total_chunks += result["chunks_inserted"]
    return {
        "files_indexed": len(results),
        "chunks_indexed": total_chunks,
        "results": results,
    }


@app.post("/ingest-drive")
def ingest_drive(req: DriveIngestRequest) -> dict:
    try:
        rag = get_rag()
        return rag.ingest_drive(req.file_id, req.folder_id, req.max_files)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@app.get("/documents")
def list_documents(limit: int = 100) -> dict:
    try:
        rag = get_rag()
        return {"documents": rag.list_documents(limit=limit)}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@app.get("/documents/{source_id}")
def get_document(source_id: str) -> dict:
    try:
        rag = get_rag()
        return rag.get_document(source_id)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@app.get("/documents/{source_id}/images")
def get_document_images(source_id: str, max_images: int = 50) -> dict:
    try:
        rag = get_rag()
        return rag.list_document_images(source_id, max_images=max_images)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@app.post("/mcp/tools/read_document")
def mcp_read_document(req: DocumentReadRequest) -> dict:
    return get_document(req.source_id)


@app.post("/mcp/tools/list_documents")
def mcp_list_documents(limit: int = 100) -> dict:
    return list_documents(limit)


@app.post("/mcp/tools/list_images")
def mcp_list_images(req: DocumentReadRequest) -> dict:
    return get_document_images(req.source_id)


@app.post("/mcp/tools/search")
def mcp_search(req: QueryRequest) -> dict:
    return query_slides(req)


async def _ingest_uploaded_file(file: UploadFile) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        rag = get_rag()
        result = rag.ingest_bytes(file.filename, content, source_system="upload")
        return {"filename": file.filename, **result}
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@app.post("/query")
def query_slides(req: QueryRequest) -> dict:
    try:
        rag = get_rag()
        return rag.answer_question(req.question, req.top_k)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err
