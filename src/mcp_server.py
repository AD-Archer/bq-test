from __future__ import annotations

import os

import requests
from mcp.server.fastmcp import FastMCP


API_BASE = os.getenv("RAG_API_BASE", "http://127.0.0.1:8080").rstrip("/")
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
mcp = FastMCP("bq-slide-rag", host=MCP_HOST, port=MCP_PORT, streamable_http_path="/mcp")


def _call(method: str, path: str, payload: dict | None = None):
    url = f"{API_BASE}{path}"
    if method == "GET":
        resp = requests.get(url, timeout=120)
    else:
        resp = requests.post(url, json=payload or {}, timeout=180)
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
def search(question: str, top_k: int = 5) -> dict:
    return _call("POST", "/mcp/tools/search", {"question": question, "top_k": top_k})


@mcp.tool()
def list_documents(limit: int = 20) -> dict:
    return _call("POST", f"/mcp/tools/list_documents?limit={limit}")


@mcp.tool()
def read_document(source_id: str) -> dict:
    return _call("POST", "/mcp/tools/read_document", {"source_id": source_id})


@mcp.tool()
def list_images(source_id: str) -> dict:
    return _call("POST", "/mcp/tools/list_images", {"source_id": source_id})


if __name__ == "__main__":
    mcp.run(transport=MCP_TRANSPORT)
