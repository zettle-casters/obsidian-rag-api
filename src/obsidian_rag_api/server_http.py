"""HTTP transport for MCP Server."""

import asyncio
import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .auth import get_user_by_mcp_token
from .db import SessionLocal
from .server import list_tools, call_tool_for_user


app = FastAPI(
    title="Obsidian RAG MCP Server (HTTP)",
    description="MCP server with HTTP transport for Obsidian RAG",
    version="0.1.0",
)


class MCPRequest(BaseModel):
    """Generic MCP request."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] | None = None


class MCPResponse(BaseModel):
    """Generic MCP response."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "server": "obsidian-rag-mcp"}


def _extract_token(raw_request: Request) -> str | None:
    auth_header = raw_request.headers.get("Authorization") or raw_request.headers.get("X-MCP-Token")
    if not auth_header:
        return None
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return auth_header.strip()


def _mcp_error(request_id: int | str | None, code: int, message: str) -> MCPResponse:
    return MCPResponse(id=request_id, error={"code": code, "message": message})


def _require_user(raw_request: Request, request_id: int | str | None):
    token = _extract_token(raw_request)
    if not token:
        return None, _mcp_error(request_id, 401, "Missing MCP token")
    with SessionLocal() as db:
        user = get_user_by_mcp_token(token, db)
    if user is None:
        return None, _mcp_error(request_id, 401, "Invalid MCP token")
    return user, None


@app.post("/mcp")
async def handle_mcp(request: MCPRequest, raw_request: Request):
    """Handle MCP JSON-RPC requests over HTTP."""
    try:
        if request.method == "tools/list":
            _, error = _require_user(raw_request, request.id)
            if error:
                return error
            tools = await list_tools()
            return MCPResponse(
                id=request.id,
                result={
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.inputSchema,
                        }
                        for t in tools
                    ]
                },
            )

        elif request.method == "tools/call":
            user, error = _require_user(raw_request, request.id)
            if error:
                return error

            params = request.params or {}
            name = params.get("name", "")
            arguments = params.get("arguments", {})

            result = await call_tool_for_user(name, arguments, user)

            return MCPResponse(
                id=request.id,
                result={
                    "content": [
                        {"type": c.type, "text": c.text} for c in result
                    ]
                },
            )

        elif request.method == "initialize":
            return MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "obsidian-rag",
                        "version": "0.1.0",
                    },
                },
            )

        else:
            return _mcp_error(request.id, -32601, f"Method not found: {request.method}")

    except Exception as e:
        return _mcp_error(request.id, -32603, str(e))


def main():
    """Run the HTTP MCP server."""
    import uvicorn
    from .config import settings

    uvicorn.run(
        "obsidian_rag_mcp.server_http:app",
        host=settings.host,
        port=settings.port + 1,  # Use port + 1 for MCP HTTP
        reload=False,
    )


if __name__ == "__main__":
    main()
