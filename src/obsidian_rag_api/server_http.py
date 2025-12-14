"""HTTP transport for MCP Server."""

import asyncio
import json
from typing import Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .server import mcp_server, list_tools, call_tool


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


@app.post("/mcp")
async def handle_mcp(request: MCPRequest):
    """Handle MCP JSON-RPC requests over HTTP."""
    try:
        if request.method == "tools/list":
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
            params = request.params or {}
            name = params.get("name", "")
            arguments = params.get("arguments", {})

            result = await call_tool(name, arguments)

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
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {request.method}",
                },
            )

    except Exception as e:
        return MCPResponse(
            id=request.id,
            error={
                "code": -32603,
                "message": str(e),
            },
        )


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
