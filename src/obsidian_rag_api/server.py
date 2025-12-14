"""MCP Server for Obsidian RAG with tools: read_note, search, extend_context_using_nearest."""

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pydantic import BaseModel

from obsidian_retriever.manager import KnowledgeBaseManager

from .config import settings
from .vault_manager import get_vault_manager


# Initialize MCP server
mcp_server = Server("obsidian-rag")


class ReadNoteInput(BaseModel):
    """Input schema for read_note tool."""

    vault_id: str
    note_id: str


class SearchInput(BaseModel):
    """Input schema for search tool."""

    vault_id: str
    query: str
    top_k: int = 5


class ExtendContextInput(BaseModel):
    """Input schema for extend_context_using_nearest tool."""

    vault_id: str
    note_id: str
    query: str
    current_depth: int = 0


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="read_note",
            description="Read a specific note by its ID from a vault. Returns the full content of the note with all chunks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "vault_id": {
                        "type": "string",
                        "description": "The UUID of the vault to read from",
                    },
                    "note_id": {
                        "type": "string",
                        "description": "The unique identifier of the note (usually the file path)",
                    },
                },
                "required": ["vault_id", "note_id"],
            },
        ),
        Tool(
            name="search",
            description="Search for notes by semantic similarity using vector search in a specific vault. Returns the most relevant notes for the given query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "vault_id": {
                        "type": "string",
                        "description": "The UUID of the vault to search in",
                    },
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant notes",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["vault_id", "query"],
            },
        ),
        Tool(
            name="extend_context_using_nearest",
            description="Extend context by exploring linked notes in a vault. For a given note, finds all connected notes and checks if they contain relevant information for the query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "vault_id": {
                        "type": "string",
                        "description": "The UUID of the vault",
                    },
                    "note_id": {
                        "type": "string",
                        "description": "The note ID to extend context from",
                    },
                    "query": {
                        "type": "string",
                        "description": "The original user query to check relevance against",
                    },
                    "current_depth": {
                        "type": "integer",
                        "description": "Current recursion depth (used internally)",
                        "default": 0,
                    },
                },
                "required": ["vault_id", "note_id", "query"],
            },
        ),
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    vault_id = arguments.get("vault_id")
    if not vault_id:
        return [TextContent(type="text", text="Error: vault_id is required")]

    retriever = get_vault_manager(vault_id)
    if not retriever:
        return [
            TextContent(
                type="text",
                text=f"Error: Vault {vault_id} not found. Please upload a vault first.",
            )
        ]

    if name == "read_note":
        return await handle_read_note(retriever, arguments)
    elif name == "search":
        return await handle_search(retriever, arguments)
    elif name == "extend_context_using_nearest":
        return await handle_extend_context(retriever, arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def handle_read_note(
    retriever: KnowledgeBaseManager, arguments: dict[str, Any]
) -> list[TextContent]:
    """Handle read_note tool call."""
    note_id = arguments.get("note_id", "")

    note = retriever.get_note(note_id)
    if note is None:
        return [TextContent(type="text", text=f"Note not found: {note_id}")]

    # Get all chunks for this note
    chunks = []
    if note.chunk_ids:
        for chunk_id in note.chunk_ids:
            chunk = retriever.get_chunk(chunk_id)
            if chunk:
                chunks.append({"chunk_id": chunk.chunk_id, "index": chunk.index, "text": chunk.text})

    result = {
        "note_id": note.note_id,
        "path": note.path,
        "title": note.title,
        "linked_notes": note.links_to_notes or [],
        "chunks": chunks,
        "full_text": "\n\n".join(c["text"] for c in chunks),
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def handle_search(
    retriever: KnowledgeBaseManager, arguments: dict[str, Any]
) -> list[TextContent]:
    """Handle search tool call."""
    query = arguments.get("query", "")
    top_k = arguments.get("top_k", settings.search_top_k)

    # Search notes by semantic similarity
    results = retriever.search_notes(query, top_k=top_k)

    search_results = []
    for note_record, score in results:
        # Get chunks for preview
        preview_chunks = []
        if note_record.chunk_ids:
            for chunk_id in note_record.chunk_ids[:3]:  # First 3 chunks for preview
                chunk = retriever.get_chunk(chunk_id)
                if chunk:
                    preview_chunks.append(chunk.text[:200])

        search_results.append(
            {
                "note_id": note_record.note_id,
                "title": note_record.title,
                "path": note_record.path,
                "score": score,
                "linked_notes": note_record.links_to_notes or [],
                "preview": " ... ".join(preview_chunks)[:500],
            }
        )

    return [
        TextContent(
            type="text", text=json.dumps(search_results, ensure_ascii=False, indent=2)
        )
    ]


async def handle_extend_context(
    retriever: KnowledgeBaseManager, arguments: dict[str, Any]
) -> list[TextContent]:
    """Handle extend_context_using_nearest tool call."""
    note_id = arguments.get("note_id", "")
    query = arguments.get("query", "")
    current_depth = arguments.get("current_depth", 0)

    # Check max recursion depth
    if current_depth >= settings.max_recursion_depth:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "status": "max_depth_reached",
                        "note_id": note_id,
                        "depth": current_depth,
                        "relevant_notes": [],
                    },
                    ensure_ascii=False,
                ),
            )
        ]

    # Get linked notes (both incoming and outgoing)
    outgoing_links, incoming_links = retriever.get_note_neighbors(note_id)

    linked_note_ids = set()
    for link in outgoing_links:
        linked_note_ids.add(link.to_note_id)
    for link in incoming_links:
        linked_note_ids.add(link.from_note_id)

    if not linked_note_ids:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "status": "no_linked_notes",
                        "note_id": note_id,
                        "relevant_notes": [],
                    },
                    ensure_ascii=False,
                ),
            )
        ]

    # For each linked note, check relevance using LLM
    from .llm import check_relevance_batch

    linked_notes_data = []
    for linked_id in linked_note_ids:
        note = retriever.get_note(linked_id)
        if note:
            # Get note content
            content_parts = []
            if note.chunk_ids:
                for chunk_id in note.chunk_ids:
                    chunk = retriever.get_chunk(chunk_id)
                    if chunk:
                        content_parts.append(chunk.text)

            linked_notes_data.append(
                {
                    "note_id": note.note_id,
                    "title": note.title,
                    "content": "\n\n".join(content_parts),
                    "linked_notes": note.links_to_notes or [],
                }
            )

    # Check relevance in parallel using cheap model
    relevance_results = await check_relevance_batch(query, linked_notes_data)

    relevant_notes = []
    notes_to_extend = []

    for note_data, is_relevant in zip(linked_notes_data, relevance_results):
        if is_relevant:
            relevant_notes.append(
                {
                    "note_id": note_data["note_id"],
                    "title": note_data["title"],
                    "content": note_data["content"],
                }
            )
        else:
            # Not directly relevant, but might have relevant linked notes
            notes_to_extend.append(note_data["note_id"])

    result = {
        "status": "success",
        "note_id": note_id,
        "depth": current_depth,
        "relevant_notes": relevant_notes,
        "notes_checked": len(linked_notes_data),
        "notes_to_extend_further": notes_to_extend if current_depth < settings.max_recursion_depth - 1 else [],
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


def main():
    """Run the MCP server."""
    asyncio.run(run_server())


async def run_server():
    """Run the MCP server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(read_stream, write_stream, mcp_server.create_initialization_options())


if __name__ == "__main__":
    main()
