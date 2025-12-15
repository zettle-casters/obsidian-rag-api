"""FastAPI application with /agent endpoint for Obsidian RAG."""

import uuid
import tempfile
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .agent import run_agent_with_vault, get_agent_for_vault
from .vault_manager import upload_vault, upload_vault_with_progress, get_vault_manager, list_vaults, restore_vaults_from_metadata
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    # Restore vaults from metadata on startup
    restore_vaults_from_metadata()
    yield


app = FastAPI(
    title="Obsidian RAG Agent",
    description="RAG agent for querying Obsidian knowledge base with multi-vault support",
    version="0.2.0",
    lifespan=lifespan,
)

# Add CORS middleware for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentRequest(BaseModel):
    """Request model for the agent endpoint."""

    query: str
    vault_id: str
    thread_id: str | None = None


class AgentResponse(BaseModel):
    """Response model for the agent endpoint."""

    query: str
    reformulated_query: str
    answer: str
    notes_used: int
    max_depth_reached: int
    thread_id: str
    vault_id: str


class UploadResponse(BaseModel):
    """Response model for the upload endpoint."""

    vault_id: str
    message: str
    status: str


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/upload", response_model=UploadResponse)
async def upload_endpoint(
        file: UploadFile = File(...),
        include_paths: str = Form(""),
        exclude_paths: str = Form(""),
        chunk_size: int = Form(500),
):
    """
    Upload and initialize a vault from a ZIP file.

    Args:
        file: ZIP file containing the Obsidian vault
        include_paths: Comma-separated list of paths to include (optional)
        exclude_paths: Comma-separated list of paths to exclude (optional)
        chunk_size: Maximum chunk size in characters (default: 500)

    Returns:
        vault_id: UUID identifier for the uploaded vault
    """
    # Validate file type
    if not file.filename.endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Only ZIP files are supported. Please upload a .zip file.",
        )

    # Parse include/exclude paths
    include_list = [p.strip() for p in include_paths.split(",") if p.strip()]
    exclude_list = [p.strip() for p in exclude_paths.split(",") if p.strip()]

    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Upload and initialize vault
        vault_id = await upload_vault(
            zip_file_path=tmp_path,
            include_paths=include_list,
            exclude_paths=exclude_list,
            chunk_size=chunk_size,
        )

        # Clean up temporary file
        Path(tmp_path).unlink()

        return UploadResponse(
            vault_id=vault_id,
            message=f"Vault uploaded successfully. Use this vault_id in your queries.",
            status="success",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload vault: {str(e)}")


@app.post("/upload/stream")
async def upload_stream_endpoint(
        file: UploadFile = File(...),
        vault_name: str = Form(""),
        include_paths: str = Form(""),
        exclude_paths: str = Form(""),
        chunk_size: int = Form(500),
):
    """
    Upload and initialize a vault from a ZIP file with streaming progress updates.

    Args:
        file: ZIP file containing the Obsidian vault
        vault_name: Custom name for the vault (optional, defaults to filename)
        include_paths: Comma-separated list of paths to include (optional)
        exclude_paths: Comma-separated list of paths to exclude (optional)
        chunk_size: Maximum chunk size in characters (default: 500)

    Returns:
        Server-Sent Events stream with progress updates
    """
    # Validate file type
    if not file.filename.endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Only ZIP files are supported. Please upload a .zip file.",
        )

    # Parse include/exclude paths
    include_list = [p.strip() for p in include_paths.split(",") if p.strip()]
    exclude_list = [p.strip() for p in exclude_paths.split(",") if p.strip()]

    async def generate():
        tmp_path = None
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name

            # Use custom vault name or fallback to filename without .zip
            final_vault_name = vault_name.strip() if vault_name.strip() else file.filename.replace(".zip", "")

            # Stream progress updates
            async for progress_update in upload_vault_with_progress(
                    zip_file_path=tmp_path,
                    vault_name=final_vault_name,
                    include_paths=include_list,
                    exclude_paths=exclude_list,
                    chunk_size=chunk_size,
            ):
                yield f"data: {json.dumps(progress_update, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_event = {
                "stage": "error",
                "progress": 0,
                "message": f"Upload failed: {str(e)}",
                "error": str(e),
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        finally:
            # Clean up temporary file
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.get("/vaults")
async def list_vaults_endpoint():
    """List all uploaded vaults."""
    vaults = list_vaults()
    return {"vaults": vaults, "count": len(vaults)}


@app.post("/agent", response_model=AgentResponse)
async def agent_endpoint(request: AgentRequest):
    """
    Query the Obsidian RAG agent for a specific vault.

    The agent will:
    1. Reformulate the query for better search
    2. Search for relevant notes in the specified vault
    3. Check if context is sufficient
    4. Recursively extend context by exploring linked notes if needed
    5. Generate a final answer based on accumulated knowledge
    """
    # Verify vault exists
    if not get_vault_manager(request.vault_id):
        raise HTTPException(
            status_code=404,
            detail=f"Vault {request.vault_id} not found. Please upload a vault first using /upload.",
        )

    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        result = await run_agent_with_vault(
            query=request.query,
            vault_id=request.vault_id,
            thread_id=thread_id,
        )
        return AgentResponse(
            query=result["query"],
            reformulated_query=result["reformulated_query"],
            answer=result["answer"],
            notes_used=result["notes_used"],
            max_depth_reached=result["max_depth_reached"],
            thread_id=thread_id,
            vault_id=request.vault_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/stream")
async def agent_stream_endpoint(request: AgentRequest):
    """
    Stream the agent's progress as it processes the query.

    Returns Server-Sent Events with status updates.
    """
    # Verify vault exists
    if not get_vault_manager(request.vault_id):
        raise HTTPException(
            status_code=404,
            detail=f"Vault {request.vault_id} not found. Please upload a vault first using /upload.",
        )

    thread_id = request.thread_id or str(uuid.uuid4())

    async def generate():
        agent = get_agent_for_vault(request.vault_id)

        initial_state = {
            "original_query": request.query,
            "vault_id": request.vault_id,
            "messages": [],
            "reformulated_query": "",
            "search_results": [],
            "knowledge_base": [],
            "notes_to_explore": [],
            "explored_notes": set(),
            "current_depth": 0,
            "final_answer": "",
            "status": "started",
        }

        config = {"configurable": {"thread_id": thread_id}}

        async for event in stream_events(agent, initial_state, config, request):
            yield event

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


async def stream_events(agent, initial_state, config, request):
    """Stream LangGraph agent events with proper handling."""
    thread_id = config["configurable"].get("thread_id")
    vault_id = request.vault_id

    yield f"data: {json.dumps({'status': 'started', 'thread_id': thread_id})}\n\n"

    async for chunk in agent.astream_events(initial_state, config, version="v2"):
        kind = chunk.get("event")

        # Стриминг токенов от LLM
        if kind == "on_chat_model_stream":
            content = chunk.get("data", {}).get("chunk", {})
            if hasattr(content, "content") and content.content:
                yield f"data: {json.dumps({
                    'type': 'token',
                    'content': content.content,
                    'thread_id': thread_id
                }, ensure_ascii=False)}\n\n"

        # Обновления состояния нод
        elif kind == "on_chain_end":
            metadata = chunk.get("metadata", {})
            node_name = metadata.get("langgraph_node")
            if node_name:
                output = chunk.get("data", {}).get("output", {})
                serializable_output = serialize_output(output)
                yield f"data: {json.dumps({
                    'type': 'node_complete',
                    'node': node_name,
                    'output': serializable_output,
                    'thread_id': thread_id
                }, ensure_ascii=False)}\n\n"

    yield f"data: {json.dumps({'status': 'complete', 'thread_id': thread_id, 'vault_id': vault_id})}\n\n"


def extract_token_content(token_obj):
    """Извлекает чистый текст из любого токена LangChain."""
    if hasattr(token_obj, 'content') and token_obj.content:
        return token_obj.content
    if hasattr(token_obj, 'text') and token_obj.text:
        return token_obj.text
    if hasattr(token_obj, 'message') and token_obj.message.content:
        return token_obj.message.content

    token_str = str(token_obj)
    if "content='" in token_str:
        start = token_str.find("content='") + 9
        end = token_str.find("'", start)
        return token_str[start:end]

    return ""

def serialize_output(output):
    """Безопасная сериализация вывода ноды."""
    if not isinstance(output, dict):
        return str(output)

    serializable = {}
    for key, value in output.items():
        if isinstance(value, set):
            serializable[key] = list(value)
        elif value is None:
            serializable[key] = None
        elif isinstance(value, (str, int, float, bool)):
            serializable[key] = value
        elif isinstance(value, dict):
            serializable[key] = serialize_output(value)
        else:
            serializable[key] = str(value)
    return serializable


def main():
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "obsidian_rag_mcp.api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
