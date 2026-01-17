"""FastAPI application with /agent endpoint for Obsidian RAG."""

import uuid
import tempfile
import json
import asyncio
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel

from .agent import run_agent_with_vault, get_agent_for_vault
from .auth import (
    build_google_auth_redirect,
    create_session,
    ensure_demo_user,
    exchange_code_for_user,
    get_current_user,
    get_or_create_mcp_token,
    rotate_mcp_token,
)
from .db import get_db, init_db, SessionLocal
from obsidian_retriever.utils.hash import text_hash

from .models import Vault, VaultFile, Session as DbSession, User
from .vault_manager import (
    upload_vault,
    upload_vault_with_progress,
    get_or_create_vault_manager,
    list_vaults,
    restore_vaults_from_metadata,
)
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    init_db()
    with SessionLocal() as db:
        ensure_demo_user(db)
        restore_vaults_from_metadata(db)
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
    allow_origins=settings.cors_origins,
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


class McpTokenResponse(BaseModel):
    """Response model for MCP token."""

    token: str
    created_at: str
    last_used_at: str | None = None


class SyncChange(BaseModel):
    type: Literal["upsert", "delete", "rename"]
    path: str | None = None
    content: str | None = None
    from_path: str | None = None
    to_path: str | None = None


class SyncRequest(BaseModel):
    vault_id: str
    changes: list[SyncChange]


class SyncResponse(BaseModel):
    applied: int
    skipped: int
    errors: list[str]


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def _serialize_user(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "avatar_url": user.avatar_url,
        "is_demo": user.is_demo,
    }


def _get_vault_or_404(db, user: User, vault_id: str) -> Vault:
    vault = (
        db.query(Vault)
        .filter(Vault.id == vault_id, Vault.user_id == user.id)
        .first()
    )
    if vault is None:
        raise HTTPException(
            status_code=404,
            detail=f"Vault {vault_id} not found for current user.",
        )
    return vault


@app.get("/auth/me")
async def auth_me(user: User = Depends(get_current_user)):
    return {"user": _serialize_user(user)}


@app.get("/auth/google/login")
async def google_login(return_to: str | None = None, db=Depends(get_db)):
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(status_code=500, detail="Google OAuth is not configured")
    redirect_target = return_to or settings.ui_base_url
    if not redirect_target.startswith(settings.ui_base_url):
        redirect_target = settings.ui_base_url
    url = build_google_auth_redirect(redirect_target, db)
    return RedirectResponse(url)


@app.get("/auth/google/callback")
async def google_callback(request: Request, code: str | None = None, state: str | None = None, db=Depends(get_db)):
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing OAuth code or state")

    user, return_to = await exchange_code_for_user(code, state, db)
    session = create_session(user, db)

    response = RedirectResponse(return_to or settings.ui_base_url)
    response.set_cookie(
        settings.session_cookie_name,
        session.token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=settings.session_ttl_days * 24 * 60 * 60,
        path="/",
    )
    return response


@app.post("/auth/logout")
async def logout(request: Request, db=Depends(get_db)):
    token = request.cookies.get(settings.session_cookie_name)
    if token:
        db.query(DbSession).filter(DbSession.token == token).delete()
        db.commit()
    response = JSONResponse({"status": "ok"})
    response.delete_cookie(settings.session_cookie_name, path="/")
    return response


@app.get("/auth/mcp-token", response_model=McpTokenResponse)
async def auth_mcp_token(user: User = Depends(get_current_user), db=Depends(get_db)):
    if user.is_demo:
        raise HTTPException(status_code=403, detail="Login required to access MCP token")
    record = get_or_create_mcp_token(user, db)
    return McpTokenResponse(
        token=record.token,
        created_at=record.created_at.isoformat(),
        last_used_at=record.last_used_at.isoformat() if record.last_used_at else None,
    )


@app.post("/auth/mcp-token/rotate", response_model=McpTokenResponse)
async def auth_rotate_mcp_token(user: User = Depends(get_current_user), db=Depends(get_db)):
    if user.is_demo:
        raise HTTPException(status_code=403, detail="Login required to access MCP token")
    record = rotate_mcp_token(user, db)
    return McpTokenResponse(
        token=record.token,
        created_at=record.created_at.isoformat(),
        last_used_at=record.last_used_at.isoformat() if record.last_used_at else None,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_endpoint(
        file: UploadFile = File(...),
        vault_name: str = Form(""),
        include_paths: str = Form(""),
        exclude_paths: str = Form(""),
        chunk_size: int = Form(500),
        db=Depends(get_db),
        user: User = Depends(get_current_user),
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
        final_vault_name = vault_name.strip() if vault_name.strip() else file.filename.replace(".zip", "")
        vault_id = await upload_vault(
            zip_file_path=tmp_path,
            db=db,
            owner_id=user.id,
            include_paths=include_list,
            exclude_paths=exclude_list,
            chunk_size=chunk_size,
            vault_name=final_vault_name,
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
        user: User = Depends(get_current_user),
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
        db_session = SessionLocal()
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
                    db=db_session,
                    owner_id=user.id,
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
            db_session.close()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/vaults")
async def list_vaults_endpoint(
    db=Depends(get_db),
    user: User = Depends(get_current_user),
):
    """List all uploaded vaults."""
    vaults = list_vaults(db, user.id)
    return {"vaults": vaults, "count": len(vaults)}


@app.post("/sync/push", response_model=SyncResponse)
async def sync_push_endpoint(
    request: SyncRequest,
    db=Depends(get_db),
    user: User = Depends(get_current_user),
):
    vault = _get_vault_or_404(db, user, request.vault_id)
    manager = get_or_create_vault_manager(request.vault_id)

    applied = 0
    skipped = 0
    errors: list[str] = []

    for change in request.changes:
        try:
            if change.type == "upsert":
                if not change.path or change.content is None:
                    raise ValueError("Missing path/content for upsert")

                content_hash = text_hash(change.content)
                record = (
                    db.query(VaultFile)
                    .filter(VaultFile.vault_id == vault.id, VaultFile.path == change.path)
                    .first()
                )
                if record and record.content_hash == content_hash:
                    skipped += 1
                    continue

                manager.upsert_note_from_content(change.path, change.content, max_chunk_size=vault.chunk_size or 500)

                if record:
                    record.content_hash = content_hash
                else:
                    db.add(VaultFile(vault_id=vault.id, path=change.path, content_hash=content_hash))
                    vault.notes_count = (vault.notes_count or 0) + 1

                db.commit()
                applied += 1

            elif change.type == "delete":
                if not change.path:
                    raise ValueError("Missing path for delete")

                record = (
                    db.query(VaultFile)
                    .filter(VaultFile.vault_id == vault.id, VaultFile.path == change.path)
                    .first()
                )

                manager.delete_note_by_path(change.path)

                if record:
                    db.delete(record)
                    vault.notes_count = max(0, (vault.notes_count or 0) - 1)

                db.commit()
                applied += 1

            elif change.type == "rename":
                if not change.from_path or not change.to_path or change.content is None:
                    raise ValueError("Missing from_path/to_path/content for rename")

                old_record = (
                    db.query(VaultFile)
                    .filter(VaultFile.vault_id == vault.id, VaultFile.path == change.from_path)
                    .first()
                )

                manager.delete_note_by_path(change.from_path)

                if old_record:
                    db.delete(old_record)

                content_hash = text_hash(change.content)
                manager.upsert_note_from_content(change.to_path, change.content, max_chunk_size=vault.chunk_size or 500)

                new_record = (
                    db.query(VaultFile)
                    .filter(VaultFile.vault_id == vault.id, VaultFile.path == change.to_path)
                    .first()
                )
                if new_record:
                    new_record.content_hash = content_hash
                else:
                    db.add(VaultFile(vault_id=vault.id, path=change.to_path, content_hash=content_hash))
                    if not old_record:
                        vault.notes_count = (vault.notes_count or 0) + 1

                db.commit()
                applied += 1

        except Exception as exc:
            db.rollback()
            errors.append(f"{change.type}:{change.path or change.from_path}->{change.to_path}: {exc}")

    return SyncResponse(applied=applied, skipped=skipped, errors=errors)


@app.post("/agent", response_model=AgentResponse)
async def agent_endpoint(
    request: AgentRequest,
    db=Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Query the Obsidian RAG agent for a specific vault.

    The agent will:
    1. Reformulate the query for better search
    2. Search for relevant notes in the specified vault
    3. Check if context is sufficient
    4. Recursively extend context by exploring linked notes if needed
    5. Generate a final answer based on accumulated knowledge
    """
    _get_vault_or_404(db, user, request.vault_id)
    get_or_create_vault_manager(request.vault_id)

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
async def agent_stream_endpoint(
    request: AgentRequest,
    db=Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Stream the agent's progress as it processes the query.

    Returns Server-Sent Events with status updates.
    """
    _get_vault_or_404(db, user, request.vault_id)
    get_or_create_vault_manager(request.vault_id)

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

        # Начало работы ноды
        elif kind == "on_chain_start":
            metadata = chunk.get("metadata", {})
            node_name = metadata.get("langgraph_node")
            if node_name:
                yield f"data: {json.dumps({
                    'type': 'node_start',
                    'node': node_name,
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


def find_project_root() -> Path:
    """
    Find the project root by searching for the workspace root.

    Returns:
        Path to the project root
    """
    current = Path(__file__).resolve()

    # Prefer the workspace root that contains tests (avoid submodule .git files).
    for parent in [current] + list(current.parents):
        if (parent / "obsidian_rag_tests").exists():
            return parent

    # Search up the directory tree for a real git directory.
    for parent in [current] + list(current.parents):
        # Check for .git directory (most reliable indicator)
        git_dir = parent / ".git"
        if git_dir.exists() and git_dir.is_dir():
            return parent

    # Fallback: go up 4 levels from current file
    return Path(__file__).parent.parent.parent.parent


@app.post("/tests/run")
async def run_tests_endpoint():
    """
    Run all tests from the obsidian_rag_tests submodule with streaming progress.

    Returns:
        Server-Sent Events stream with detailed test execution progress for each test case
    """
    async def generate():
        try:
            # Import test modules
            import sys
            project_root = find_project_root()
            tests_path = project_root / "obsidian_rag_tests"

            if not tests_path.exists():
                error_event = {
                    "type": "error",
                    "message": "obsidian_rag_tests submodule not found",
                    "error": f"Path {tests_path} does not exist. Project root: {project_root}"
                }
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
                return

            # Add tests path to sys.path
            tests_src = str(tests_path / "src")
            if tests_src not in sys.path:
                sys.path.insert(0, tests_src)

            # Import required modules
            from obsidian_rag_api.llm import check_relevance, should_extend_context

            # Notify start
            yield f"data: {json.dumps({'type': 'suite_start', 'message': 'Начало выполнения тестов'}, ensure_ascii=False)}\n\n"

            # ==== Test Suite 1: check_relevance ====
            yield f"data: {json.dumps({'type': 'test_suite_start', 'suite': 'check_relevance', 'name': 'Проверка релевантности заметок'}, ensure_ascii=False)}\n\n"

            check_relevance_path = tests_path / "src" / "obsidian_rag_tests" / "test_check_relevance"
            cases_file = check_relevance_path / "test_cases.json"

            with open(cases_file, 'r', encoding='utf-8') as f:
                check_relevance_data = json.load(f)

            check_relevance_cases = check_relevance_data.get("cases", [])
            check_relevance_results = []

            for i, case in enumerate(check_relevance_cases):
                test_id = case.get("id")
                query = case.get("query")
                note = case["note"]
                expected = case.get("expected_value")
                complexity = case.get("complexity")

                # Notify test start
                yield f"data: {json.dumps({'type': 'test_start', 'suite': 'check_relevance', 'id': test_id, 'query': query[:50] + '...' if len(query) > 50 else query, 'complexity': complexity}, ensure_ascii=False)}\n\n"

                # Run test
                try:
                    result = await check_relevance(query, note["content"], note["title"])
                    passed = (bool(result) == bool(expected)) if expected is not None else None

                    check_relevance_results.append({
                        "id": test_id,
                        "relevant": bool(result),
                        "expected_value": expected,
                        "complexity": complexity,
                    })

                    # Notify test complete
                    yield f"data: {json.dumps({'type': 'test_complete', 'suite': 'check_relevance', 'id': test_id, 'result': bool(result), 'expected': expected, 'passed': passed, 'complexity': complexity}, ensure_ascii=False)}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'type': 'test_error', 'suite': 'check_relevance', 'id': test_id, 'error': str(e)}, ensure_ascii=False)}\n\n"

            # Save check_relevance results
            results_file = check_relevance_path / "test_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({"results": check_relevance_results}, f, ensure_ascii=False, indent=2)

            # Calculate stats for check_relevance
            stats = {}
            for case in check_relevance_results:
                complexity = str(case.get("complexity"))
                if complexity not in stats:
                    stats[complexity] = {"total": 0, "passed": 0, "failed": 0, "unknown": 0}

                stats[complexity]["total"] += 1
                expected = case.get("expected_value")
                if expected is None:
                    stats[complexity]["unknown"] += 1
                else:
                    passed = bool(case.get("relevant")) == bool(expected)
                    if passed:
                        stats[complexity]["passed"] += 1
                    else:
                        stats[complexity]["failed"] += 1

            # Save stats
            stats_file = check_relevance_path / "test_stats.txt"
            with open(stats_file, 'w', encoding='utf-8') as f:
                for level, vals in stats.items():
                    f.write(f"Complexity: {level}\n")
                    f.write(f"  total: {vals['total']}\n")
                    f.write(f"  passed: {vals['passed']}\n")
                    f.write(f"  failed: {vals['failed']}\n")
                    f.write(f"  unknown: {vals['unknown']}\n\n")

            yield f"data: {json.dumps({'type': 'test_suite_complete', 'suite': 'check_relevance', 'stats': stats}, ensure_ascii=False)}\n\n"

            # ==== Test Suite 2: should_extend_context ====
            yield f"data: {json.dumps({'type': 'test_suite_start', 'suite': 'should_extend_context', 'name': 'Проверка необходимости расширения контекста'}, ensure_ascii=False)}\n\n"

            extend_context_path = tests_path / "src" / "obsidian_rag_tests" / "test_should_extend_context"
            cases_file = extend_context_path / "test_cases.json"

            with open(cases_file, 'r', encoding='utf-8') as f:
                extend_context_data = json.load(f)

            extend_context_cases = extend_context_data.get("cases", [])
            extend_context_results = []

            for i, case in enumerate(extend_context_cases):
                test_id = case.get("id")
                query = case.get("query")
                context = case.get("context", [])
                expected = case.get("expected_value")
                complexity = case.get("complexity")

                # Notify test start
                yield f"data: {json.dumps({'type': 'test_start', 'suite': 'should_extend_context', 'id': test_id, 'query': query[:50] + '...' if len(query) > 50 else query, 'complexity': complexity}, ensure_ascii=False)}\n\n"

                # Run test
                try:
                    result = await should_extend_context(query, context)
                    passed = (bool(result) == bool(expected)) if expected is not None else None

                    extend_context_results.append({
                        "id": test_id,
                        "needs_extension": bool(result),
                        "expected_value": expected,
                        "complexity": complexity,
                    })

                    # Notify test complete
                    yield f"data: {json.dumps({'type': 'test_complete', 'suite': 'should_extend_context', 'id': test_id, 'result': bool(result), 'expected': expected, 'passed': passed, 'complexity': complexity}, ensure_ascii=False)}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'type': 'test_error', 'suite': 'should_extend_context', 'id': test_id, 'error': str(e)}, ensure_ascii=False)}\n\n"

            # Save should_extend_context results
            results_file = extend_context_path / "test_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({"results": extend_context_results}, f, ensure_ascii=False, indent=2)

            # Calculate stats for should_extend_context
            stats = {}
            for case in extend_context_results:
                complexity = str(case.get("complexity"))
                if complexity not in stats:
                    stats[complexity] = {"total": 0, "passed": 0, "failed": 0, "unknown": 0}

                stats[complexity]["total"] += 1
                expected = case.get("expected_value")
                if expected is None:
                    stats[complexity]["unknown"] += 1
                else:
                    passed = bool(case.get("needs_extension")) == bool(expected)
                    if passed:
                        stats[complexity]["passed"] += 1
                    else:
                        stats[complexity]["failed"] += 1

            # Save stats
            stats_file = extend_context_path / "test_stats.txt"
            with open(stats_file, 'w', encoding='utf-8') as f:
                for level, vals in stats.items():
                    f.write(f"Complexity: {level}\n")
                    f.write(f"  total: {vals['total']}\n")
                    f.write(f"  passed: {vals['passed']}\n")
                    f.write(f"  failed: {vals['failed']}\n")
                    f.write(f"  unknown: {vals['unknown']}\n\n")

            yield f"data: {json.dumps({'type': 'test_suite_complete', 'suite': 'should_extend_context', 'stats': stats}, ensure_ascii=False)}\n\n"

            # Send final completion
            yield f"data: {json.dumps({'type': 'suite_complete', 'message': 'Все тесты завершены'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_event = {
                "type": "error",
                "message": f"Ошибка при запуске тестов: {str(e)}",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/tests/results")
async def get_test_results():
    """
    Get the latest test results from both test suites.

    Returns:
        JSON with test results and statistics for both test suites
    """
    try:
        # Find the project root
        project_root = find_project_root()
        tests_path = project_root / "obsidian_rag_tests"

        if not tests_path.exists():
            raise HTTPException(
                status_code=404,
                detail="obsidian_rag_tests submodule not found"
            )

        results = {}

        # Read check_relevance results
        check_relevance_results_path = tests_path / "src" / "obsidian_rag_tests" / "test_check_relevance" / "test_results.json"
        check_relevance_stats_path = tests_path / "src" / "obsidian_rag_tests" / "test_check_relevance" / "test_stats.txt"

        if check_relevance_results_path.exists():
            with open(check_relevance_results_path, 'r', encoding='utf-8') as f:
                results['check_relevance'] = json.load(f)

        if check_relevance_stats_path.exists():
            with open(check_relevance_stats_path, 'r', encoding='utf-8') as f:
                results['check_relevance_stats'] = f.read()

        # Read should_extend_context results
        extend_context_results_path = tests_path / "src" / "obsidian_rag_tests" / "test_should_extend_context" / "test_results.json"
        extend_context_stats_path = tests_path / "src" / "obsidian_rag_tests" / "test_should_extend_context" / "test_stats.txt"

        if extend_context_results_path.exists():
            with open(extend_context_results_path, 'r', encoding='utf-8') as f:
                results['should_extend_context'] = json.load(f)

        if extend_context_stats_path.exists():
            with open(extend_context_stats_path, 'r', encoding='utf-8') as f:
                results['should_extend_context_stats'] = f.read()

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No test results found. Run tests first using /tests/run"
            )

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
