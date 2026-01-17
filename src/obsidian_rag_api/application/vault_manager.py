"""Vault management with multi-tenant support via UUID."""

import uuid
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, AsyncGenerator
from datetime import datetime

from obsidian_retriever.manager import KnowledgeBaseManager
from obsidian_retriever.schemas import NoteRecord

from ..infrastructure.config import settings
from ..domain.models import Vault
from sqlalchemy.orm import Session


# Global registry: vault_id -> KnowledgeBaseManager
_vault_managers: dict[str, KnowledgeBaseManager] = {}


def create_vault_manager(vault_id: str) -> KnowledgeBaseManager:
    """Create a new KnowledgeBaseManager for a specific vault."""
    manager = KnowledgeBaseManager(
        vault_id=vault_id,
        db_url=settings.neo4j_url,
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        prefer_grpc=settings.qdrant_prefer_grpc,
        model_name=settings.embeddings_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    _vault_managers[vault_id] = manager
    return manager


def get_vault_manager(vault_id: str) -> Optional[KnowledgeBaseManager]:
    """Get existing vault manager by ID."""
    return _vault_managers.get(vault_id)


def get_or_create_vault_manager(vault_id: str) -> KnowledgeBaseManager:
    """Get existing manager or create a new one."""
    manager = get_vault_manager(vault_id)
    if manager is None:
        manager = create_vault_manager(vault_id)
    return manager


async def upload_vault(
    zip_file_path: str,
    db: Session,
    owner_id: str,
    include_paths: list[str] = None,
    exclude_paths: list[str] = None,
    chunk_size: int = 500,
    vault_name: str | None = None,
) -> str:
    """
    Upload and initialize a vault from a ZIP file.

    Returns:
        vault_id: UUID identifier for the uploaded vault
    """
    # Generate unique vault ID
    vault_id = str(uuid.uuid4())

    # Create manager for this vault
    manager = create_vault_manager(vault_id)

    # Initialize vault from ZIP
    include_paths = include_paths or []
    exclude_paths = exclude_paths or []

    manager.init_vault_from_zip(
        zip_path=zip_file_path,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        max_chunk_size=chunk_size,
    )

    final_name = vault_name if vault_name else Path(zip_file_path).stem
    db.add(Vault(
        id=vault_id,
        user_id=owner_id,
        name=final_name,
        created_at=datetime.now(),
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        chunk_size=chunk_size,
        notes_count=0,
    ))
    db.commit()

    return vault_id


async def upload_vault_with_progress(
    zip_file_path: str,
    db: Session,
    owner_id: str,
    vault_name: str = None,
    include_paths: list[str] = None,
    exclude_paths: list[str] = None,
    chunk_size: int = 500,
) -> AsyncGenerator[dict, None]:
    """
    Upload and initialize a vault from a ZIP file with progress updates.

    Args:
        zip_file_path: Path to the ZIP file
        vault_name: Custom name for the vault (optional, defaults to filename)
        include_paths: List of paths to include
        exclude_paths: List of paths to exclude
        chunk_size: Maximum chunk size in characters

    Yields progress updates as dict with:
        - stage: str - current stage name
        - progress: float - progress percentage (0-100)
        - message: str - human-readable message
        - vault_id: str - UUID (only in final message)
    """
    import asyncio
    import time
    from obsidian_parser.parse import load_obsidian_with_filters

    # Generate unique vault ID
    vault_id = str(uuid.uuid4())

    def format_time(seconds):
        """Format seconds to human readable string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            return f"{hours}h {mins}m"

    try:
        # Stage 1: Extracting ZIP
        stage_start = time.time()
        yield {
            "stage": "extracting",
            "progress": 0,
            "message": "Extracting ZIP archive...",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_file_path, "r") as zf:
                file_list = zf.namelist()
                total_files = len(file_list)

                for i, file in enumerate(file_list):
                    zf.extract(file, tmpdir)
                    if i % max(1, total_files // 20) == 0 and i > 0:  # Update every ~5%
                        elapsed = time.time() - stage_start
                        avg_per_item = elapsed / i
                        remaining_items = total_files - i
                        eta = avg_per_item * remaining_items

                        progress = (i / total_files) * 30  # 0-30%
                        yield {
                            "stage": "extracting",
                            "progress": progress,
                            "message": f"Extracted {i}/{total_files} files...",
                            "elapsed": format_time(elapsed),
                            "eta": format_time(eta),
                        }
                await asyncio.sleep(0)

            elapsed = time.time() - stage_start
            yield {
                "stage": "extracting",
                "progress": 30,
                "message": f"Extracted {total_files} files",
                "elapsed": format_time(elapsed),
            }

            # Stage 2: Parsing notes
            yield {
                "stage": "parsing",
                "progress": 30,
                "message": "Parsing markdown files...",
            }

            include_paths = include_paths or []
            exclude_paths = exclude_paths or []

            notes_data = load_obsidian_with_filters(
                vault_path=tmpdir,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
            )

            total_notes = len(notes_data)
            yield {
                "stage": "parsing",
                "progress": 50,
                "message": f"Parsed {total_notes} notes",
            }

            # Stage 3: Initializing database
            yield {
                "stage": "initializing",
                "progress": 50,
                "message": "Creating vault manager...",
            }

            manager = create_vault_manager(vault_id)

            yield {
                "stage": "processing",
                "progress": 55,
                "message": "Processing notes and building graph...",
            }

            # Build path mappings
            path_to_note_id = {}
            for note_dict in notes_data:
                path = note_dict["path"]
                note_id = manager._qualify_note_id(path)
                path_to_note_id[path] = note_id

            # Process chunks and build graph structures
            stage_start = time.time()
            note_chunks = {}
            chunk_links_raw = {}
            anchor_to_chunk = {}
            chunk_to_note = {}

            for i, note_dict in enumerate(notes_data):
                path = note_dict["path"]
                note_id = path_to_note_id[path]

                chunks_for_note, links_for_note, anchors_for_note = manager._flatten_note_chunks(
                    note_id, note_dict
                )

                note_chunks[note_id] = chunks_for_note

                for chunk_id, links in links_for_note.items():
                    chunk_links_raw[chunk_id] = links
                    chunk_to_note[chunk_id] = note_id

                for anchor_title, chunk_id in anchors_for_note.items():
                    anchor_to_chunk[(note_id, anchor_title)] = chunk_id

                if i % max(1, total_notes // 10) == 0 and i > 0:  # Update every ~10%
                    elapsed = time.time() - stage_start
                    avg_per_item = elapsed / i
                    remaining_items = total_notes - i
                    eta = avg_per_item * remaining_items

                    progress = 55 + (i / total_notes) * 20  # 55-75%
                    yield {
                        "stage": "processing",
                        "progress": progress,
                        "message": f"Processed {i}/{total_notes} notes...",
                        "elapsed": format_time(elapsed),
                        "eta": format_time(eta),
                    }
                await asyncio.sleep(0)

            elapsed = time.time() - stage_start
            yield {
                "stage": "processing",
                "progress": 75,
                "message": f"Processed {total_notes} notes",
                "elapsed": format_time(elapsed),
            }

            # Stage 4: Storing in databases
            stage_start = time.time()
            yield {
                "stage": "storing",
                "progress": 75,
                "message": "Storing notes in graph and vector databases...",
            }

            for i, note_dict in enumerate(notes_data):
                path = note_dict["path"]
                name = note_dict["name"]
                note_id = path_to_note_id[path]
                chunks_for_note = note_chunks[note_id]

                note_record = NoteRecord(
                    note_id=note_id,
                    path=path,
                    title=name,
                    block_ids=None,
                    chunk_ids=[c.chunk_id for c in chunks_for_note],
                    links_to_notes=None,
                )

                manager.add_note(note_record, chunks_for_note)

                if i % max(1, total_notes // 10) == 0 and i > 0:
                    elapsed = time.time() - stage_start
                    avg_per_item = elapsed / i
                    remaining_items = total_notes - i
                    eta = avg_per_item * remaining_items

                    progress = 75 + (i / total_notes) * 15  # 75-90%
                    yield {
                        "stage": "storing",
                        "progress": progress,
                        "message": f"Stored {i}/{total_notes} notes...",
                        "elapsed": format_time(elapsed),
                        "eta": format_time(eta),
                    }
                await asyncio.sleep(0)

            elapsed = time.time() - stage_start
            yield {
                "stage": "storing",
                "progress": 90,
                "message": f"Stored {total_notes} notes",
                "elapsed": format_time(elapsed),
            }

            # Stage 5: Building links
            yield {
                "stage": "linking",
                "progress": 90,
                "message": "Building note links...",
            }

            manager._build_links_from_wikilinks(
                chunk_links_raw=chunk_links_raw,
                chunk_to_note=chunk_to_note,
                path_to_note_id=path_to_note_id,
                anchor_to_chunk=anchor_to_chunk,
            )

            yield {
                "stage": "linking",
                "progress": 100,
                "message": "Link building complete",
            }

        # Save metadata
        final_name = vault_name if vault_name else Path(zip_file_path).stem
        db.add(Vault(
            id=vault_id,
            user_id=owner_id,
            name=final_name,
            created_at=datetime.now(),
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            chunk_size=chunk_size,
            notes_count=total_notes,
        ))
        db.commit()

        # Final success message
        yield {
            "stage": "complete",
            "progress": 100,
            "message": "Vault uploaded successfully",
            "vault_id": vault_id,
            "notes_count": total_notes,
        }

    except Exception as e:
        yield {
            "stage": "error",
            "progress": 0,
            "message": f"Error: {str(e)}",
            "error": str(e),
        }


def list_vaults(db: Session, owner_id: str) -> list[dict]:
    """List all registered vaults with metadata for a specific user."""
    vaults = (
        db.query(Vault)
        .filter(Vault.user_id == owner_id)
        .order_by(Vault.created_at.desc())
        .all()
    )
    return [
        {
            "vault_id": vault.id,
            "name": vault.name,
            "created_at": vault.created_at.isoformat() if vault.created_at else None,
            "notes_count": vault.notes_count or 0,
        }
        for vault in vaults
    ]


def delete_vault(db: Session, vault_id: str, owner_id: str) -> bool:
    """Remove a vault from the registry."""
    deleted = (
        db.query(Vault)
        .filter(Vault.id == vault_id, Vault.user_id == owner_id)
        .delete()
    )
    if deleted:
        db.commit()
        _vault_managers.pop(vault_id, None)
        return True
    return False


def restore_vaults_from_metadata(db: Session) -> None:
    """
    Restore vault managers from saved metadata.
    Called on application startup.
    """
    vaults = db.query(Vault).all()
    for vault in vaults:
        try:
            create_vault_manager(vault.id)
            print(f"Restored vault {vault.id}: {vault.name} ({vault.notes_count or 0} notes)")
        except Exception as e:
            print(f"Failed to restore vault {vault.id}: {e}")
