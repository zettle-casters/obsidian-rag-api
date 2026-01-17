from typing import List

from pydantic import BaseModel

class NoteRecord(BaseModel):
    # Note tree root
    note_id : str
    path : str
    title : str
    block_ids : List[str] | None # For notes with complex hierarchical block structure
    chunk_ids : List[str] | None # For notes without block structure(plain text)
    links_to_notes : List[str] | None

class BlockRecord(BaseModel):
    # Note tree internal node
    block_id : str
    note_id : str
    index : int
    parent_block_id : str | None # None for root children
    child_block_ids : List[str] | None
    chunk_ids : List[str] | None # For terminal node parents

class ChunkRecord(BaseModel):
    # Note tree leaf
    chunk_id : str
    note_id : str
    block_id : str | None # For notes without block structure(plain text)
    index : int
    text : str
    links_to_chunks : List[str] | None

class NoteLinkRecord(BaseModel):
    from_note_id : str
    to_note_id : str
    link_type : str

class ChunkLinkRecord(BaseModel):
    from_chunk_id : str
    to_chunk_id : str
    link_type : str
